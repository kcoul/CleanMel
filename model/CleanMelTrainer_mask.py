
from model.utils.base_cli import BaseCLI

import os
import data_loader
import torch
import math
import soundfile as sf
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import model.utils.general_steps as GS

from tqdm import tqdm
from typing import *
from torch import Tensor
from glob import glob
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from model.utils.metrics import cal_metrics_functional
from model.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TrainModule(pl.LightningModule):
    name: str 
    import_path: str = 'model.CleanMelTrainer_mask.TrainModule'

    def __init__(
        self,
        arch: nn.Module,
        input_stft: nn.Module,
        target_stft: Optional[nn.Module] = None,
        optimizer: Tuple[str, Dict[str, Any]] = ("AdamW", {
            "lr": 0.001,
            "weight_decay": 0.001
        }),
        lr_scheduler: Optional[Tuple[str, Dict[str, Any]]] = ('ReduceLROnPlateau', {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-4
        }),
        write_examples: int = 200,
        exp_name: str = "exp",
        log_eps=1e-5,
        metrics: List[str] = ['SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ', 'eSTOI'],
        output_path: Optional[str] = None, # use only for inference
        arch_ckpt: Optional[str] = None,
        vocos_ckpt: Optional[str] = None,
        vocos_config: Optional[str] = None
    ):
        super().__init__()

        args = locals().copy()  
        # save parameters to `self`
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)
            
        self.online = arch.online
        self.name = self.exp_name

        # Load pretrained model
        # CleanMel
        if arch_ckpt is not None:
            if "HF" in vocos_ckpt:
                # Load pretrained model by HuggingFace Hub
                from huggingface_hub import hf_hub_download
                REPO_ID = "WestlakeAudioLab/CleanMel"
                arch_id = arch_ckpt.split("!")[-1]
                arch_ckpt = hf_hub_download(repo_id=REPO_ID, filename=arch_id)
            self.arch.load_state_dict(torch.load(arch_ckpt, map_location='cpu'), strict=True)
        # Vocos
        if vocos_config is not None:
            if "HF" in vocos_ckpt:
                # Load pretrained model by HuggingFace Hub
                from huggingface_hub import hf_hub_download
                REPO_ID = "WestlakeAudioLab/CleanMel"
                vocos_id = vocos_ckpt.split("!")[-1]
                vocos_ckpt = hf_hub_download(repo_id=REPO_ID, filename=vocos_id)
            if self.online:
                from model.vocos.online.pretrained import Vocos
            else:
                from model.vocos.offline.pretrained import Vocos
            self.vocos = Vocos.from_hparams(config_path=vocos_config)
            self.vocos = Vocos.from_pretrained(None, model_path=vocos_ckpt, model=self.vocos)
            self.vocos.requires_grad_(False)
    
        self.val_cpu_metric_input = []
        self.val_wavs = []
        self.test_wavs = []
        self.sample_rate = self.target_stft.sample_rate  
        
    def on_train_start(self):
        """Called by PytorchLightning automatically at the start of training"""
        GS.on_train_start(self=self, exp_name=self.exp_name, model_name=self.name, num_chns=1, nfft=self.target_stft.n_fft, model_class_path=self.import_path)
         
    def safe_log(self, x):           
        return torch.log(torch.clip(x, min=self.log_eps))  
    
    def get_mrm_target(self, yr, x, X_norm=None):
        Y_target = self.target_stft(yr, X_norm)
        X_noisy = self.target_stft(x, X_norm)
        mrm = torch.sqrt(Y_target) / (torch.sqrt(X_noisy) + 1e-10)
        mrm = mrm.clamp(max=1)
        assert mrm.abs().max() <= 1, f"MRM max value: {mrm.abs().max()}"
        return mrm

    def get_mrm_pred(self, Y_hat, x, X_norm=None):
        X_noisy = self.target_stft(x, X_norm)
        Y_hat = Y_hat.squeeze()
        Y_hat = torch.square(Y_hat * (torch.sqrt(X_noisy) + 1e-10))
        return Y_hat
    
    def forward(self, x: Tensor, y: Tensor, inference=False):
        # STFT for noisy and clean waveform + normalization
        X, X_norm = self.input_stft(x)
        Y = self.target_stft(y, X_norm)
        # Target Mel-spectrogram
        Y = self.safe_log(Y)
        # Model Forward
        MRM_hat = self.arch(X, inference=inference)
        # Apply sigmoid for masking
        MRM_hat = torch.sigmoid(MRM_hat)
        # Obtain MRM prediction/target
        MRM_target = self.get_mrm_target(y, x, X_norm)
        Y_hat = self.get_mrm_pred(MRM_hat, x, X_norm)
        Y = self.get_mrm_pred(MRM_target, x, X_norm)    # ideal logMel target
        return MRM_hat, MRM_target, self.safe_log(Y_hat), self.safe_log(Y), X_norm

    def training_step(self, batch, batch_idx):
        """training step on self.device, called automaticly by PytorchLightning"""
        x, ys, paras = batch  # x: [B,T], ys: [B,T]
        # Model forward
        MRM_hat, MRM_target, Y_hat, Y, _ = self.forward(x, ys)
        mrm_loss = F.mse_loss(MRM_hat, MRM_target)
        logmel_mse = F.mse_loss(Y_hat, Y)
        logmel_l1 = F.l1_loss(Y_hat, Y)
        self.log('train/mrm_loss', mrm_loss, batch_size=ys[0].shape[0], sync_dist=True, prog_bar=False)
        self.log('train/logmel_mse', logmel_mse, batch_size=ys[0].shape[0], sync_dist=True, prog_bar=False)
        self.log('train/logmel_L1', logmel_l1, batch_size=ys[0].shape[0], sync_dist=True, prog_bar=False)
        return mrm_loss

    def validation_step(self, batch, batch_idx):
        """validation step on self.device, called automaticly by PytorchLightning"""
        x, ys, paras = batch
        # forward & loss
        MRM_hat, MRM_target, Y_hat, Y, _ = self.forward(x, ys)
        mrm_loss = F.mse_loss(MRM_hat, MRM_target)
        logmel_mse = F.mse_loss(Y_hat, Y)
        logmel_l1 = F.l1_loss(Y_hat, Y)
        self.log('val/mrm_loss', mrm_loss, batch_size=ys[0].shape[0], sync_dist=True, prog_bar=False)
        self.log('val/logmel_mse', logmel_mse, batch_size=ys[0].shape[0], prog_bar=False)
        self.log('val/logmel_L1', logmel_l1, batch_size=ys[0].shape[0], prog_bar=False)
        self.log('val/metric', -mrm_loss, batch_size=ys.shape[0])  # log val/metric for checkpoint picking 
            
    def on_validation_epoch_end(self) -> None:
        """calculate heavy metrics for every N epochs"""
        GS.on_validation_epoch_end(self=self, cpu_metric_input=self.val_cpu_metric_input, N=5)

    def on_test_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.results, self.cpu_metric_input = [], []

    def on_test_epoch_end(self):
        GS.on_test_epoch_end(self=self, results=self.results, cpu_metric_input=self.cpu_metric_input, exp_save_path=self.exp_save_path)

    def chunk_forward(self, x: Tensor, y: Tensor, chunk_len=20, overlap=5):
        chunk_len = chunk_len * self.sample_rate
        overlap = overlap * self.sample_rate
        chunk_Y, chunk_Y_hat = [], []
        start_pos = None
        for st in tqdm(range(0, x.shape[-1], chunk_len - overlap)):
            ed = min(st + chunk_len, x.shape[-1])
            x_chunk = x[..., st:ed]
            y_chunk = y[..., st:ed]
            _, _, Y_hat, Y_target, _ = self.forward(x_chunk, y_chunk)
            if start_pos is None:
                start_pos = int(overlap / self.stft.hop_length) + 1
                chunk_Y_hat.append(Y_hat)
                chunk_Y.append(Y_target)
            else:
                chunk_Y_hat.append(Y_hat[..., start_pos:])
                chunk_Y.append(Y_target[..., start_pos:])
        chunk_Y_hat = torch.cat(chunk_Y_hat, dim=-1)
        chunk_Y = torch.cat(chunk_Y, dim=-1)
        return chunk_Y_hat, chunk_Y

    def test_step(self, batch, batch_idx):
        x, ys, paras = batch
        min_len = min(x.shape[-1], ys.shape[-1])
        x = x[..., :min_len]
        ys = ys[..., :min_len]
        assert x.shape[-1] == ys.shape[-1], f"Input and target length mismatch: {x.shape[-1]} vs {ys.shape[-1]}"
        sample_rate = self.sample_rate if 'sample_rate' not in paras[0] else paras[0]['sample_rate']
        # forward & loss
        MRM_hat, MRM_target, Y_hat, Y, X_norm = self.forward(x, ys)
        mrm_loss = F.mse_loss(MRM_hat, MRM_target)
        logmel_mse = F.mse_loss(Y_hat, Y)
        logmel_l1 = F.l1_loss(Y_hat, Y)
        # logging images/audios
        Y_hat = Y_hat.clamp(min=math.log(self.log_eps))
        y_hat = self.vocos(Y_hat, X_norm).clamp(min=-1, max=1)
        clip_length = min(y_hat.shape[-1], ys.shape[-1])
        y_hat = y_hat[..., :clip_length]
        y_rconst = y_rconst[..., :clip_length]
        ys = ys[..., :clip_length]
        x = x[..., :clip_length]
        wavname = os.path.basename(paras[0]['saveto'])
        result_dict = {
            'id': batch_idx, 
            'wavname': wavname, 
            "LogMel_MSE": logmel_mse.item(), 
            "LogMel_L1": logmel_l1.item(), 
            "MRM_MSE": mrm_loss.item()
            }

        # calculate metrics, input_metrics, improve_metrics on GPU
        metrics, input_metrics, imp_metrics = cal_metrics_functional(
            self.metrics, y_hat[0], ys[0], x[0], sample_rate, device_only='gpu')
        result_dict.update(input_metrics)
        result_dict.update(imp_metrics)
        result_dict.update(metrics)
        self.cpu_metric_input.append((
            self.metrics, y_hat[0].detach().cpu(), ys[0].detach().cpu(), x[0].detach().cpu(), sample_rate, 'cpu'))
        # write examples
        if self.write_examples < 0 or paras[0]['index'] < self.write_examples:
            GS.test_setp_write_example(
                self=self,
                xr=x/x.abs().max(),
                yr=ys.unsqueeze(1)/ys.abs().max(),
                yr_hat=y_hat.unsqueeze(1)/y_hat.abs().max(),
                sample_rate=sample_rate,
                paras=paras,
                result_dict=result_dict,
                wavname=wavname.replace(".wav", ".flac"),
                exp_save_path=self.exp_save_path,
            )
            # save predictions
            Y_hat_numpy = Y_hat.cpu().numpy()
            numpy_save_path = os.path.join(self.exp_save_path, 'examples', paras[0]["saveto"])
            np.save(numpy_save_path + "/pred.npy", Y_hat_numpy)
        if 'metrics' in paras[0]:
            del paras[0]['metrics']
        result_dict['paras'] = paras[0]
        self.results.append(result_dict)
        return result_dict
    
    def configure_optimizers(self):
        """configure optimizer and lr_scheduler"""
        return GS.configure_optimizers(
            self=self,
            optimizer=self.optimizer[0],
            optimizer_kwargs=self.optimizer[1],
            monitor='val/loss',
            lr_scheduler=self.lr_scheduler[0] if self.lr_scheduler is not None else None,
            lr_scheduler_kwargs=self.lr_scheduler[1] if self.lr_scheduler is not None else None,
        )

    def on_predict_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path + "/logmel/", exist_ok=True)
        os.makedirs(self.exp_save_path + "/wav/", exist_ok=True)

    def predict_step(self, batch, batch_idx):
        x, wavename = batch
        # Enhanced Mel-spectrogram + waveform
        if (x.shape[1] / self.sample_rate) > 20 and not self.online:
            Y_hat, Y, X_norm = self.chunk_forward(x, x)
        else:
            MRM_hat, MRM_target, Y_hat, Y, X_norm = self.forward(x, x)
        y_hat = self.vocos(Y_hat, X_norm).clamp(min=-1, max=1)
        # Save result
        for i in range(len(x)):
            out_name = os.path.basename(wavename[i])
            sf.write(
                f"{self.exp_save_path}/wav/{out_name}",
                y_hat[i].detach().cpu().squeeze().numpy(), 
                self.sample_rate)
            np.save(
                f"{self.exp_save_path}/logmel/{out_name.replace('.wav', '.npy')}",
                Y_hat[i].cpu().numpy())
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        GS.on_load_checkpoint(self=self, checkpoint=checkpoint, weightavg_opts=False, compile=self.compile_model)
                
                
class TrainCLI(BaseCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_metric{val/metric:.4f}",
            "model_checkpoint.monitor": "val/metric",
            "model_checkpoint.mode": "max",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": -1,  # save all checkpoints
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        self.add_model_invariant_arguments_to_parser(parser)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    cli = TrainCLI(
        TrainModule,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
    )
    
