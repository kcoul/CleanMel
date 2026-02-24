if __name__ == '__main__':
    import os
    import sys
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.dirname(parent_dir))
    __package__ = os.path.basename(parent_dir)

import os
import random
import warnings
from sqlalchemy import collate
import torch
import numpy as np
import soundfile as sf
import pandas as pd

from pathlib import Path
from glob import glob
from typing import Callable, List, Optional, Tuple
from copy import deepcopy

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info # type: ignore
from torch.utils.data import DataLoader, Dataset
from data_loader.utils.collate_func import default_collate_func
from data_loader.utils.mix import *
from data_loader.utils.my_distributed_sampler import MyDistributedSampler

class CleanMelDataset(Dataset):
    def __init__(
        self,
        speech_dir: str,  # a dir contains [train, val, test]
        noise_dir: str,  # a dir contains [train, val, test]
        rir_dir: str,  # a dir contains [train.csv, val.csv, test.csv]
        dataset: str,
        snr=[20, 20],  # signal noise ratio
        audio_time_len: float = 4.0,
        sample_rate: int = 16000,
        no_reverb_prob: float = 0.2,
        dataset_len=None
    ) -> None:
        super().__init__()
        assert dataset in ['SimTrain', 'SimVal', 'SimTest'], dataset
        assert sample_rate == 16000, ('Not implemented for sample rate ', sample_rate)
        assert not (snr is None), "please provide snr"
        self.dataset = dataset
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate
        self.dataset_len = dataset_len
        self.no_reverb_prob = no_reverb_prob

        # scan uttrss
        self.speech_dir = speech_dir + {'SimTrain': '/train/', 'SimVal': '/val/', 'SimTest': '/test/'}[dataset]
        self.uttrs = glob(f"{self.speech_dir}/*.wav") + glob(f"{self.speech_dir}/*.flac")
        self.uttrs.sort()
        
        # scan rirs
        self.rir_csv = rir_dir + {'SimTrain': 'train.csv', 'SimVal': 'val.csv', 'SimTest': 'test.csv'}[dataset]
        rir_csv = pd.read_csv(self.rir_csv)
        self.rirs = rir_csv["filename"].tolist()
        self.rir_t60_dict = {x.split("/")[-1]: y for x, y in zip(rir_csv["filename"], rir_csv["t60"])}
        self.rirs.sort()
        self.rir_wav = True

        # scan noise
        self.noise_dir = noise_dir + {'SimTrain': '/train/', 'SimVal': '/val/', 'SimTest': '/test/'}[dataset]
        self.noises = glob(f"{self.noise_dir}/*.wav")
        self.noises.sort()
        self.snr = snr
        
        # check
        assert len(self.uttrs) > 0 and len(self.rirs) > 0 and len(self.noises) > 0, (
            'dir does not exist or is empty', self.speech_dir, len(self.uttrs), self.noise_dir, len(self.noises)
            )
        
        rank_zero_info(f"{dataset} speech duration: {sum([sf.info(x).duration for x in self.uttrs]) / 3600:.2f}")
        rank_zero_info(f"{dataset} noise duration: {sum([sf.info(x).duration for x in self.noises]) / 3600:.2f}")
        rank_zero_info(f"{dataset} num of rirs: {len(self.rirs)}")
        
    def __getitem__(self, index_seed: tuple[int, int]):
        index, seed = index_seed
        rng = np.random.default_rng(np.random.PCG64(seed))
        
        uttr_id = rng.integers(low=0, high=len(self.uttrs))
        
        # step 1: load clean speech
        org_src, sr_src = sf.read(self.uttrs[uttr_id], dtype='float32')
        if len(org_src) == 0 or np.abs(org_src).sum() == 0:
            # handle empty file
            del self.uttrs[uttr_id]
            return self.__getitem__(index_seed=(rng.integers(low=0, high=len(self)), rng.integers(low=0, high=9999999999)))
        assert sr_src == 16000, f"Wrong source sampling rate! {sr_src}"

        # step 2: load rirs
        rir_name = self.rirs[rng.integers(low=0, high=len(self.rirs))]
        t60 = self.rir_t60_dict[rir_name.split("/")[-1]]
        rir_real, rir_fs = sf.read(rir_name)
        assert rir_fs == 16000, f"Wrong rir sampling rate! {rir_fs}"
        if rir_real.ndim > 1:
            # handle multi-channel rir
            rir_idx = np.random.randint(0, rir_real.shape[1])
            rir_real = rir_real[:, rir_idx]
        peek_index = int(np.argmax(np.abs(rir_real)))
        dp_start = max(0, peek_index-32)
        rir = rir_real[np.newaxis, dp_start: ]
        rir_target = rir_real[np.newaxis, dp_start: peek_index+32]
        
        # step 3: adjust clean speech length
        target_len = int(self.sample_rate * self.audio_time_len) if self.audio_time_len is not None else len(org_src)
        clean = pad_or_cut_sample(wav=org_src, length=target_len, rng=rng)
        
        # step 4: convolve rir and clean speech
        if rng.random() < 0.2:
            # no rir case
            mix = deepcopy(clean[np.newaxis, :target_len])
            target = deepcopy(clean[np.newaxis, :target_len])
            t60 = 0
        else:
            mix, target = convolve(wav=clean, rir=rir, rir_target=rir_target, ref_channel=0, align=False)
            mix = mix[:, :target_len]
            target = target[:, :target_len]
        assert len(mix[0]) == len(target[0]), (len(mix[0]), len(target[0]), len(rir[0]), len(rir_target[0]))
        
        # step 5: add noise
        nidx = rng.integers(low=0, high=len(self.noises))
        noise_path = self.noises[nidx]
        noise, sr_noise = sf.read(noise_path, dtype='float32', always_2d=True)  # [T, num_mic]
        noise = noise[:, rng.choice(noise.shape[1])]    # random select one mic
        if np.abs(noise).sum() == 0:
            # handle empty file
            del self.noises[nidx]
            return self.__getitem__(index_seed=(rng.integers(low=0, high=len(self)), rng.integers(low=0, high=9999999999)))
        assert sr_noise == self.sample_rate, (sr_noise, self.sample_rate)
        
        # adjust noise length
        if noise.shape[0] < target_len:
            noise = pad_or_cut_sample(wav=noise, length=target_len, rng=rng)

        istart = rng.integers(low=0, high=max(noise.shape[0]-target_len, 1))
        noise = noise[istart:istart+target_len, np.newaxis].T
        
        # adjust snr
        snr_this = rng.uniform(low=self.snr[0], high=self.snr[1])
        coeff = cal_coeff_for_adjusting_relative_energy(wav1=mix, wav2=noise, target_dB=snr_this)
        if coeff is None:
            return self.__getitem__(index_seed=(rng.integers(low=0, high=len(self)), rng.integers(low=0, high=9999999999)))
        else:
            noise *= coeff
        # compute real snr (allow slightly different from snr_this)
        snr_real = 10 * np.log10(np.sum(mix**2) / np.sum(noise**2))
        if not np.isclose(snr_this, snr_real, atol=0.1):  # something wrong happen, skip this item
            warnings.warn(f'skip CleanMel/{self.dataset} item ({index},{seed})')
            return self.__getitem__(index_seed=(rng.integers(low=0, high=len(self)), rng.integers(low=0, high=9999999999)))
        assert np.isclose(snr_this, snr_real, atol=0.1), (snr_this, snr_real)
        
        # add noise
        mix = mix + noise

        # step 6: normalization (only for avoiding overflow/underflow)
        # The normalization of online/offline neural networks is in ./model/io/stft.forward()
        scale_value = 0.9 / np.max(np.abs(mix))
        mix *= scale_value
        target *= scale_value
        if np.max(np.abs(mix)) > 1 or np.max(np.abs(target)) > 1:
            return self.__getitem__(index_seed=(rng.integers(low=0, high=len(self)), rng.integers(low=0, high=9999999999)))
            
        paras = {
            'index': str(index),
            'seed': str(seed),
            'sample_rate': 16000,
            'dataset': f'CleanMel/{self.dataset}',
            'saveto': f"{index}.wav",
            'snr': float(snr_real) if snr_real is not None else None,
            't60': t60,
            'audio_time_len': self.audio_time_len
        }
        return torch.as_tensor(mix, dtype=torch.float32).squeeze(), torch.as_tensor(target, dtype=torch.float32).squeeze(), paras

    def __len__(self):
        if self.dataset_len is None:
            len_dict = {'SimTrain': 100000, 'SimVal': 3000, 'SimTest': 3000}
        else:
            len_dict = {'SimTrain': self.dataset_len[0], 'SimVal': self.dataset_len[1], 'SimTest': self.dataset_len[2]}
        return len_dict[self.dataset]


class CleanMelDataModule(LightningDataModule):
    def __init__(
        self,
        speech_dir: str = 'YOUR_SPEECH_DIR',  # a dir contains [train-clean-100, train-clean-360]
        noise_dir: str = 'YOUR_NOISE_DIR',  # a dir contains [reverb_tools_for_Generate_mcTrainData/NOISE, reverb_tools_for_Generate_SimData/NOISE]
        rir_dir: str = 'YOUR_RIR_DIR',  # a dir contains [train, validation, test]
        datasets: Tuple[str, str, str, List[str]] = ('SimTrain', 'SimVal'),  # datasets for train/val/test/predict
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float]] = [4.0, 4.0],  # audio_time_len (seconds) for training, val, test.
        snr: Optional[Tuple[float, float]] = [-5, 20],  # SNR dB
        batch_size: List[int] = [1, 1],
        num_workers: int = 10,
        collate_func: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int] = [None, 2],  # random seeds for train, val and test sets
        pin_memory: bool = True,
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
        dataset_len = None,
        no_reverb_prob: float = 0.2
    ):
        super().__init__()
        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        self.rir_dir = rir_dir
        self.datasets = datasets
        self.audio_time_len = audio_time_len
        self.snr = snr
        self.persistent_workers = persistent_workers
        self.dataset_len = dataset_len
        self.no_reverb_prob = no_reverb_prob

        self.batch_size = batch_size
        assert len(batch_size) == 2, batch_size
        if len(batch_size) <= 2:
            self.batch_size.append(1)

        rank_zero_info(f"dataset: CleanMel \ntrain/valid/: {self.datasets}")
        rank_zero_info(f'batch size: train={self.batch_size[0]}; val={self.batch_size[1]}')
        rank_zero_info(f'audio_time_length: train={self.audio_time_len[0]}; val={self.audio_time_len[1]}')

        self.num_workers = num_workers

        self.collate_func = collate_func
        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, audio_time_len, seed, shuffle, batch_size, collate_fn):
        ds = CleanMelDataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            rir_dir=self.rir_dir,
            dataset=dataset,  # 
            snr=self.snr,
            audio_time_len=audio_time_len,
            dataset_len=self.dataset_len,
            no_reverb_prob=self.no_reverb_prob
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),  #
            batch_size=batch_size,  #
            collate_fn=collate_fn,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[0],
            audio_time_len=self.audio_time_len[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func,
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            audio_time_len=self.audio_time_len[1],
            seed=self.seeds[1],
            shuffle=False,  # Please dont shuffle the data for validation
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func,
        )

if __name__ == '__main__':
    """To simulate the data:
        python -m data_loader.SPencn_NSdns_RIRreal"""
    from jsonargparse import ArgumentParser
    import pytorch_lightning as pl
    import json
    
    pl.seed_everything(0)
    parser = ArgumentParser("")
    parser.add_class_arguments(CleanMelDataModule, nested_key='data')
    parser.add_argument('--save_dir', type=str, default='dataset/MelSPNet')
    parser.add_argument('--dataset', type=str, default='val')
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if not args.gen_unprocessed and not args.gen_target:
        exit()

    args_dict = args.data
    args_dict['batch_size'] = [1, 1]
    args_dict['dataset_len'] = [100000, 3000, 3000]
    args_dict['num_workers'] = 0  # for debuging
    args_dict['prefetch_factor'] = None  # for debuging
    args_dict['snr'] = [-5, 20]
    datamodule = CleanMelDataModule(**args_dict)
    datamodule.setup()

    if args.dataset.startswith('train'):
        dataloader = datamodule.train_dataloader()
    elif args.dataset.startswith('val'):
        dataloader = datamodule.val_dataloader()
    elif args.dataset.startswith('test'):
        dataloader = datamodule.test_dataloader()
    else:
        assert args.dataset.startswith('predict'), args.dataset
        dataloader = datamodule.predict_dataloader()

    if type(dataloader) != dict:
        dataloaders = {args.dataset: dataloader}
    else:
        dataloaders = dataloader

    os.system(f"rm -r ./{args.save_dir}")
    for idx, packs in enumerate(dataloader):
        print(f'{idx}/{len(dataloader)}')
        
        # write target to dir
        noisy, tar, paras = packs
        tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/target").expanduser()
        noisy_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noisy").expanduser()
        param_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/param").expanduser()
        
        noisy_path.mkdir(parents=True, exist_ok=True)
        tar_path.mkdir(parents=True, exist_ok=True)
        param_path.mkdir(parents=True, exist_ok=True)
        
        sp = tar_path / (f"{idx}"  + f"_{paras[0]['t60']}" + ".flac")
        n_sp = noisy_path / (f"{idx}"  + f"_{paras[0]['t60']}" + ".flac")
        sf.write(sp, tar[0, :].numpy(), samplerate=paras[0]['sample_rate'])
        sf.write(n_sp, noisy[0,:].numpy(), samplerate=paras[0]['sample_rate'])
        para_path = param_path / (f"{idx}"  + f"_{paras[0]['t60']}" + ".json")
        json.dump(paras, open(para_path, "w"), indent=4)
                
