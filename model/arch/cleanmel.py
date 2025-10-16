from typing import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
import librosa
import soundfile as sf
import yaml

from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.common_types import _size_1_t

from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

from model.vocos.offline.pretrained import Vocos
from huggingface_hub import hf_hub_download

class LinearGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"

class LayerNorm(nn.LayerNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        """
        Arg s:
            seq_last (bool): whether the sequence dim is the last dim
        """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)  # [B, H, Seq] -> [B, Seq, H], or [B,H,w,h] -> [B,h,w,H]
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o

class CausalConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t | str = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        look_ahead: int = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.look_ahead = look_ahead
        assert look_ahead <= self.kernel_size[0] - 1, (look_ahead, self.kernel_size)

    def forward(self, x: Tensor, state: Dict[int, Any] = None) -> Tensor:
        # x [B,H,T]
        B, H, T = x.shape
        if state is None or id(self) not in state:
            x = F.pad(x, pad=(self.kernel_size[0] - 1 - self.look_ahead, self.look_ahead))
        else:
            x = torch.concat([state[id(self)], x], dim=-1)
        if state is not None:
            state[id(self)] = x[..., -self.kernel_size + 1:]
        x = super().forward(x)
        return x

class CleanMelLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_squeeze: int,
            n_freqs: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            f_kernel_size: int = 5,
            f_conv_groups: int = 8,
            padding: str = 'zeros',
            full: nn.Module = None,
            mamba_state: int = None,
            mamba_conv_kernel: int = None,
            online: bool = False,
    ) -> None:
        super().__init__()
        self.online = online
        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            LayerNorm(seq_last=True, normalized_shape=dim_hidden),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = LayerNorm(seq_last=False, normalized_shape=dim_hidden)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(n_freqs, n_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            LayerNorm(seq_last=True, normalized_shape=dim_hidden),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # narrow-band block
        self.norm_mamba = LayerNorm(seq_last=False, normalized_shape=dim_hidden)
        if online:
            self.mamba = Mamba(d_model=dim_hidden, d_state=mamba_state, d_conv=mamba_conv_kernel, layer_idx=0)
        else:
            self.mamba = nn.ModuleList([
                Mamba(d_model=dim_hidden, d_state=mamba_state, d_conv=mamba_conv_kernel, layer_idx=0),
                Mamba(d_model=dim_hidden, d_state=mamba_state, d_conv=mamba_conv_kernel, layer_idx=1),
            ])
        
        self.dropout_mamba = nn.Dropout(dropout[0])

    def forward(self, x: Tensor, inference: bool = False) -> Tensor:
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)        
        if self.online:
            x = x + self._mamba(x, self.mamba, self.norm_mamba, self.dropout_mamba, inference)
        else:
            x_fw = x + self._mamba(x, self.mamba[0], self.norm_mamba, self.dropout_mamba, inference)
            x_bw = x.flip(dims=[2]) + self._mamba(x.flip(dims=[2]), self.mamba[1], self.norm_mamba, self.dropout_mamba, inference)
            x = (x_fw + x_bw.flip(dims=[2])) / 2 
        return x

    def _mamba(self, x: Tensor, mamba: Mamba, norm: nn.Module, dropout: nn.Module, inference: bool = False):
        B, F, T, H = x.shape
        x = norm(x)
        x = x.reshape(B * F, T, H)
        if inference:
            inference_params = InferenceParams(T, B * F)
            xs = []
            for i in range(T):
                inference_params.seqlen_offset = i
                xi = mamba.forward(x[:, i, :], inference_params) #TODO: Brackets removed around i here, is it ok???
                xs.append(xi)
            x = torch.concat(xs, dim=1)
        else:
            inference_all = InferenceParams(T, B * F) #TODO: Are these inference params ok???
            x = mamba.forward(x, inference_params=inference_all)
        x = x.reshape(B, F, T, H)
        return dropout(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)
        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class CleanMel(nn.Module):

    def __init__(
        self,
        dim_input: int,  # the input dim for each time-frequency point
        dim_output: int,  # the output dim for each time-frequency point
        n_layers: int,
        n_freqs: int,
        n_mels: int = 80,
        layer_linear_freq: int = 1,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        f_kernel_size: int = 5,
        f_conv_groups: int = 8,
        padding: str = 'zeros',
        mamba_state: int = 16,
        mamba_conv_kernel: int = 4,
        online: bool = True,
        sr: int = 16000,
        n_fft: int = 512,
    ):
        super().__init__()
        self.layer_linear_freq = layer_linear_freq
        self.online = online
        # encoder
        self.encoder = CausalConv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, look_ahead=0)
        # cleanmel layers
        full = None
        layers = []
        for l in range(n_layers):
            layer = CleanMelLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=8 if l < layer_linear_freq else dim_hidden,
                n_freqs=n_freqs if l < layer_linear_freq else n_mels,
                dropout=dropout,
                f_kernel_size=f_kernel_size,
                f_conv_groups=f_conv_groups,
                padding=padding,
                full=full if l > layer_linear_freq else None,
                online=online,
                mamba_conv_kernel=mamba_conv_kernel,
                mamba_state=mamba_state,    
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        # Mel filterbank
        linear2mel = librosa.filters.mel(**{"sr": sr, "n_fft": n_fft, "n_mels": n_mels})
        self.register_buffer("linear2mel", torch.nn.Parameter(torch.tensor(linear2mel.T, dtype=torch.float32)))
        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, x: Tensor, inference: bool = False) -> Tensor:
        # x: [Batch, Freq, Time, Feature]
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        
        H = x.shape[2]
        x = x.reshape(B, F, T, H)
        # First Cross-Narrow band block in Linear Frequency
        for i in range(self.layer_linear_freq):
            m = self.layers[i]
            x = m(x, inference).contiguous()
        
        # Mel-filterbank
        x = torch.einsum("bfth,fm->bmth", x, self.linear2mel)

        for i in range(self.layer_linear_freq, len(self.layers)):
            m = self.layers[i]
            x = m(x, inference).contiguous()
        
        y = self.decoder(x).squeeze(-1)
        return y.contiguous()

def load_cleanmel(model_name):
    model_config = f"../../configs/model/cleanmel_offline.yaml"
    model_config = yaml.safe_load(open(model_config, "r"))["model"]["arch"]["init_args"]
    cleanmel = CleanMel(**model_config)
    REPO_ID = "WestlakeAudioLab/CleanMel"
    arch_ckpt = hf_hub_download(repo_id=REPO_ID, filename=f"ckpts/CleanMel/{model_name}")
    cleanmel.load_state_dict(torch.load(arch_ckpt))
    return cleanmel.eval()

def load_vocos():
    vocos = Vocos.from_hparams(config_path="../../configs/model/vocos_offline.yaml")
    REPO_ID = "WestlakeAudioLab/CleanMel"
    vocos_ckpt = hf_hub_download(repo_id=REPO_ID, filename="ckpts/Vocos/vocos_offline.pt")
    vocos = Vocos.from_pretrained(None, vocos_ckpt, model=vocos)
    return vocos.eval()

DEVICE = torch.device("cpu") #"cuda"

if __name__ == '__main__':
    # a quick demo here for the CleanMel model
    # input: wavs
    # output: enhanced log-mel spectrogram
    pytorch_lightning.seed_everything(1234)
    import soundfile as sf
    import matplotlib.pyplot as plt
    import numpy as np
    from model.io.stft import InputSTFT
    from model.io.stft import TargetMel
    from torch.utils.flop_counter import FlopCounterMode
    
    online=False
    # Define input STFT and target Mel
    stft = InputSTFT(
        n_fft=512, 
        n_win=512, 
        n_hop=128,
        center=True, 
        normalize=False, 
        onesided=True, 
        online=online).to("cpu") # "cuda"
    
    target_mel = TargetMel(
        sample_rate=16000,
        n_fft=512,
        n_win=512,
        n_hop=128,
        n_mels=80,
        f_min=0,
        f_max=8000,
        power=2,
        center=True,
        normalize=False,
        onesided=True,
        mel_norm="slaney",
        mel_scale="slaney",
        librosa_mel=True,
        online=online).to("cpu") # "cuda"

    def customize_soxnorm(wav, gain=-3, factor=None):
        wav = np.clip(wav, a_max=1, a_min=-1)
        if factor is None:
            linear_gain = 10 ** (gain / 20)
            factor = linear_gain / np.abs(wav).max()
            wav = wav * factor
            return wav, factor
        else:
            wav = wav * factor
            return wav, None

    # Noisy file path
    wav = "../../src/demos/noisy_CHIME-real_F05_442C020S_STR_REAL.wav"
    wavname = wav.split("/")[-1].split(".")[0]
    
    print(f"Processing {wav}")
    noisy, fs = sf.read(wav)
    dur = len(noisy) / fs
    noisy, factor = customize_soxnorm(noisy, gain=-3)
    noisy = torch.tensor(noisy).unsqueeze(0).float().to("cpu") # "cuda"
    # vocos norm
    x, x_norm = stft(noisy)
    # Load the model
    hidden=96
    depth=8
    model = CleanMel(
        dim_input=2,
        dim_output=1,
        n_layers=depth,
        dim_hidden=hidden,
        layer_linear_freq=1,
        f_kernel_size=5,
        f_conv_groups=8,
        n_freqs=257,
        mamba_state=16,
        mamba_conv_kernel=4,
        online=online,
        sr=16000,
        n_fft=512
    ).to("cpu") # "cuda"

    # Load the pretrained model
    state_dict = torch.load("../../pretrained/offline_CleanMel_S_mask.ckpt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval()
    with FlopCounterMode(model, display=False) as fcm:
        y_hat = model(x, inference=False)
        flops_forward_eval = fcm.get_total_flops()
    params_eval = sum(param.numel() for param in model.parameters())
    print(f"flops_forward={flops_forward_eval/1e9 / dur:.2f}G")
    print(f"params={params_eval/1e6:.2f} M")

    vocos = load_vocos().to(DEVICE)
    audio = vocos(y_hat, x_norm).clamp(min=-1, max=1)

    audio_np = audio.numpy()

    #TODO: Output data probably needs scaling to be correct here
    sf.write("output_audio.wav", np.ravel(audio_np), 16000, subtype='PCM_16')

    # TODO: Readd sanity check - Useful for ensuring optimizations do not cause incorrect output
    #if wavname == "noisy_CHIME-real_F05_442C020S_STR_REAL":
    #    assert np.allclose(y_hat, np.load("../../src/inference_example/check_CHIME-real_F05_442C020S_STR_REAL.npy"), atol=1e-5)

    # TODO: Optional: plot the enhanced mel spectrogram
    # y_hat is the enhanced log-mel spectrogram
    # y_hat = y_hat[0].cpu().detach().numpy()

    # noisy_mel = target_mel(noisy)
    # noisy_mel = torch.log(noisy_mel.clamp(min=1e-5))[0].cpu().detach().numpy()
    # vmax = math.log(1e2)
    # vmin = math.log(1e-5)
    # plt.figure(figsize=(8, 4))
    # plt.subplot(2, 1, 1)
    # plt.imshow(noisy_mel, aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
    # plt.colorbar()
    # plt.subplot(2, 1, 2)
    # plt.imshow(y_hat, aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(f"../../src/inference_example/{wavname}.png")
