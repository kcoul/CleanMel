import torch
import librosa
import torch.nn as nn
import random
from torch import Tensor
from typing import Optional
from torchaudio.transforms import Spectrogram
from model.io.norm import recursive_normalization
from torchaudio.transforms import Spectrogram, MelScale

def soxnorm(wav: torch.Tensor, gain, factor=None):
    """sox norm, used in Vocos codes;
    """
    wav = torch.clip(wav, max=1, min=-1).float()
    if factor is None:
        linear_gain = 10 ** (gain / 20)
        factor = linear_gain / torch.abs(wav).max().item()
        wav = wav * factor
    else:
        # for clean speech, normed by the noisy factor
        wav = wav * factor
    assert torch.all(wav.abs() <= 1), f"out wavform is not in [-1, 1], {wav.abs().max()}"
    return wav, factor


class InputSTFT(nn.Module):
    """
    The STFT of the input signal of CleanMel (STFT coefficients);
    In online mode, the recursive normalization is used.
    """
    def __init__(
        self, 
        n_fft: int,
        n_win: int, 
        n_hop: int, 
        center: bool,
        normalize: bool,
        onesided: bool,
        online: bool = False):
        super().__init__()
        
        self.online = online
        self.stft=Spectrogram(
            n_fft=n_fft,
            win_length=n_win,
            hop_length=n_hop,
            normalized=normalize,
            center=center,
            onesided=onesided,
            power=None
        )
    
    def forward(self, x):
        if self.online:
            # recursive normalization
            x = self.stft(x)
            x_mag = x.abs()
            x_norm = recursive_normalization(x_mag)
            x = x / x_norm.clamp(min=1e-8)
            x = torch.view_as_real(x)
        else:
            # vocos dBFS normalization
            x, x_norm = soxnorm(x, random.randint(-6, -1) if self.training else -3)
            x = self.stft(x)
            x = torch.view_as_real(x)
        return x , x_norm


class LibrosaMelScale(nn.Module):
    r"""Pytorch implementation of librosa mel scale to align with common ESPNet ASR models; 
    You might need to define .
    """
    def __init__(self, n_mels, sample_rate, f_min, f_max, n_stft, norm=None, mel_scale="slaney"):
        super(LibrosaMelScale, self).__init__()
        
        _mel_options = dict(
            sr=sample_rate,
            n_fft=(n_stft - 1) * 2,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max if f_max is not None else float(sample_rate // 2),
            htk=mel_scale=="htk",
            norm=norm
        )
        
        fb = torch.from_numpy(librosa.filters.mel(**_mel_options).T).float()
        self.register_buffer("fb", fb)
    
    def forward(self, specgram):
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return mel_specgram


class TargetMel(nn.Module):
    """
    This class generates the enhancement TARGET mel spectrogram;
    """
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_win: int,
        n_hop: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        power: int,
        center: bool,
        normalize: bool,
        onesided: bool,
        mel_norm: str | None,
        mel_scale: str,
        librosa_mel: bool = True,
        online: bool = False,
        ):
        super().__init__()
        # This implementation vs torchaudio.transforms.MelSpectrogram: Add librosa melscale
        # librosa melscale is numerically different from the torchaudio melscale (x_diff > 1e-5)
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.online = online
        self.stft = Spectrogram(
            n_fft=n_fft,
            win_length=n_win,
            hop_length=n_hop,
            power=None if online else power,
            normalized=normalize,
            center=center,
            onesided=onesided,
        )
        mel_method = LibrosaMelScale if librosa_mel else MelScale
        self.mel_scale = mel_method(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            norm=mel_norm,
            mel_scale=mel_scale,
        )
        
    def forward(self, x: Tensor, x_norm=None):      
        if self.online:
            # apply recursive normalization to target waveform
            spectrogram = self.stft(x)
            spectrogram = spectrogram / (x_norm + 1e-8)
            spectrogram = spectrogram.abs().pow(2)  # to power spectrogram
        else:
            # apply vocos dBFS normalization to target waveform
            x, _ = soxnorm(x, None, x_norm)
            spectrogram = self.stft(x)
        # mel spectrogram
        mel_specgram = self.mel_scale(spectrogram)
        return mel_specgram
