import torch
import torchaudio
import torchaudio.transforms as at
import torch.nn as nn
import torch.nn.functional as F
from torch_audiomentations import Compose, Shift
import numpy as np
import random
from pathlib import Path
from typing import Union
from librosa.filters import mel as librosa_mel_fn


MAX_WAV_VALUE = 32768.0


def load_wave(
    wave_file_path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = False,
) -> torch.Tensor:
    """load wave

    Args:
        wave_file_path (str): file path
        sample_rate (int, optional): if -1 return original sample rate. Defaults to -1.
        is_torch (bool, optional): return torch.Tensor or np.ndarray. Defaults to True.
        mono (bool, optional):
            True: return [wave]
            False: return [channel, wave].
            Defaults to False.

    Returns:
        wave torch.Tensor or np.ndarray return
        sample_rate (int)
    """

    wave, sr = torchaudio.load(wave_file_path)
    if mono:
        wave = wave[0]
    if sample_rate != sr:
        wave = torchaudio.transforms.Resample(sr, sample_rate)(wave)
    return wave


def save_wave(
    wave: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    sample_rate: int = 16000,
) -> None:
    """save wave"""
    if not isinstance(wave, torch.Tensor):
        wave = torch.from_numpy(wave)

    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    torchaudio.save(
        filepath=str(output_path), src=wave.to(torch.float32), sample_rate=sample_rate
    )

## Signal processing

def dynamic_range_compression_torch(
    x: torch.Tensor, C: int = 1, clip_val: float = 1e-5
) -> torch.Tensor:
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: torch.Tensor, C: int = 1) -> torch.Tensor:
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_decompression_torch(magnitudes)
    return output

class Spectrogram(nn.Module):
    def __init__(self, window_size:int, hoplength_div:int=4, power=2.0, channel_ignore=False) -> None:
        super().__init__()
        self.win_size = window_size
        self.n_fft = window_size
        self.hop_size = int(window_size // hoplength_div)
        self.channel_ignore = channel_ignore
        self.hann_window = {}
    
    def forward(self, y):
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))
        dtype_device = str(y.dtype) + "_" + str(y.device)
        wnsize_dtype_device = str(self.win_size) + "_" + dtype_device
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(self.win_size).to(
                dtype=y.dtype, device=y.device
            )
        
        if y.dim() == 2:
            y =y.unsqueeze(1)
        y = torch.nn.functional.pad(
            y,
            (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window[wnsize_dtype_device],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        if self.channel_ignore and spec.dim() == 4:
            spec = spec.squeeze(1)
        return spec

class MelSpectrogram(nn.Module):
    def __init__(self, sr:int, window_size:int, n_mels:int=80, hoplength_div:int=4, channel_ignore=False, fmin=0, fmax=11025) -> None:
        super().__init__()
        self.win_size = window_size
        self.n_fft = window_size
        self.n_mels = n_mels
        self.hop_size = int(window_size // hoplength_div)
        self.sr = sr
        self.channel_ignore = channel_ignore
        
        self.mel_basis, self.hann_window = {}, {}
        self.fmin = fmin
        self.fmax = fmax
    
    def forward(self, y):
        if y.dim() == 3:
            y.squeeze(1)
            
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))
        dtype_device = str(y.dtype) + "_" + str(y.device)
        fmax_dtype_device = str(self.fmax) + "_" + dtype_device
        wnsize_dtype_device = str(self.win_size) + "_" + dtype_device
        if fmax_dtype_device not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
            )
            self.mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
                dtype=y.dtype, device=y.device
            )
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(self.win_size).to(
                dtype=y.dtype, device=y.device
        )
            
        if y.dim() == 2:
            y =y.unsqueeze(1)
        y = torch.nn.functional.pad(
            y,
            (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window[wnsize_dtype_device],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

        spec = torch.matmul(self.mel_basis[fmax_dtype_device], spec)
        spec = spectral_normalize_torch(spec)
        if self.channel_ignore and spec.dim() == 4:
            spec = spec.squeeze(1)
        return spec
    


#############
# データ拡張 #
############

def random_jitter(wave, max_jitter_steps):
    r"""Temporal jitter."""
    shape = wave.shape
    wave = F.pad(wave, (0, 0, max_jitter_steps, max_jitter_steps))
    wave = F.random_crop(wave, shape=shape)
    return wave

def random_scaling(x:torch.Tensor, low:float=0.0, high:float=1.0):
    """Random scaling a Variable.


    Args:
        x (torch.Tensor): (batch, 1, time length)
        low (float, optional): _description_. Defaults to 0.0.
        high (float, optional): _description_. Defaults to 1.0.
    """
    scale = (torch.rand((x.size()[0], 1, 1)) * (high - low) + low).to(x.device)
    return x * scale

def random_flip(x:torch.Tensor):
    """Random flipping sign of a Variable. (信号の上下を入れ替え)

    Args:
        x (torch.Tensor): _description_
    """
    scale = (2 * torch.randint(0, 2, (x.size()[0], 1, 1)) - 1).to(x.device)
    return x * scale

def get_random_jitter_transform(max_jitter_rate=0.001):
    """Temporal jitter.
    https://github.com/asteroid-team/torch-audiomentations
    """
    
    apply_augmenation = Compose(
        transforms=[
            Shift(
                min_shift=-max_jitter_rate,
                max_shift=max_jitter_rate,
                #shift_unit="samples",
                rollover=False,
                #p_mode="per_example",
                p=1.0
            )
        ]
    )
    return apply_augmenation

def random_split(x, lo, hi):
    """メルスペクトログラムをn分割し、分割したものを並べ替える
    x: メルスペクトログラム(batch, freq, time)
    lo: 分割エリアの最小サンプル数
    hi: 分割エリアの最大サンプル数
    """
    #x = torch.transpose(logmel, dim0=1, dim1=2)
    b, f, n = x.shape
    idx_list = []
    for _ in range(b):
        idx = np.cumsum(
            [random.randint(lo, hi)
            for _ in range(n // lo)])
        idx = idx[idx < n]
        partition = np.split(np.arange(n), idx)
        random.shuffle(partition)
        #print(partition)
        idx = np.zeros((1, f, n))
        idx[0, :] = np.hstack(partition)
        idx_list.append(idx)
    idx_list = np.vstack(idx_list)
    x = torch.gather(x, dim=2, index=torch.from_numpy(idx_list).long().to(x.device))
    return x

def change_vol(waveform, gain:float=1):
    return at.Vol(gain)(waveform)

def pitch_shift(waveform:torch.Tensor, sample_rate:int, pitch_shift:int=0):
    # waveform: 2d
    
    effects = [
        ["pitch", str(pitch_shift)],
        ["rate", f"{sample_rate}"],
        
    ]
    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )
    return waveform

def time_stretch(waveform:torch.Tensor, sample_rate:int, speed_rate:float=1.0):
    # waveform: 2d
    effects = [
        ["speed", str(speed_rate)],
        ["rate", f"{sample_rate}"],
        
    ]
    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )
    return waveform

def spec_aug(spec, time_mask_param:int=80, freq_mask_param:int=80):
    spec = at.TimeMasking(time_mask_param=time_mask_param)(spec)
    spec = at.FrequencyMasking(freq_mask_param=freq_mask_param)(spec)
    return spec
    
if __name__ == "__main__":
    wave_path = "./src/__example/001.wav"
    sr = 16000
    window_size = 512
    waveform = load_wave(wave_path, sr).cuda() # torch.Size([1, 32825]) # c, wave
    
    bwaveform = waveform.unsqueeze(0) # torch.Size([1, 1, 32825]) # b, c, wave
    bwaveform = torch.cat((bwaveform, bwaveform), dim=0) # to batch
    
    import time
    s = time.time()
    bwaveform = random_flip(bwaveform)
    #print("#a1", bwaveform.shape,  bwaveform[:, 0, :10])
    bwaveform = random_scaling(bwaveform, 0.25, 1.0)
    #print("#a2", bwaveform.shape, bwaveform[:, 0, :10])
    
    random_jitter = get_random_jitter_transform()
    bwaveform = random_jitter(bwaveform, sr)
    #print(time.time() - s)
    #print(bwaveform[:,0, :40])
    #print(bwaveform[:,0, -40:])
    #print("#a3", bwaveform.shape) # torch.Size([2, 1, 32825])
    spectf = Spectrogram(window_size=window_size).to(bwaveform.device)
    
    spec = spectf(bwaveform)
    
    print("#1", spec.shape) # torch.Size([2, 1, 257, 257])
    #print("#2", waveform_grif.shape)

    logspec = spectf(bwaveform)
    #print(logspec.shape) # torch.Size([2, 1, 257, 257])
    
    
    melspectf = MelSpectrogram(sr, 512, n_mels=80, channel_ignore=True).to(bwaveform.device)
    logmelspec = melspectf(bwaveform)
    #print("lms", logmelspec.shape) # torch.Size([2, 80, 257])
    print(logmelspec.shape)
    random_split(logmelspec, 30, 45)
    waveform_gain = change_vol(waveform.cpu(), gain=0.5)
    #print(waveform.max(), waveform_gain.max())
    
    waveform_pitch = pitch_shift(waveform.cpu(), sample_rate=sr, pitch_shift=3)
    #print(waveform.max(), waveform_gain.max())
    
    waveform_speed = time_stretch(waveform.cpu(), sample_rate=sr, speed_rate=1.4)
    #print(waveform_speed.shape)
    
    #print("#2", waveform_grif.shape)