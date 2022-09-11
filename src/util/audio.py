import torch
import torchaudio
import torchaudio.transforms as at
import torch.nn as nn
import torch.nn.functional as F
from torch_audiomentations import Compose, Shift
import numpy as np
import random
from nvcnet.utils.audio import random_jitter

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

def save_wave(wave, output_path, sample_rate:int=16000) -> torch.Tensor:
    if wave.dim() == 1: wave.unsqueeze(0)
    torchaudio.save(filepath=str(output_path), src=wave, sample_rate=sample_rate)
    

class Spectrogram(nn.Module):
    def __init__(self, window_size:int, hoplength_div:int=4, power=2.0, channel_ignore=False) -> None:
        super().__init__()
        self._tf = at.Spectrogram(
            n_fft=window_size,
            win_length=window_size,
            hop_length=int(window_size // hoplength_div),
            power=power
        )
        
        self._inv_tf =  at.GriffinLim(
            n_fft=window_size,
            win_length=window_size,
            hop_length=int(window_size // hoplength_div)
        )
        
        self.channel_ignore = channel_ignore
    
    def forward(self, wave):
        spec = self._tf(wave)
        if self.channel_ignore and spec.dim() == 4:
            spec = spec[:, 0, ...]
        return spec
    
    def inverse_by_griffinlim(self, spec):
        return self._inv_tf(spec)

class MelSpectrogram(nn.Module):
    def __init__(self, sr:int, window_size:int, n_mels:int=80, hoplength_div:int=4, channel_ignore=False) -> None:
        super().__init__()
        self._tf = at.MelSpectrogram(
            sr, 
            n_fft=window_size,
            win_length=window_size,
            hop_length=int(window_size // hoplength_div),
            n_mels=n_mels,
            f_min=80.0, f_max=7600.0,
        )
        self.channel_ignore = channel_ignore
    
    def forward(self, wave):
        melspec = self._tf(wave)
        if self.channel_ignore and melspec.dim() == 4:
            melspec = melspec[:, 0, ...]
        return melspec
    


#############
# データ拡張 #
############

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
    #print("#1", spec.shape) # torch.Size([2, 1, 257, 257])
    waveform_grif = spectf.inverse_by_griffinlim(spec.abs())
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
    
    maskspec = spec_aug(spec)
    waveform_grif = spectf.inverse_by_griffinlim(maskspec)
    #print("#2", waveform_grif.shape)