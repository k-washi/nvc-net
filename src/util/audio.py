import torch
import torchaudio
import torchaudio.transforms as at
import torch.nn as nn

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

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

    def log(self, wave):
        return self(wave).log2()
    
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

    def log(self, wave):
        return self(wave).log2()
    


#############
# データ拡張 #
############
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
    print(bwaveform.shape) # torch.Size([2, 1, 32825])
    spectf = Spectrogram(window_size=window_size).to(bwaveform.device)
    
    spec = spectf(bwaveform)
    print("#1", spec.shape) # torch.Size([2, 1, 257, 257])
    waveform_grif = spectf.inverse_by_griffinlim(spec.abs())
    print("#2", waveform_grif.shape)

    logspec = spectf.log(bwaveform)
    print(logspec.shape) # torch.Size([2, 1, 257, 257])
    
    
    melspectf = MelSpectrogram(sr, 512, n_mels=80, channel_ignore=True).to(bwaveform.device)
    logmelspec = melspectf.log(bwaveform)
    print("lms", logmelspec.shape) # torch.Size([2, 1, 80, 257])
    
    waveform_gain = change_vol(waveform.cpu(), gain=0.5)
    print(waveform.max(), waveform_gain.max())
    
    waveform_pitch = pitch_shift(waveform.cpu(), sample_rate=sr, pitch_shift=3)
    print(waveform.max(), waveform_gain.max())
    
    waveform_speed = time_stretch(waveform.cpu(), sample_rate=sr, speed_rate=1.4)
    print(waveform_speed.shape)
    
    maskspec = spec_aug(spec)
    waveform_grif = spectf.inverse_by_griffinlim(maskspec)
    print("#2", waveform_grif.shape)