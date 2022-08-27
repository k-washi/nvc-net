import torch
import torchaudio
import torchaudio.transforms as at

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

def spectrogram(wave:torch.Tensor, window_size:int, hoplength_div:int=4, power=2.0) -> torch.Tensor:
    
    return at.Spectrogram(
        n_fft=window_size,
        win_length=window_size,
        hop_length=int(window_size // hoplength_div),
        power=power
    )(wave)

def spec_to_inv_by_griffinlim(spec:torch.Tensor, window_size:int, hoplength_div:int=4) -> torch.Tensor:
    return at.GriffinLim(
        n_fft=window_size,
        win_length=window_size,
        hop_length=int(window_size / hoplength_div)
    )(spec)

def log_spectrogram(wave:torch.Tensor, window_size:int) -> torch.Tensor:
    spec = spectrogram(wave, window_size)
    return spec.log2()

def mel_spectrogram(wave: torch.Tensor, sr:int, window_size:int, n_mels:int=80, hoplength_div:int=4) -> torch.Tensor:
    return at.MelSpectrogram(
        sr, 
        n_fft=window_size,
        win_length=window_size,
        hop_length=int(window_size // hoplength_div),
        n_mels=n_mels,
        f_min=80.0, f_max=7600.0
    )(wave)

def log_mel_spectrogram(wave: torch.Tensor, sr:int, window_size:int, n_mels:int=80) -> torch.Tensor:
    melspec = mel_spectrogram(
        wave, sr, window_size, n_mels
    )
    return melspec.log2()

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
    waveform = load_wave(wave_path, sr) # torch.Size([1, 32825]) # c, wave
    
    bwaveform = waveform.unsqueeze(0) # torch.Size([1, 1, 32825]) # b, c, wave
    bwaveform = torch.cat((bwaveform, bwaveform), dim=0) # to batch
    print(bwaveform.shape) # torch.Size([2, 1, 32825])
    
    spec = spectrogram(bwaveform, window_size=window_size)
    print("#1", spec.shape) # torch.Size([2, 1, 257, 257])
    waveform_grif = spec_to_inv_by_griffinlim(spec.abs(), window_size)
    print("#2", waveform_grif.shape)

    logspec = log_spectrogram(waveform, window_size=512)
    print(logspec.shape) # torch.Size([2, 1, 257, 257])
    
    logmelspec = log_mel_spectrogram(waveform, sr, 512, n_mels=80)
    print(logmelspec.shape) # torch.Size([2, 1, 80, 257])
    
    waveform_gain = change_vol(waveform, gain=0.5)
    print(waveform.max(), waveform_gain.max())
    
    waveform_pitch = pitch_shift(waveform, sample_rate=sr, pitch_shift=3)
    print(waveform.max(), waveform_gain.max())
    
    waveform_speed = time_stretch(waveform, sample_rate=sr, speed_rate=1.4)
    print(waveform_speed.shape)
    
    maskspec = spec_aug(spec)
    waveform_grif = spec_to_inv_by_griffinlim(maskspec, window_size)
    print("#2", waveform_grif.shape)