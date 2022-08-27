import torch
import torchaudio
import torchaudio.transforms as at

def load_wave(wave_path, sample_rate:int=16000):
    waveform, sr = torchaudio.load(wave_path)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


def spectrogram(wave:torch.Tensor, window_size:int) -> torch.Tensor:
    
    return at.Spectrogram(
        n_fft=window_size,
        win_length=window_size,
        hop_length=window_size // 4
    )(wave)

def log_spectrogram(wave:torch.Tensor, window_size:int) -> torch.Tensor:
    spec = spectrogram(wave, window_size)
    return torch.log(spec*1e4 + 1)

def mel_spectrogram(wave: torch.Tensor, sr:int, window_size:int, n_mels:int=80) -> torch.Tensor:
    return at.MelSpectrogram(
        sr, 
        n_fft=window_size,
        win_length=window_size,
        hop_length=window_size // 4,
        n_mels=n_mels,
        f_min=80.0, f_max=7600.0
    )(wave)

def log_mel_spectrogram(wave: torch.Tensor, sr:int, window_size:int, n_mels:int=80) -> torch.Tensor:
    melspec = mel_spectrogram(
        wave, sr, window_size, n_mels
    )
    return torch.log(melspec*1e4 + 1.0)



if __name__ == "__main__":
    wave_path = "./src/__example/001.wav"
    sr = 16000
    waveform = load_wave(wave_path, sr) # torch.Size([1, 32825])
    
    waveform = waveform.unsqueeze(0) # torch.Size([1, 1, 32825])
    waveform = torch.cat((waveform, waveform), dim=0) # to batch
    print(waveform.shape) # torch.Size([2, 1, 32825])
    
    spec = spectrogram(waveform, window_size=512)
    print(spec.shape) # torch.Size([2, 1, 257, 257])

    logspec = log_spectrogram(waveform, window_size=512)
    print(logspec.shape) # torch.Size([2, 1, 257, 257])
    
    logmelspec = log_mel_spectrogram(waveform, sr, 512, n_mels=80)
    print(logmelspec.shape) # torch.Size([2, 1, 80, 257])
    
    