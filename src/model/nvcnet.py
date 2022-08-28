import numpy as np
from torch import nn
from torch.nn import functional as F
from src.model.ops import (
    ResidualBlock,
    ResnetBlock,
    UpBlock,
    DownBlock
)

class Encoder(nn.Module):
    def __init__(self, cfg, in_channels=1) -> None:
        super().__init__()
        self._ratios = cfg.model.ratios
        self._bottleneck_dim = cfg.model.bottleneck_dim
        self._ngf = cfg.model.ngf
        self._n_residual_layers = cfg.model.n_residual_layers
    
        self.first_layer = nn.Sequential(
            nn.ReflectionPad1d((3, 3)),
            nn.Conv1d(in_channels, self._ngf, kernel_size=7)
        )
        
        self.blocks = []
        in_channels = self._ngf
        for i, r in enumerate(reversed(self._ratios)):
            out_channels = 2**(i+1) * self._ngf
            self.blocks.append(
                DownBlock(
                    in_channels,
                    out_channels,
                    r,
                    self._n_residual_layers
                )
            )
            in_channels = out_channels
        
        self.last_layer = nn.Sequential(
            nn.GELU(),
            nn.ReflectionPad1d((3, 3)),
            nn.Conv1d(in_channels, in_channels, kernel_size=7)
        )
        
        self.content_layer = nn.Sequential(
            nn.GELU(),
            nn.ReflectionPad1d((3, 3)),
            nn.Conv1d(in_channels, self._bottleneck_dim, kernel_size=7, bias=False)
        )
    
    
    def forward(self, x):
        x = self.first_layer(x)
        for b in self.blocks:
            x = b(x)
        x = self.last_layer(x)
        x = self.content_layer(x)
        x = x / torch.sum(x**2 + 1e-12, dim=1, keepdim=True)**0.5
        return x

class Decoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self._ratios = cfg.model.ratios
        self._bottleneck_dim = cfg.model.bottleneck_dim
        self._ngf = cfg.model.ngf
        self._n_residual_layers = cfg.model.n_residual_layers
        self._n_speaker_embedding = cfg.model.n_speaker_embedding
        self.hop_length = np.prod(self._ratios)
        mult = int(2 **len(self._ratios))
        
        out_channels = mult * self._ngf
        self.upsample_layer = nn.Sequential(
            nn.ReflectionPad1d((3, 3)),
            nn.Conv1d(self._bottleneck_dim, out_channels, kernel_size=7)
        )
        
        self.first_layer = nn.Sequential(
            nn.GELU(),
            nn.ReflectionPad1d((3, 3)),
            nn.Conv1d(out_channels, out_channels, kernel_size=7)
        )
        
        self.blocks = []
        in_channels = out_channels
        for i, r in enumerate(self._ratios):
            out_channels = (mult // (2**i)) * self._ngf // 2
            self.blocks.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    self._n_speaker_embedding,
                    r,
                    self._n_residual_layers
                )
            )
            in_channels = out_channels
        self.waveform_layer = nn.Sequential(
            nn.GELU(),
            nn.ReflectionPad1d((3, 3)),
            nn.Conv1d(out_channels, 1, kernel_size=7),
            nn.Tanh()
        )
    
    def forward(self, x, spk_emb):
        x = self.upsample_layer(x)
        x = self.first_layer(x)
        for b in self.blocks:
            x = b(x, spk_emb)
        return self.waveform_layer(x)

if __name__ == "__main__":
    import torch
    from src.util.conf import get_hydra_cnf
    cfg = get_hydra_cnf(config_dir="./src/conf", config_name="default")
    def test_enc_dec(cfg):
        x = torch.ones((2, 1, 32768))
        spk_emb = torch.ones((1, 128, 1))
        print(x.shape)
        enc = Encoder(cfg)
        o = enc(x)
        print(o.shape)
        
        dec = Decoder(cfg)
        o = dec(o, spk_emb)
        print(o.shape)
    
    test_enc_dec(cfg)
        