import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional
from torchmetrics import ScaleInvariantSignalNoiseRatio

from src.model.ops import (
    ResidualBlock,
    ResnetBlock,
    UpBlock,
    DownBlock
)

from src.util.audio import MelSpectrogram

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
        
        blocks = []
        in_channels = self._ngf
        for i, r in enumerate(reversed(self._ratios)):
            out_channels = 2**(i+1) * self._ngf
            blocks.append(
                DownBlock(
                    in_channels,
                    out_channels,
                    r,
                    self._n_residual_layers
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
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
        
        blocks = []
        in_channels = out_channels
        for i, r in enumerate(self._ratios):
            out_channels = (mult // (2**i)) * self._ngf // 2
            blocks.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    self._n_speaker_embedding,
                    r,
                    self._n_residual_layers
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        
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

class Speaker(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        self._n_speaker_embedding = cfg.model.n_speaker_embedding
        self._n_spk_layers = cfg.model.n_spk_layers
        kernel, pad = 3, 1
        
        in_channels = cfg.dataset.n_mels
        out_channles = 32
        self.init_conv = nn.Conv1d(
            in_channels, out_channles,
            kernel_size=kernel, padding=pad
        )
        
        res_blocks = []
        in_channels = out_channles
        for _ in range(self._n_spk_layers):
            out_channles = min(in_channels*2, 512)
            res_blocks.append(
                ResidualBlock(
                    in_channels,
                    out_channles,
                    kernel=kernel,
                    pad=pad
                )
            )
            in_channels = out_channles
        self.res_blocks = nn.ModuleList(res_blocks)
        self.last_layer = nn.Sequential(
            nn.LeakyReLU(0.2)
        )
        
        self.mu = nn.Conv1d(
            in_channels,
            self._n_speaker_embedding,
            kernel_size=1
        )
        
        self.logvar = nn.Conv1d(
            in_channels,
            self._n_speaker_embedding,
            kernel_size=1
        )
    
    def forward(self, x):
        # log mel spec
        x = self.init_conv(x)
        for b in self.res_blocks:
            x = b(x)
        kernel_size = (1, x.shape[-1])
        x = F.avg_pool2d(x, kernel_size, stride=kernel_size, padding=(0, 0))
        x = self.last_layer(x)
        
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    @staticmethod
    def kl_loss(mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        """Returns the Kullback-Leibler divergence loss with a standard Gaussian.

        Args:
            mu (torch.Tensor): Mean of the distribution of shape (B, D, 1).
            logvar (torch.Tensor): Log variance of the distribution of shape (B, D, 1).

        Returns:
            torch.Tensor: Kullback-Leibler divergence loss.
        """
        return 0.5 * torch.mean(torch.sum(logvar.exp() + mu.pow(2) - 1. - logvar, dim=1))

class NVCNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.speaker = Speaker(cfg)
        
    
    def forward(self, x:torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Convert an audio for a given reference.

        Args:
            x (torch.Tensor): Input audio of shape (B, 1, L).
            y_tar (torch.Tensor): Target class of shape (B, n_mels, times).

        Returns:
            torch.Tensor: Converted audio.
        """
        style, _, _ = self.embed(y)
        content = self.encode(x)
        out = self.decode(content, style)
        return out
    
    def encode(self, x:torch.Tensor) ->torch.Tensor:
        """Encode an input audio

        Args:
            x (torch.Tensor): Input audio of shape (B, 1, L)

        Returns:
            Content info (B, bottoleneck_dim, .)
        """
        return self.encoder(x)
    
    def decode(self, x:torch.Tensor, spk_emb:torch.Tensor) -> torch.Tensor:
        """Generate an audio from content and speaker info.

        Args:
            x (torch.Tensor): Content info (B, bottoleneck_dim, .)
            spk_emb (torch.Tensor): Traget class of shape (B or 1, speaker_emb, 1)
        Returns:
            torch.Tensor: Generated audio. (B, 1, L)
        """
        return self.decoder(x, spk_emb)
    
    def embed(self, x:torch.Tensor) -> torch.Tensor:
        """_Returns an embedding for a given audio reference.

        Args:
            x (torch.Tensor): Log mel spectrogram (B, n_mels, time)
        
        Returns:
            torch.Tensor: Embedding of the audio of shape (B, D, 1).
            torch.Tensor: Mean of the output distribution of shape (B, D, 1).
            torch.Tensore: Log variance of the output distribution of shape (B, D, 1).
        """
        
        mu, logvar = self.speaker(x)
        spk_emb = self.sample(mu, logvar)
        l2_norm_emb = torch.norm(spk_emb, p=2, dim=-1, keepdim=True)
        spk_emb = spk_emb / l2_norm_emb
        return spk_emb, mu, logvar
    
    def sample(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        """Samples from a Gaussian distribution.

        Args:
            mu (torch.Tensor): Mean of the distribution of shape (B, D, 1).
            logvar (torch.Tensor): Log variance of the distribution of shape (B, D, 1).

        Returns:
            torch.Tensor: A sample (B, D, 1).
        """
        if self.training:
            eps = torch.randn(size=mu.shape, device=mu.device)
            return mu + torch.exp(0.5 * logvar) * eps
        return mu
    
    def kl_loss(self, mu, logvar):
        """Returns the Kullback-Leibler divergence loss with a standard Gaussian.

        Args:
            mu (Tensor): Mean of the distribution of shape (B, D, 1).
            logvar (Tensor): Log variance of the distribution of
                shape (B, D, 1).

        Returns:
            Tensor: Kullback-Leibler divergence loss.
        """
        return - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

class NLayerDiscriminator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._ndf = cfg.model.ndf
        self._n_layers_d = cfg.model.n_layers_D
        self._num_d = cfg.model.num_D
        self._downsamp_factor = cfg.model.downsamp_factor
        self._n_d_updates = cfg.model.n_D_updates
        self._n_speakers = cfg.dataset.n_speakers
        
        
        discriminator_blocks = []
        in_channel = 1
        out_channel = self._ndf
        discriminator_blocks.append(nn.Sequential(
            nn.ReflectionPad1d((7,7)),
            nn.Conv1d(in_channel, out_channel, kernel_size=15),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        in_channel = out_channel
        stride = self._downsamp_factor
        
        
        for _ in range(1, self._n_layers_d + 1):
            out_channel = min(in_channel * stride, 1024)
            discriminator_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channel,
                        out_channel,
                        kernel_size=(stride * 10 + 1),
                        padding = stride * 5,
                        stride=stride,
                        groups=in_channel // 4
                    ),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channel = out_channel
        self.discriminator_blocks = nn.ModuleList(discriminator_blocks)
        out_channel = min(in_channel * 2, 1024)
        self.discriminator_blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channel,
                    out_channel,
                    kernel_size=5,
                    padding=2
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        in_channel = out_channel
        self.last_layer = nn.Sequential(
                nn.Conv1d(
                    in_channel,
                    self._n_speakers,
                    kernel_size=3,
                    padding=1
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )
    
    def forward(self, x, y:Optional[torch.Tensor]=None):
        results = []
        for b in self.discriminator_blocks:
            x = b(x)
            results.append(x)
            
        x = self.last_layer(x) # torch.Size([b, 103, 128]), b, speaker_num, time_num
        if y is not None:
            # 予測結果xのspeaker軸で正解speaker位置の予測結果を取得
            b, _, time_seq_num = x.size()
            # togazer ref
            # https://gundo0102.medium.com/comparison-of-torch-gather-and-tf-gather-nd-c306d05417b6
            idx = torch.zeros((b, time_seq_num))
            idx[:, :] = y
            idx = idx.unsqueeze(1) # add seqker axis
            x = torch.gather(x, dim=1, index=idx.long().to(x.device)).squeeze(1)
        results.append(x)
        return results

class Discriminator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # Discriminatorの設定
        self.num_dim = cfg.model.num_D
        nlayer_dis_list = []
        for _ in range(self.num_dim):
            nlayer_dis_list.append(
                NLayerDiscriminator(cfg)
            )
        self._nlayer_dis_list = nn.ModuleList(nlayer_dis_list)
        
        # LgoMelSpec変換
        self.window_sizes = cfg.model.window_sizes
        melspectf_list = [MelSpectrogram(
            cfg.dataset.sr,
            window_size = window_size,
            n_mels = cfg.dataset.n_mels,
            channel_ignore=True,
            fmin=cfg.dataset.fmin,
            fmax=cfg.dataset.fmax_loss
        ) for window_size in self.window_sizes]
        self.melspectf_list = nn.ModuleList(melspectf_list)
        
        #self._adversarial_bceloss = nn.BCEWithLogitsLoss(reduction="sum") # sigmoid and BCEloss (同時に計算すると計算的に安定)
        self._mean_squared_error = nn.MSELoss(reduction="mean")
        if cfg.model.sisnr_loss.is_use:
            self._si_snr_loss = ScaleInvariantSignalNoiseRatio()
    def forward(self, x, y):
        results = []
        for i in range(self.num_dim):
            # inputの時間方向を小さくする
            # torch.Size([2, 1, 32768])
            # torch.Size([2, 1, 16384])
            # torch.Size([2, 1, 8192])
            results.append(self._nlayer_dis_list[i](x, y))
            x = F.avg_pool2d(
                x, (1, 4), stride=(1, 2), padding=(0, 1),
                count_include_pad=True
            )
        return results
    
    def adversarial_loss(self, results, v, reduction="mean"):
        """Returns the adversarial loss.

        Args:
            results (list): Output from discriminator.
            v (int, optional): Target value. Real=1.0, fake=0.0.

        Returns:
            Tensor: Output variable.
        """
        loss = 0
        for out in results:
            # 時間方向に異なるデータを処理した、Discriminatorの結果ごとにlossを計算
            t = (torch.ones(out[-1].shape) * v).to(out[-1].device)
            
            if self.cfg.model.adv_loss_type == "bce":
                r = F.binary_cross_entropy_with_logits(out[-1], t.detach(), reduction=reduction)
            elif self.cfg.model.adv_loss_type == "mse":
                r = F.mse_loss(out[-1], t.detach(), reduction=reduction)
            else:
                raise NotImplementedError(self.cfg.model.adv_loss_type)
            loss += r
        return loss
    
    def preservation_loss(self, x, target):
        """Returns content preservation loss.
            Encoderの出力であるcontentに関する損失
            Args:
                x (Tensor): Input content variable
                target (Tensor): Target content variable.

            Returns:
                Tensor: Output loss.
        """
        return self._mean_squared_error(x, target)
    def spk_perservatoin_loss(self, x, target):
        """Speaker Embeddingの出力の一貫性を確認する損失

        Args:
                x (Tensor): Input content variable
                target (Tensor): Target content variable.

            Returns:
                Tensor: Output loss.
        """
        
        if self.cfg.model.spk_loss.type == "mse":
            return self._mean_squared_error(x, target)
        elif self.cfg.model.spk_loss.type == "cosine":
            return 1 - F.cosine_similarity(x, target, dim=-1).mean()
    
    def perceptual_loss(self, x, target):
        """Returns perceptual loss."""
        loss = 0.0
        out_x, out_t = self(x, None), self(target, None)
        
        for (a, t) in zip(out_x, out_t):
            # それぞれのNLayerDiscriminatorの結果
            for la, lt in zip(a[:-1], t[:-1]):
                lt = lt.detach() # avoid grads flowing though targets
                loss += torch.nn.functional.l1_loss(la, lt)
        return loss
        
    def spectral_loss(self, x, target):
        """Returns the multi-scale spectral loss.

        Args:
            x (Tensor): Input variable.
            target (Tensor): Target variable.

        Returns:
            Tensor: Multi-scale spectral loss.
        """
        loss = 0.0
        for melspectf in self.melspectf_list:
            #with torch.no_grad():
            st = melspectf(target)
            sx = melspectf(x)
            loss += self._mean_squared_error(sx, st.detach())
        return loss
    def si_snr_loss(self, x, target):
        """
        IMPROVING GAN-BASED VOCODER FOR FAST AND HIGH-QUALITY SPEECH SYNTHESIS
        https://www.isca-speech.org/archive/pdfs/interspeech_2022/mengnan22_interspeech.pdf
        snrは負のため、-1 * lossにしている
        """
        x, target = x.squeeze(1), target.squeeze(1)
        loss = self._si_snr_loss(x, target.detach())
        return -loss

if __name__ == "__main__":
    from src.util.conf import get_hydra_cnf
    cfg = get_hydra_cnf(config_dir="./src/conf", config_name="default")
    def test_enc_dec(cfg, nvc, dis):
        melspectf = MelSpectrogram(
            cfg.dataset.sr,
            window_size = cfg.dataset.window_size,
            n_mels = cfg.dataset.n_mels,
            channel_ignore=True,
            fmin=cfg.dataset.fmin,
            fmax=cfg.dataset.fmax
        ).cuda()
        
        batch = 2
        x, y = torch.ones((batch, 1, 32768)).cuda(), torch.ones((batch, 1, 32768)).cuda()
        label_x, label_y = torch.ones((batch, 1)).cuda(), torch.ones((batch, 1)).cuda() # spkear ids
        
        x_real_con = nvc.encode(x)
        logmelx = melspectf.log(x).cuda()
        s_r, s_mu, s_logvar = nvc.embed(logmelx)
        x_real = nvc.decode(x_real_con, s_r)
        
        # torch.Size([2, 4, 128]) torch.Size([2, 80, 129]) torch.Size([2, 128, 1]) torch.Size([2, 1, 32768])
        # [b, 4ch, 128times], [b, 80mel, 129times], [b, 128feat, 1], [b, 1, raw audio times]
        print(x_real_con.shape, logmelx.shape, s_r.shape, x_real.shape)
        
        logmely = melspectf.log(y).cuda()
        r_fake = nvc.embed(logmely)[0]
        x_fake = nvc.decode(x_real_con, r_fake) # fake 生成音
        x_fake_con = nvc.encode(x_fake) # fake content
        
        # torch.Size([2, 128, 1]) torch.Size([2, 1, 32768]) torch.Size([2, 4, 128])
        # [b, 128feat, 1], [b, 1, raw audio times], [b, 4ch, times]
        print(r_fake.shape, x_fake.shape, x_fake_con.shape)
        
        print(nvc.training)
        print(nvc.eval().training)
        
        # Discrimiator
        
        res = dis(x, label_x)
        adl = dis.adversarial_loss(res, 1)
        #print(adl)
        pres = dis.preservation_loss(x_fake_con, x_real_con)
        #print(pres)
        
        kld = nvc.kl_loss(s_mu, s_logvar)
        #print(kld)
        
        per = dis.perceptual_loss(x_real, x_real)
        
        specloss = dis.spectral_loss(x_real, x_real)
        print(specloss)
        loss = adl + pres + kld + per + specloss
        print(loss)
        loss.backward()
        #print(dis_real_x.shape)
        
        
    nvc = NVCNet(cfg).cuda()
    dis = Discriminator(cfg).cuda()

    for _ in range(5):
        test_enc_dec(cfg, nvc, dis)
        #print(torch.cuda.memory_summary())
    
        