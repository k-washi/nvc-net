import torch
from pytorch_lightning import LightningModule
from pathlib import Path

from src.model.nvcnet import NVCNet, Discriminator
from src.util.audio import (
    random_flip,
    random_scaling,
    get_random_jitter_transform,
    MelSpectrogram,
    save_wave,
    random_split
)

from src.util.logger import get_logger
logger = get_logger(debug=True)

class NVCNetModelModule(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
        self.generator = NVCNet(cfg)
        self.discriminator = Discriminator(cfg)
        
        self.__random_jitter = get_random_jitter_transform(self.cfg.dataset.max_jitter_rate)
        self.__melspec_transformer = MelSpectrogram(
            cfg.dataset.sr,
            window_size = cfg.dataset.window_size,
            n_mels = cfg.dataset.n_mels,
            channel_ignore=True
        ) #.cuda()
        
        
    def __data_aug(self, v):
        v = random_flip(v)
        v = random_scaling(v, self.cfg.dataset.scale_low, self.cfg.dataset.scale_high)
        return v
        
    def forward(self, x, y):
        """Convert an audio for a given reference.

        Args:
            x (torch.Tensor): Input audio of shape (B, 1, L).
            y_tar (torch.Tensor): Target class of shape (B, n_mels, times).

        Returns:
            torch.Tensor: Converted audio.
        """
        return self.generator(x, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        gen_cond = optimizer_idx == 0
        dis_cond = optimizer_idx == 1 and batch_idx % self.cfg.ml.n_D_updates == 0
        
        if not (gen_cond or dis_cond):
            return None
        
        input_x, label_x = batch # (b, 1, wave length), (b, 1)
        label_x = label_x.unsqueeze(1)
        if not (input_x.dim() == 3 and input_x.size()[1] == 1):
            logger.error(f"音声データバッチのサイズが謝っている: dim {input_x.dim()}, size {input_x.size()}")
        if not (label_x.dim() == 2 and label_x.size()[1] == 1):
            logger.error(f"話者バッチのサイズが謝っている: dim {label_x.dim()}, size {label_x.size()}")
        
        idx = torch.randperm(self.cfg.ml.batch_size)
        input_y, label_y = input_x[idx], label_x[idx]
        
        x_aug = self.__data_aug(input_x)
        r_jitter_x = self.__random_jitter(x_aug, self.cfg.dataset.max_jitter_rate)
        
        self.__melspec_transformer = self.__melspec_transformer.to(input_x.device)
        
        # コンテンツ情報の取得
        x_real_con = self.generator.encode(x_aug)
        
        # 別話者音声生成 (A -> B)
        logmely =  self.__melspec_transformer(self.__data_aug(input_y))
        # 訓練時のみ、ランダムに分割入れ替え
        logmely = random_split(logmely, lo=self.cfg.dataset.split_low, hi=self.cfg.dataset.split_high)
        #print(logmely.shape)
        r_fake = self.generator.embed(logmely)[0] # spker_embed取得
        x_fake = self.generator.decode(x_real_con, r_fake) # 別のspeakerの音声生成
        
        if gen_cond:
            # train generater
            
            # 音声生成(A -> A)
            logmelx = self.__melspec_transformer(self.__data_aug(input_x))
            # 訓練時のみ、ランダムに分割入れ替え
            logmelx = random_split(logmelx, lo=self.cfg.dataset.split_low, hi=self.cfg.dataset.split_high)
            s_real, s_mu, s_logvar = self.generator.embed(logmelx)
            x_real = self.generator.decode(x_real_con, s_real)
            
            
            # 別のspeaker埋め込みから生成した音声のコンテンツ取得
            x_fake_con = self.generator.encode(random_flip(x_fake)) 
            
            # 損失の計算
            # 生成した音声を本物と間違うように学習する損失
            g_loss_adversarial = self.discriminator.adversarial_loss(
                self.discriminator(x_fake, label_y), 1.0
            )
            
            # コンテンツのロス
            g_loss_con = self.discriminator.preservation_loss(x_fake_con, x_real_con)
            
            # 話者埋め込みを正規分布に近づける
            g_loss_kld = self.generator.kl_loss(s_mu, s_logvar)
            
            # 再構成損失
            
            perceptual_loss = self.discriminator.perceptual_loss(x_real, r_jitter_x)
            spectral_loss = self.discriminator.spectral_loss(x_real, x_aug)
            g_loss_rec = perceptual_loss + spectral_loss
            self.log("train/g_loss_adv", g_loss_adversarial, on_step=True, prog_bar=True, logger=True)
            self.log("train/g_loss_con", g_loss_con, on_step=True, prog_bar=True, logger=True)
            self.log("train/g_loss_kld", g_loss_kld, on_step=True, prog_bar=True, logger=True)
            self.log("train/g_loss_rec", g_loss_rec, on_step=True, prog_bar=True, logger=True)
            
            return g_loss_adversarial + g_loss_con + g_loss_kld + g_loss_rec
        
        elif dis_cond:
            # train discriminator
            
            # Discriminatorを計算し、対象話者の時間方向のベクトルを取得
            dis_real_x = self.discriminator(self.__data_aug(input_x), label_x)
            dis_fake_y = self.discriminator(self.__data_aug(x_fake.detach()), label_y) # ここで生成器の学習は行わないのでx_fakeはdetach
            
            # Discriminatorの結果を見分けるように損失を計算
            d_loss = self.discriminator.adversarial_loss(dis_real_x, 1.0) \
                + self.discriminator.adversarial_loss(dis_fake_y, 0.0)
            self.log("train/d_loss", d_loss, on_step=True, prog_bar=True, logger=True)
            return d_loss
        
    def validation_step(self, batch, batch_idx):
        if batch_idx % 10 != 0:
            return None
        input_x, label_x = batch # (b, 1, wave length), (b, 1)
        label_x = label_x.unsqueeze(1)
        idx = torch.randperm(self.cfg.ml.batch_size)
        input_y, label_y = input_x[idx], label_x[idx]

        x_aug = self.__data_aug(input_x)
        r_jitter_x = self.__random_jitter(x_aug, self.cfg.dataset.max_jitter_rate)

        self.__melspec_transformer = self.__melspec_transformer.to(input_x.device)

        # コンテンツ情報の取得
        x_real_con = self.generator.encode(x_aug)
        
        # 音声生成(A -> A)
        logmelx = self.__melspec_transformer(self.__data_aug(input_x))
        s_real, s_mu, s_logvar = self.generator.embed(logmelx)
        x_real = self.generator.decode(x_real_con, s_real)
        
        # 別話者音声生成 (A -> B)
        logmely =  self.__melspec_transformer(self.__data_aug(input_y))
        r_fake = self.generator.embed(logmely)[0] # spker_embed取得
        x_fake = self.generator.decode(x_real_con, r_fake) # 別のspeakerの音声生成
        
        # 別のspeaker埋め込みから生成した音声のコンテンツ取得
        x_fake_con = self.generator.encode(random_flip(x_fake)) 
        
        # 損失の計算
        # 生成した音声を本物と間違うように学習する損失
        g_loss_adversarial = self.discriminator.adversarial_loss(
            self.discriminator(x_fake, label_y), 1.0
        )
        
        # コンテンツのロス
        g_loss_con = self.discriminator.preservation_loss(x_fake_con, x_real_con)
        
        # 話者埋め込みを正規分布に近づける
        g_loss_kld = self.generator.kl_loss(s_mu, s_logvar)
        
        # 再構成損失
        g_loss_rec = \
            self.discriminator.perceptual_loss(x_real, r_jitter_x) \
            + self.discriminator.spectral_loss(x_real, r_jitter_x)
        
        self.log("val/g_loss_adv", g_loss_adversarial, on_step=True, prog_bar=True, logger=True)
        self.log("val/g_loss_con", g_loss_con, on_step=True, prog_bar=True, logger=True)
        self.log("val/g_loss_kld", g_loss_kld, on_step=True, prog_bar=True, logger=True)
        self.log("val/g_loss_rec", g_loss_rec, on_step=True, prog_bar=True, logger=True)
        
        # Discriminatorを計算し、対象話者の時間方向のベクトルを取得
        dis_real_x = self.discriminator(self.__data_aug(input_x), label_x)
        dis_fake_y = self.discriminator(self.__data_aug(x_fake.detach()), label_y) # ここで生成器の学習は行わないのでx_fakeはdetach
        
        # Discriminatorの結果を見分けるように損失を計算
        d_loss = self.discriminator.adversarial_loss(dis_real_x, 1.0) \
            + self.discriminator.adversarial_loss(dis_fake_y, 0.0)
        self.log("val/d_loss", d_loss, on_step=True, prog_bar=True, logger=True)

        # 出力
        output_dir = Path(self.cfg.ml.art_output_dir)
        output_dir.mkdir(exist_ok=True)
        output_dir = output_dir / f"{self.cfg.ml.exp_id:05d}" / f"{self.current_epoch:05d}"
        output_dir.mkdir(exist_ok=True, parents=True)
        for i, (x, z) in enumerate(zip(input_x.detach().cpu(), x_fake.detach().cpu())):
            x_output_path = output_dir / f"{batch_idx:08d}_{i:05d}_in.wav"
            z_output_path = output_dir / f"{batch_idx:08d}_{i:05d}_fake.wav"
            
            save_wave(x, x_output_path, self.cfg.dataset.sr)
            save_wave(z, z_output_path, self.cfg.dataset.sr)
            
        return None

    def test_step(self):
        pass
        
        
    def configure_optimizers(self):
        weight_decay = self.cfg.ml.weight_decay
        g_lr = self.cfg.ml.g_lr
        d_lr = self.cfg.ml.d_lr
        beta1 = self.cfg.ml.beta1
        beta2 = self.cfg.ml.beta2
        
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=d_lr, betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        return [opt_g, opt_d], []
    
    
        
        
        
    