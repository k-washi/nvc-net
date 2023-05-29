from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from pytorch_lightning import LightningModule
from pathlib import Path
import numpy as np
import itertools

from src.model.nvcnet import NVCNet, Discriminator
from src.util.audio import (
    random_flip,
    random_scaling,
    get_random_jitter_transform,
    MelSpectrogram,
    save_wave,
    random_split
)

from librosa.util import normalize
from src.util.ml import torch_random_int
from src.util.audio import load_wave, MAX_WAV_VALUE


from src.util.logger import get_logger
logger = get_logger(debug=True)

class NVCNetModelModule(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
        self.automatic_optimization = False
        self.generator = NVCNet(cfg)
        self.discriminator = Discriminator(cfg)
        
        self.__random_jitter = get_random_jitter_transform(self.cfg.dataset.max_jitter_rate)
        self.__melspec_transformer = MelSpectrogram(
            cfg.dataset.sr,
            window_size = cfg.dataset.window_size,
            n_mels = cfg.dataset.n_mels,
            channel_ignore=True,
            fmin=cfg.dataset.fmin,
            fmax=cfg.dataset.fmax
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
    
    def training_step(self, batch, batch_idx):
        dis_cond = batch_idx % self.cfg.ml.n_D_updates == 0
        optimizer_g, optimizer_d = self.optimizers()
        adv_real_label = 1
        if self.cfg.model.label_smooth:
            adv_real_label = 0.9
        
        input_x, label_x, _ = batch # (b, 1, wave length), (b, 1)
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
            self.discriminator(x_fake, label_y), adv_real_label
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
        si_snr_loss = 0
        if self.cfg.model.sisnr_loss.is_use:
            # SISNR loss
            si_snr_loss = self.discriminator.si_snr_loss(x_real, x_aug)
            self.log("train/g_loss_snr", si_snr_loss, on_step=True, prog_bar=True, logger=True)
            
        g_loss_spk_emb = 0
        if self.cfg.model.spk_loss.is_use:
            # Spk embedding preservation loss
            logmely = self.__melspec_transformer(x_fake)
            _r_fake = self.generator.embed(logmely)[0] # spker_embed取得
            g_loss_spk_emb = self.discriminator.spk_perservatoin_loss(_r_fake, r_fake)
            self.log("train/g_loss_spk", g_loss_spk_emb, on_step=True, prog_bar=True, logger=True)
            
        g_loss = g_loss_adversarial + \
                self.cfg.model.lambda_con * g_loss_con + \
                self.cfg.model.lambda_kld * g_loss_kld + \
                self.cfg.model.lambda_rec * g_loss_rec + \
                self.cfg.model.sisnr_loss.lambda_snr * si_snr_loss + \
                self.cfg.model.spk_loss.lambda_spk * g_loss_spk_emb
        self.log("train/g_loss", g_loss, on_step=True, prog_bar=True, logger=True)
        
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        
    
        if dis_cond:
            # train discriminator
            self.toggle_optimizer(optimizer_d)
            
            # Discriminatorを計算し、対象話者の時間方向のベクトルを取得
            dis_real_x = self.discriminator(self.__data_aug(input_x), label_x)
            dis_fake_y = self.discriminator(self.__data_aug(x_fake.detach()), label_y) # ここで生成器の学習は行わないのでx_fakeはdetach
            
            # Discriminatorの結果を見分けるように損失を計算
            d_loss = self.discriminator.adversarial_loss(dis_real_x, adv_real_label) \
                + self.discriminator.adversarial_loss(dis_fake_y, 0.0)
            self.log("train/d_loss", d_loss, on_step=True, prog_bar=True, logger=True)
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
    
    def on_validation_epoch_start(self) -> None:
        self._val_fp_dic = {}
        
    def validation_step(self, batch, batch_idx):
        _, labels, fp_list = batch
        for label, fp in zip(labels, fp_list):
            if label not in self._val_fp_dic:
                label = label.item()
                self._val_fp_dic[label] = []
            self._val_fp_dic[label].append(fp)
    
    def __preprocess(self, fp):
        w = load_wave(fp, sample_rate=self.cfg.dataset.sr, mono=True)
        w = w.numpy() / MAX_WAV_VALUE
        w = normalize(w) * 0.95
        
        segment_length = self.cfg.dataset.val_segment_length
        if len(w) > segment_length:
            idx = torch_random_int(0, len(w) - segment_length)
            w = w[idx:idx+segment_length]
        else:
            w = np.pad(w, (0, segment_length - len(w)), mode='constant')
        w = w[np.newaxis, :]
        w = torch.from_numpy(w).unsqueeze(0)
        return w
    
    def on_validation_epoch_end(self) -> None:
        adv_real_label = 1
        if self.cfg.model.label_smooth:
            adv_real_label = 0.9
            
        min_dataset_num = 10*10
        g_loss_list, g_loss_adv_list, g_loss_con_list, g_loss_kld_list, g_loss_rec_list, d_loss_list = [], [], [], [], [], []
        si_snr_loss_list,  g_loss_spk_emb_list = [], []
        for fp_list in self._val_fp_dic.values():
            min_dataset_num = min(min_dataset_num, len(fp_list))
        label_prod = itertools.permutations(self._val_fp_dic.keys(), 2)
        for i in range(min_dataset_num):
            for label1, label2 in label_prod:
                fp1 = self._val_fp_dic[label1][i]
                fp2 = self._val_fp_dic[label2][i]
                
                input_x = self.__preprocess(fp1)
                input_y = self.__preprocess(fp2)
                label_x = torch.tensor([label1], dtype=torch.long)
                label_y = torch.tensor([label2], dtype=torch.long)
                
                input_x = input_x.to(self.device)
                input_y = input_y.to(self.device)
                label_x = label_x.to(self.device)
                label_y = label_y.to(self.device)
                
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
                    self.discriminator(x_fake, label_y), adv_real_label
                )
                
                # コンテンツのロス
                g_loss_con = self.discriminator.preservation_loss(x_fake_con, x_real_con)
                
                # 話者埋め込みを正規分布に近づける
                g_loss_kld = self.generator.kl_loss(s_mu, s_logvar)
                
                # 再構成損失
                g_loss_rec = \
                    self.discriminator.perceptual_loss(x_real, r_jitter_x) \
                    + self.discriminator.spectral_loss(x_real, r_jitter_x)
                
                si_snr_loss = 0
                if self.cfg.model.sisnr_loss.is_use:
                    # SISNR loss
                    si_snr_loss = self.discriminator.si_snr_loss(x_real, x_aug)
                    si_snr_loss_list.append(si_snr_loss.detach().cpu().item())
                    
                g_loss_spk_emb = 0
                if self.cfg.model.spk_loss.is_use:
                    # Spk embedding preservation loss
                    logmely = self.__melspec_transformer(x_fake)
                    _r_fake = self.generator.embed(logmely)[0] # spker_embed取得
                    g_loss_spk_emb = self.discriminator.spk_perservatoin_loss(_r_fake, r_fake)
                    g_loss_spk_emb_list.append(g_loss_spk_emb.detach().cpu().item())
                    
                g_loss = g_loss_adversarial + \
                    self.cfg.model.lambda_con * g_loss_con + \
                    self.cfg.model.lambda_kld * g_loss_kld + \
                    self.cfg.model.lambda_rec * g_loss_rec + \
                    self.cfg.model.sisnr_loss.lambda_snr * si_snr_loss + \
                    self.cfg.model.spk_loss.lambda_spk * g_loss_spk_emb

                g_loss_adv_list.append(g_loss_adversarial.detach().cpu().item())
                g_loss_con_list.append(g_loss_con.detach().cpu().item())
                g_loss_kld_list.append(g_loss_kld.detach().cpu().item())
                g_loss_rec_list.append(g_loss_rec.detach().cpu().item())
                g_loss_list.append(g_loss.detach().cpu().item())
                
                # Discriminatorを計算し、対象話者の時間方向のベクトルを取得
                dis_real_x = self.discriminator(self.__data_aug(input_x), label_x)
                dis_fake_y = self.discriminator(self.__data_aug(x_fake.detach()), label_y) # ここで生成器の学習は行わないのでx_fakeはdetach
                
                # Discriminatorの結果を見分けるように損失を計算
                d_loss = self.discriminator.adversarial_loss(dis_real_x, adv_real_label) \
                    + self.discriminator.adversarial_loss(dis_fake_y, 0.0)
                d_loss_list.append(d_loss.detach().cpu().item())
                
                

                # 出力
                output_dir = Path(self.cfg.ml.art_output_dir)
                output_dir.mkdir(exist_ok=True)
                output_dir = output_dir / f"{self.cfg.ml.exp_id:05d}" / f"{self.current_epoch:05d}"
                output_dir.mkdir(exist_ok=True, parents=True)
                for i, (x, y, r, z) in enumerate(zip(input_x.detach().cpu(), input_y.detach().cpu(), x_real.detach().cpu(), x_fake.detach().cpu())):
                    x_output_path = output_dir / f"{i:05d}_{label1}_in.wav"
                    y_output_path = output_dir / f"{i:05d}_{label1}_ref.wav"
                    r_output_path = output_dir / f"{i:05d}_{label1}_real.wav"
                    z_output_path = output_dir / f"{i:05d}_{label1}_to_{label2}_fake.wav"
                    
                    save_wave(x, x_output_path, self.cfg.dataset.sr)
                    save_wave(y, y_output_path, self.cfg.dataset.sr)
                    save_wave(r, r_output_path, self.cfg.dataset.sr)
                    save_wave(z, z_output_path, self.cfg.dataset.sr)
        
        self.log("val/g_loss_adv", sum(g_loss_adv_list) / (len(g_loss_adv_list) + 1e-8), on_step=False, prog_bar=True, logger=True)
        self.log("val/g_loss_con", sum(g_loss_con_list) / (len(g_loss_con_list) + 1e-8), on_step=False, prog_bar=True, logger=True)
        self.log("val/g_loss_kld", sum(g_loss_kld_list) / (len(g_loss_kld_list) + 1e-8), on_step=False, prog_bar=True, logger=True)
        self.log("val/g_loss_rec", sum(g_loss_rec_list) / (len(g_loss_rec_list) + 1e-8), on_step=False, prog_bar=True, logger=True)
        self.log("val/g_loss", sum(g_loss_list) / (len(g_loss_list) + 1e-8), on_step=False, prog_bar=True, logger=True)
        self.log("val/d_loss", sum(d_loss_list) / (len(d_loss_list) + 1e-8), on_step=False, prog_bar=True, logger=True)
        if self.cfg.model.sisnr_loss.is_use:
            self.log("val/g_loss_snr", sum(si_snr_loss_list) / (len(si_snr_loss_list) + 1e-8), on_step=False, prog_bar=True, logger=True)
        if self.cfg.model.spk_loss.is_use:
            self.log("val/g_loss_spk", sum(g_loss_spk_emb_list) / (len(g_loss_spk_emb_list) + 1e-8), on_step=False, prog_bar=True, logger=True)
        self._val_fp_dic = {}
        return None

    def test_step(self):
        pass
        
        
    def configure_optimizers(self):
        weight_decay = self.cfg.ml.weight_decay
        g_lr = self.cfg.ml.g_lr
        d_lr = self.cfg.ml.d_lr
        beta1 = self.cfg.ml.beta1
        beta2 = self.cfg.ml.beta2
        
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=g_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=d_lr, betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        return [opt_g, opt_d], []
    
    
        
        
        
    