import os
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.util.logger import get_logger
logger = get_logger(debug=True)

from src.train.base import (
    pick_log_hyperparams
)

from src.dataset.setter import dataset_setter


SEED = 3407
seed_everything(SEED, workers=True)

DATASET_LIST = ("vctk")

src_path = Path(__file__, "..","..", "..", "..", "..").resolve()

@hydra.main(
    version_base=None, 
    config_path=str(src_path.joinpath("src/conf")), 
    config_name="default"
)
def main(cfg: DictConfig):
    # config update
    cfg.ml.art_output_dir = str(Path(cfg.ml.art_output_dir, f"{cfg.ml.exp_id:05d}"))
    #cfg.ml.log_output_dir = str(Path(cfg.ml.log_output_dir, f"{cfg.ml.exp_id:05d}"))
    
    print(cfg)
    
    # create output path
    Path(cfg.ml.art_output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.ml.log_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Logger設定
    # {save_dir}/{name}/{version} (ex. log/outputs/vctk/00001) にlgoを格納
    tflogger = TensorBoardLogger(
        save_dir=cfg.ml.log_output_dir, 
        name=f"{cfg.dataset.target}", 
        version=f"{cfg.ml.exp_id:05d}"
    )
    tflogger.log_hyperparams(
        params=pick_log_hyperparams(cfg)
    )
    
    # モデル保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.ml.art_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 # all model save
    )
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    
    # 環境の確認
    try:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        device = "cpu"
    logger.info(f"accelerator: {device} / devices: {cfg.ml.gpu_devices}")
    
    data_module = dataset_setter(cfg, DATASET_LIST)
    
    if cfg.model.target == "nvcnet":
        from src.model.nvcnetmodule import NVCNetModelModule
        model = NVCNetModelModule(cfg)
    else:
        raise NotImplementedError(f"{cfg.model.target}のモデル設定はありません。")

    trainer = Trainer(
        precision=cfg.ml.mix_precision,
        accelerator=device,
        devices=cfg.ml.gpu_devices,
        max_epochs=cfg.ml.epochs,
        accumulate_grad_batches=cfg.ml.accumulate_grad_batches,
        #gradient_clip_val=cfg.ml.gradient_clip_val,
        profiler=cfg.ml.profiler,
        fast_dev_run=cfg.ml.fast_dev_run,
        logger=tflogger,
        callbacks=callback_list,
        num_sanity_val_steps=2
    )
    trainer.fit(model, data_module)
if __name__ == "__main__":
    main()