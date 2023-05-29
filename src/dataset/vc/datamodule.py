from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.dataset.vc.dataset import VCDataset

class VCDataModule(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.cfg = cfg
    
    def train_dataloader(self) -> DataLoader:
        print("Loading train dataset ...")
        dataset = VCDataset(
            self.cfg,
            self.cfg.dataset.data_dir,
            self.cfg.dataset.dataset_metadata_train_file
        )
        print(f"train dataset num: {len(dataset)}")
        return DataLoader(
            dataset,
            batch_size=self.cfg.ml.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.cfg.ml.num_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        print("Loading val dataset ...")
        dataset = VCDataset(
            self.cfg,
            self.cfg.dataset.data_dir,
            self.cfg.dataset.dataset_metadata_val_file
        )
        print(f"val dataset num: {len(dataset)}")
        return DataLoader(
            dataset,
            batch_size=self.cfg.ml.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.cfg.ml.num_workers
        )
    
    def test_dataloader(self):
        dataset = VCDataset(
            self.cfg,
            self.cfg.dataset.data_dir,
            self.cfg.dataset.dataset_metadata_test_file
        )
        print(f"test dataset num: {len(dataset)}")
        return DataLoader(
            dataset,
            batch_size=self.cfg.ml.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.cfg.ml.num_workers
        )