import numpy as np

from torch.utils.data import Dataset
from pathlib import Path

from src.util.ml import torch_random_int

class VCTKDataset(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.segment_length = self.cfg.dataset.segment_length

        self._path = Path(cfg.dataset.data_dir)
        self._wave_npz_list = []
        with open(cfg.dataset.dataset_metadata_file, "r") as f:
            for l in f:
                fname = l.strip()
                self._wave_npz_list.append(fname)
        sorted(self._wave_npz_list)
        
    def __len__(self):
        return len(self._wave_npz_list)
    
    def __getitem__(self, index):
        fname = self._wave_npz_list[index]
        data = np.load(self._path / fname)
        w, speaker_id = data["wave"], data["speaker_id"]
        
        w *= 0.99 / (np.max(np.abs(w)) + 1e-7)
        if len(w) > self.segment_length:
            idx = torch_random_int(0, len(w) - self.segment_length)
            w = w[idx:idx+self.segment_length]
        else:
            w = np.pad(w, (0, self.segment_length - len(w)), mode='constant')
        return w, speaker_id

if __name__ == "__main__":
    from src.util.conf import get_hydra_cnf
    cfg = get_hydra_cnf(config_dir="./src/conf", config_name="default")
    cfg.dataset.data_dir = "data/vctk/train/data"
    cfg.dataset.dataset_metadata_file = "data/vctk/train/metadata_train.csv"
    
    dataset = VCTKDataset(cfg)
    w, speaker_id = next(iter(dataset))
    print(w.shape, speaker_id)