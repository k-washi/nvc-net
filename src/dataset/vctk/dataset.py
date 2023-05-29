import numpy as np

from torch.utils.data import Dataset
from pathlib import Path

from src.util.ml import torch_random_int

class VCTKDataset(Dataset):
    def __init__(self, cfg, data_dir, dataset_metadata_file) -> None:
        super().__init__()
        self.cfg = cfg
        self.segment_length = self.cfg.dataset.segment_length

        self._path = Path(data_dir)
        self._wave_npz_list = []
        with open(dataset_metadata_file, "r") as f:
            for l in f:
                fname = l.strip()
                self._wave_npz_list.append(fname)
        sorted(self._wave_npz_list)
        
    def __len__(self):
        return len(self._wave_npz_list)
    
    def __getitem__(self, index):
        fname = self._wave_npz_list[index]
        fp = self._path / fname
        data = np.load(self._path / fname)
       
        w, speaker_id = data["wave"], data["speaker_id"]
        
        w *= 0.99 / (np.max(np.abs(w)) + 1e-7)
        if len(w) > self.segment_length:
            idx = torch_random_int(0, len(w) - self.segment_length)
            w = w[idx:idx+self.segment_length]
        else:
            w = np.pad(w, (0, self.segment_length - len(w)), mode='constant')
        w = w[np.newaxis, :]
        return w, speaker_id, str(fp)

if __name__ == "__main__":
    from src.util.conf import get_hydra_cnf
    from torch.utils.data import DataLoader
    cfg = get_hydra_cnf(config_dir="./src/conf", config_name="default")
    data_dir = "data/vctk/train/data"
    dataset_metadata_train_file = "data/vctk/train/metadata_train.csv"
    
    dataset = VCTKDataset(cfg, data_dir, dataset_metadata_train_file)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for w, speaker_id in dataloader:
        speaker_id = speaker_id.unsqueeze(1)
        print(w.shape, speaker_id, speaker_id.shape)
        break
    print(w.shape, speaker_id, speaker_id.shape)