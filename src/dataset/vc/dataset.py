import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
from librosa.util import normalize
from src.util.ml import torch_random_int
from src.util.audio import load_wave, MAX_WAV_VALUE

class VCDataset(Dataset):
    def __init__(self, cfg, data_dir, dataset_metadata_file, is_aug=True) -> None:
        super().__init__()
        self.cfg = cfg
        self.segment_length = self.cfg.dataset.segment_length
        
        # spk idを取得
        with open(cfg.dataset.speaker_list_file, "r") as f:
            spk_list =  f.read().splitlines()
        self.spk_dict = {spk: i for i, spk in enumerate(spk_list)}
        assert len(self.spk_dict) == cfg.dataset.n_speakers, f"{len(self.spk_dict)}!= {cfg.dataset.n_speakers}"
        
        
        self._path = Path(data_dir)
        with open(dataset_metadata_file, "r") as f:
            self._use_spk_list = f.read().splitlines()
        
        # datasetを作成
        self._dataset_list = []
        for spk in self._use_spk_list:
            spk_dir = self._path / spk
            if not spk_dir.exists():
                raise FileNotFoundError(f"{spk_dir} does not exist")
            spk_id = self.spk_dict[spk]
            fplist = sorted(list((spk_dir / "wav").glob("*")))
            assert len(fplist) > 0, f"{spk_dir} has no wav files"
            for fp in fplist:
                self._dataset_list.append((fp, spk_id))
        
    def __len__(self):
        return len(self._dataset_list)
    
    def __getitem__(self, index):
        fp, spk_id = self._dataset_list[index]
        w = load_wave(fp, sample_rate=self.cfg.dataset.sr, mono=True)
        w = w.numpy() / MAX_WAV_VALUE
        w = normalize(w) * 0.95
        
        if len(w) > self.segment_length:
            idx = torch_random_int(0, len(w) - self.segment_length)
            w = w[idx:idx+self.segment_length]
        else:
            w = np.pad(w, (0, self.segment_length - len(w)), mode='constant')
        w = w[np.newaxis, :]
        return w, spk_id, str(fp)

if __name__ == "__main__":
    from src.util.conf import get_hydra_cnf
    from torch.utils.data import DataLoader
    cfg = get_hydra_cnf(config_dir="./src/conf", config_name="default")
    data_dir = "/data/karanovc"
    dataset_metadata_train_file = "/nvc_net/results/train_karanovc_spk_list.txt"
    cfg.dataset.n_speakers = 604
    cfg.dataset.speaker_list_file = "/nvc_net/results/karanovc_spk_list.txt"
    
    dataset = VCDataset(cfg, data_dir, dataset_metadata_train_file)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for w, speaker_id, fp in dataloader:
        speaker_id = speaker_id.unsqueeze(1)
        print(w.shape, speaker_id, speaker_id.shape, fp)
        break
    print(w.shape, speaker_id, speaker_id.shape)