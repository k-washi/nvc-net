import os
from pathlib import Path
import hydra
from omegaconf import DictConfig

from src.util.logger import get_logger
logger = get_logger(debug=True)


@hydra.main(version_base=None, config_path="../conf", config_name="default")
def main(cfg: DictConfig):
    print(cfg)
    
    # create output path
    Path(cfg.ml.output_path).mkdir(parents=True, exist_ok=True)
    
    with open(cfg.dataset.speaker_list_file) as f:
        cfg.dataset.n_speakers = len(f.read().split('\n'))
        logger.info(f'Training data with {cfg.dataset.n_speakers} speakers.')
    pass

if __name__ == "__main__":
    main()