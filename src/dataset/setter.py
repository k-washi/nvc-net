from src.util.logger import get_logger
logger = get_logger(debug=True)

def dataset_setter(cfg, dataset_list=("vctk")):
    if cfg.dataset.target not in dataset_list:
        raise NotImplementedError(f"{cfg.dataset.target}に関する設定はありません。")

    with open(cfg.dataset.speaker_list_file) as f:
        cfg.dataset.n_speakers = len(f.read().split('\n'))
        logger.info(f'Training data with {cfg.dataset.n_speakers} speakers.')

    if cfg.dataset.target == "vctk":
        from src.dataset.vctk.datamodule import VCTKDataModule
        return VCTKDataModule(cfg)
    else:
        raise NotImplementedError(f"{cfg.dataset.target}に関する設定はありません。")