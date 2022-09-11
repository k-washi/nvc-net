
def pick_log_hyperparams(cfg):
    return {
        "hp/batch_size": cfg.ml.batch_size,
        "epoch": cfg.ml.epochs,
        "g_lr": cfg.ml.g_lr,
        "d_lr": cfg.ml.d_lr,
        "lr_beta1": cfg.ml.beta1,
        "lr_beta2": cfg.ml.beta2,
        "weight_decay": cfg.ml.weight_decay,
        "n_D_updates": cfg.ml.n_D_updates,
        "dataset": cfg.dataset.target,
        "sr": cfg.dataset.sr,
        "segment_length": cfg.dataset.segment_length,
        "n_speakers": cfg.dataset.n_speakers,
        "window_size": cfg.dataset.window_size,
        "n_mels": cfg.dataset.n_mels,
        "arg_scale_low": cfg.dataset.scale_low,
        "arg_scale_high": cfg.dataset.scale_high,
        "arg_jitter_rate": cfg.dataset.max_jitter_rate
    }