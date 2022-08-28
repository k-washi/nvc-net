import random

import numpy as np
import torch


def set_seed(seed: int = 3407):
    """
    pytorch, numpy, randomのseedを固定する。
    Args:
        seed (int, optional): seedの値. Defaults to 3407.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True  # True:再現性は上がるが、処理パフォーマンスが低下

def tensor_to_np(waveform):
    if isinstance(waveform, np.ndarray):
        return waveform
    elif isinstance(waveform, torch.Tensor):
        return waveform.cpu().detach().numpy().copy()
    raise ValueError(f"Type error: {type(waveform)}")

def np_to_tensor(waveform):
    if isinstance(waveform, np.ndarray):
        return torch.from_numpy(waveform)
    elif isinstance(waveform, torch.Tensor):
        return waveform
    raise ValueError(f"Type error: {type(waveform)}")

def torch_random_int(low:int, high:int) -> int:
    return torch.randint(low=low, high=high, size=(1,)).item()