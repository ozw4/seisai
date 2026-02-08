import os
import random

import numpy as np
import torch

__all__ = ['get_np_rng', 'set_seed', 'worker_init_fn']


class _RngBox:
    __slots__ = ('rng',)

    def __init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng()


_NP = _RngBox()


def get_np_rng() -> np.random.Generator:
    return _NP.rng


def set_seed(seed: int = 42) -> None:
    """全ライブラリの乱数シードを固定して再現性を担保する。."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    _NP.rng = np.random.default_rng(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoaderの各workerで乱数を初期化する。."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)

    _NP.rng = np.random.default_rng(worker_seed)

    torch.manual_seed(worker_seed)
