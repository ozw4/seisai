import os
import random

import numpy as np
import torch

__all__ = ['worker_init_fn']


def worker_init_fn(worker_id: int) -> None:
	"""Seed numpy and random for each worker."""
	seed = torch.initial_seed() % 2**32
	np.random.seed(seed + worker_id)
	random.seed(seed + worker_id)


def set_seed(seed: int = 42):
	"""全ライブラリの乱数シードを固定して再現性を担保する。

	Args:
	    seed (int): 任意のシード値（デフォルト: 42）

	"""
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)
