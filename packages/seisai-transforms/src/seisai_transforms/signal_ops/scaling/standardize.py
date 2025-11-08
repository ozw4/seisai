import numpy as np
import torch
from seisai_utils.validator import validate_array
from torch import Tensor


def standardize_per_trace_np(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
	validate_array(x, allowed_ndims=(1, 2, 3, 4), name='x')
	xf = x.astype(np.float32, copy=False)
	m = xf.mean(axis=-1, keepdims=True)
	s = xf.std(axis=-1, keepdims=True) + float(eps)
	return (xf - m) / s


def standardize_per_trace_torch(x: Tensor, eps: float = 1e-10) -> Tensor:
	"""標準化（W軸で平均0・分散1）
	(W,) / (H,W) / (C,H,W) / (B,C,H,W) を受け付け、W は最後の軸と仮定。
	"""
	validate_array(x, allowed_ndims=(1, 2, 3, 4), name='x', backend='torch')
	xf: Tensor = x.to(dtype=torch.float32)
	m: Tensor = xf.mean(dim=-1, keepdim=True)
	s: Tensor = xf.std(dim=-1, keepdim=True, unbiased=False) + float(eps)
	return (xf - m) / s
