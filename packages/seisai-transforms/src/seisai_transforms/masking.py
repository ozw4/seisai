# transforms/masking.py
from dataclasses import dataclass
from typing import Literal

import numpy as np

MaskMode = Literal['replace', 'add']


@dataclass(frozen=True)
class TraceMaskerCfg:
	mask_ratio: float = 0.5
	mode: MaskMode = 'replace'
	noise_std: float = 1.0


def trace_mask_op(x: np.ndarray, *, cfg: TraceMaskerCfg, rng: np.random.Generator):
	"""純粋関数：x( H,T, float32 ) -> (x_masked, mask_indices, mask_bool)"""
	if x.ndim != 2:
		raise ValueError(f'x must be (H,T), got {x.shape}')
	H, T = x.shape
	ratio = float(cfg.mask_ratio)
	if not (0.0 <= ratio <= 1.0):
		raise ValueError('mask_ratio must be in [0,1]')
	if cfg.noise_std < 0:
		raise ValueError('noise_std must be >= 0')

	num = int(ratio * H)
	idx = [] if num == 0 else rng.choice(H, size=num, replace=False).tolist()

	x_m = x.copy()
	if num > 0 and cfg.noise_std > 0:
		noise = rng.normal(0.0, cfg.noise_std, size=(num, T)).astype(np.float32)
		if cfg.mode == 'replace':
			x_m[idx] = noise
		elif cfg.mode == 'add':
			x_m[idx] += noise
		else:
			raise ValueError(f'invalid mode: {cfg.mode}')
	mask_bool = np.zeros(H, dtype=bool)
	if num > 0:
		mask_bool[idx] = True
	return x_m, idx, mask_bool


class TraceMasker:
	def __init__(self, cfg: TraceMaskerCfg):
		self.cfg = cfg

	def __call__(self, x: np.ndarray, *, rng: np.random.Generator):
		return trace_mask_op(x, cfg=self.cfg, rng=rng)


# 使うとき（ピクル可）
# op = partial(trace_mask_op, cfg=TraceMaskerCfg(...))
# x_m, idx, m = op(x, rng=rng)
