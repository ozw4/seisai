# packages/seisai-builders/src/seisai_builders/builder.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


# ---------- Wave producers（波形から作る派生物） ----------
class IdentitySignal:
	def __init__(self, src: str = 'x_view', dst: str = 'x_id', copy: bool = False):
		self.src, self.dst, self.copy = src, dst, copy

	def __call__(self, sample: dict[str, Any], rng=None):
		x = sample[self.src]
		sample[self.dst] = x.copy() if (self.copy and isinstance(x, np.ndarray)) else x


class MaskedSignal:
	def __init__(self, masker, src: str = 'x_view', dst: str = 'x_masked'):
		self.masker, self.src, self.dst = masker, src, dst

	def __call__(self, sample: dict[str, Any], rng=None):
		x = sample[self.src]
		xm, idx = self.masker.apply(x, py_random=None)
		sample[self.dst] = xm
		sample['mask_indices'] = idx


class MakeTimeChannel:
	"""(H,W) -> (H,W) 時刻チャネル（秒）。Crop/Pad後に使うこと"""

	def __init__(self, dst: str = 'time_ch'):
		self.dst = dst

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		H, W = sample['x_view'].shape
		dt = float(sample['dt_sec'])
		t = np.arange(W, dtype=np.float32) * dt
		sample[self.dst] = np.repeat(t[None, :], H, axis=0)


class MakeOffsetChannel:
	"""(H,) オフセットを (H,W) に拡張。normalize=True で z-score"""

	def __init__(self, dst: str = 'offset_ch', normalize: bool = True):
		self.dst, self.normalize = dst, normalize

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		off = sample['offsets'].astype(np.float32)
		if self.normalize:
			s = off.std() + 1e-6
			off = (off - off.mean()) / s
		H, W = sample['x_view'].shape
		sample[self.dst] = np.repeat(off[:, None], W, axis=1)


# ---------- Label producers（ラベルから作る派生物） ----------
class FBGaussMap:
	"""fb_idx からガウスマップを作る（面積正規化）。View meta（hflip/factor/start）を反映。"""

	def __init__(self, dst: str = 'fb_map', sigma: float = 1.5):
		self.dst, self.sigma = dst, float(sigma)

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		meta = sample.get('meta', {})
		fb = np.asarray(sample['fb_idx'], dtype=np.int64).copy()
		if meta.get('hflip', False):
			fb = fb[::-1]

		factor = float(meta.get('factor', 1.0))
		start = int(meta.get('start', 0))
		H, W = sample['x_view'].shape
		assert fb.shape[0] == H, f'fb_idx length {fb.shape[0]} != H {H}'

		# ビュー変換後の fb をウィンドウ内インデックスへ
		fb = np.floor(fb * factor).astype(np.int64) - start
		fb[(fb <= 0) | (fb >= W)] = -1

		# 面積正規化（各行の合計=1）
		y = np.zeros((H, W), dtype=np.float32)
		if np.any(fb >= 0):
			xs = np.arange(W, dtype=np.float32)
			s2 = self.sigma**2
			for h in range(H):
				idx = int(fb[h])
				if idx >= 0:
					g = np.exp(-0.5 * ((xs - idx) ** 2) / s2)
					s = float(g.sum())
					if s > 0.0:
						g = g / s
					y[h] = g

		sample[self.dst] = y  # numpy のまま（SelectStackでTensor化）


# ---------- 共通セレクタ（入力にもターゲットにも使う） ----------
class SelectStack:
	"""keys の2D/3Dを (C,H,W) に連結し、dst に格納。
	- 2D(H,W) は自動で [None, ...] して C 次元を付与
	- 3D(C,H,W) はそのまま
	- 形状が合わなければ ValueError
	"""

	def __init__(self, keys, dst: str, dtype=np.float32, to_torch: bool = True):
		self.keys = list(keys) if isinstance(keys, (list, tuple)) else [keys]
		self.dst = dst
		self.dtype = dtype
		self.to_torch = to_torch

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		mats = []
		H = W = None
		for k in self.keys:
			v = sample[k]
			if isinstance(v, torch.Tensor):
				v = v.detach().cpu().numpy()
			a = np.asarray(v, dtype=self.dtype)
			if a.ndim == 2:
				a = a[None, ...]  # (1,H,W)
			elif a.ndim != 3:
				raise ValueError(f'{k}: expected 2D/3D, got shape {a.shape}')
			if H is None:
				_, H, W = a.shape
			if a.shape[1] != H or a.shape[2] != W:
				raise ValueError(f'{k}: shape mismatch {a.shape} vs (*,{H},{W})')
			mats.append(a)
		out = np.concatenate(mats, axis=0)  # (C,H,W)
		sample[self.dst] = torch.from_numpy(out) if self.to_torch else out


# ---------- パイプライン実行器 ----------
class BuildPlan:
	def __init__(
		self,
		wave_ops: Iterable,
		label_ops: Iterable,
		input_stack: SelectStack,
		target_stack: SelectStack,
	):
		self.wave_ops = list(wave_ops)
		self.label_ops = list(label_ops)
		self.input_stack = input_stack
		self.target_stack = target_stack

	def run(self, sample: dict[str, Any], rng=None) -> None:
		for op in self.wave_ops:
			op(sample, rng)
		for op in self.label_ops:
			op(sample, rng)
		self.input_stack(sample, rng)
		self.target_stack(sample, rng)
