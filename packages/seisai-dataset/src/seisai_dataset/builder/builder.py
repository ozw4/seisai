# packages/seisai-builders/src/seisai_builders/builder.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import torch
from seisai_pick.gaussian_prob import gaussian_probs1d_np


def _to_numpy(x, dtype=None) -> np.ndarray:
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()
	return np.asarray(x, dtype=dtype)


# ---------- Wave producers（波形から作る派生物） ----------
class IdentitySignal:
	def __init__(self, src: str = 'x_view', dst: str = 'x_id', copy: bool = False):
		self.src, self.dst, self.copy = src, dst, copy

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		if self.src not in sample:
			raise KeyError(f'missing sample key: {self.src}')
		x = sample[self.src]
		if not self.copy:
			sample[self.dst] = x
			return
		if isinstance(x, np.ndarray):
			sample[self.dst] = x.copy()
			return
		if isinstance(x, torch.Tensor):
			sample[self.dst] = x.clone()
			return
		raise TypeError(f'unsupported type for copy: {type(x).__name__}')


class MaskedSignal:
	"""MaskGenerator を使って x_view にピクセル単位マスクを適用し、
	破壊後テンソルと boolean マスクを sample に格納する。
	- src: 入力キー (H,T) or (C,H,T)
	- dst: 出力キー（破壊後）
	- mask_key: 生成された bool マスク (H,T) の保存先キー
	- mode: 'replace' または 'add'
	"""

	def __init__(
		self,
		generator,  # MaskGenerator インスタンス
		*,
		src: str = 'x_view',
		dst: str = 'x_masked',
		mask_key: str = 'mask_bool',
		mode: Literal['replace', 'add'] | None = None,  # ← 任意化（整合チェック用）
	):
		self.gen = generator
		self.src = src
		self.dst = dst
		self.mask_key = mask_key
		self.mode = mode

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		r = rng or np.random.default_rng()
		x = sample[self.src]
		# MaskGenerator.apply は mode 引数を受けないため渡さない（生成器に保持されている）
		xm, m = self.gen.apply(x, rng=r, mask=None, return_mask=True)
		sample[self.dst] = xm
		sample[self.mask_key] = m


class MakeTimeChannel:
	"""(H,W) -> (H,W) 時刻チャネル（秒）。Crop/Pad後に使うこと"""

	def __init__(self, dst: str = 'time_ch'):
		self.dst = dst

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		H, _ = sample['x_view'].shape
		t = sample['meta']['time_view'].astype(np.float32)
		sample[self.dst] = np.repeat(t[None, :], H, axis=0)


class MakeOffsetChannel:
	"""(H,) オフセットを (H,W) に拡張。normalize=True で z-score"""

	def __init__(self, dst: str = 'offset_ch', normalize: bool = True):
		self.dst, self.normalize = dst, normalize

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		off = sample['meta']['offsets_view'].astype(np.float32)
		if self.normalize:
			s = off.std() + 1e-6
			off = (off - off.mean()) / s
		H, W = sample['x_view'].shape
		sample[self.dst] = np.repeat(off[:, None], W, axis=1)


# ---------- Label producers（ラベルから作る派生物） ----------


class FBGaussMap:
	"""fb_idx_view → ガウスマップ（各有効行の面積=1, CE前提, ガウスはビンindex基準）"""

	def __init__(
		self, dst: str = 'fb_map', sigma: float = 1.5, src: str = 'fb_idx_view'
	):
		if float(sigma) <= 0.0:
			raise ValueError('sigma must be positive')
		self.dst = dst
		self.src = src
		self.sigma = float(sigma)

	def __call__(self, sample: dict[str, Any], rng=None) -> None:
		if 'x_view' not in sample:
			raise KeyError("missing 'x_view'")
		if self.src not in sample['meta']:
			raise KeyError(
				f"missing '{self.src}' (use ProjectToView before FBGaussMap)"
			)

		x_view = _to_numpy(sample['x_view'])
		H, W = x_view.shape

		fb = np.asarray(sample['meta'][self.src], dtype=np.int64)
		if fb.shape[0] != H:
			raise ValueError(f'{self.src} length {fb.shape[0]} != H {H}')

		valid = fb > 0
		y = np.zeros((H, W), dtype=np.float32)
		if np.any(valid):
			mu = fb[valid].astype(np.float32)  # ビンindex中心
			g = gaussian_probs1d_np(mu, self.sigma, W)  # (Nv, W)
			y[valid] = g
		sample[self.dst] = y


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
			a = _to_numpy(v, dtype=self.dtype)
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
