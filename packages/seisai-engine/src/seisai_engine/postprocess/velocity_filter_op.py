# ============================================================
# File: packages/seisai-engine/src/seisai_engine/postprocess/velocity_mask.py
# 依存: torch
# 目的: 速度円錐に基づく可到達領域マスクの生成と、logits/prob への適用
# 仕様: (B,C,H,W) 専用（Compose互換）。対象チャンネルのみ (B,H,W) に落として処理→戻す。
# ============================================================
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

# -------------------- マスク生成 --------------------


def _cos_ramp01(s: Tensor) -> Tensor:
	# s in [0,1] -> [0,1] のコサインランプ。外側は0
	s = s.clamp_(0.0, 1.0)
	return 0.5 - 0.5 * torch.cos(torch.pi * s)


def make_velocity_feasible_filt(
	*,
	offsets_m: Tensor,  # (B,H) [m]
	dt_sec: Tensor | float,  # (B,) / (B,1) / (B,1,1) / scalar [s]
	W: int,  # time bins
	vmin: float,
	vmax: float,
	t0_lo_ms: float = 0.0,  # [ms] 下側の余裕
	t0_hi_ms: float = 0.0,  # [ms] 上側の余裕
	taper_ms: float = 0.0,  # [ms] 境界のソフト化幅。0でハード
	device: torch.device | None = None,
	dtype: torch.dtype | None = None,
) -> Tensor:
	assert vmin > 0.0 and vmax > 0.0 and vmax >= vmin
	assert W > 0
	B, H = offsets_m.shape
	dev = device or offsets_m.device

	if isinstance(dt_sec, torch.Tensor):
		if dt_sec.ndim == 1:
			dt = dt_sec.view(B, 1, 1)
		elif dt_sec.ndim == 2:
			assert dt_sec.shape[1] == 1
			dt = dt_sec.view(B, 1, 1)
		else:
			assert dt_sec.shape[1:] in [(1, 1)]
			dt = dt_sec.view(B, 1, 1)
		dt = dt.to(device=dev, dtype=torch.float32)
	else:
		dt = torch.tensor(dt_sec, device=dev, dtype=torch.float32).view(B, 1, 1)

	t = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W) * dt  # (B,1,W)
	x = offsets_m.to(device=dev, dtype=torch.float32).abs().view(B, H, 1)  # (B,H,1)

	t_lo = x / float(vmax) + (t0_lo_ms / 1000.0)  # (B,H,1)
	t_hi = x / float(vmin) + (t0_hi_ms / 1000.0)  # (B,H,1)

	m = torch.zeros((B, H, W), device=dev, dtype=torch.float32)

	inside = (t >= t_lo) & (t <= t_hi)
	m[inside] = 1.0

	if taper_ms > 0.0:
		w = float(taper_ms) / 1000.0
		# 下側ランプ: [t_lo - w, t_lo] で 0->1
		s_lo = (t - (t_lo - w)) / w
		lower = _cos_ramp01(s_lo) * (t <= t_lo)
		# 上側ランプ: [t_hi, t_hi + w] で 1->0
		s_hi = (t_hi + w - t) / w
		upper = _cos_ramp01(s_hi) * (t >= t_hi)
		m = torch.maximum(m, lower)
		m = torch.maximum(m, upper)

	if dtype is not None and m.dtype != dtype:
		m = m.to(dtype=dtype)
	return m  # (B,H,W) in [0,1]


# -------------------- 適用（logits / prob） --------------------


def apply_velocity_filt_logits(
	logits: Tensor,  # (B,C,H,W)
	mask: Tensor,  # (B,H,W)
	*,
	power: float = 1.0,
) -> Tensor:
	"""softmax前の logits に log(mask^power) を加算。
	mask=0 -> -inf 相当となり、softmax 後は厳密に 0。
	"""
	if power != 1.0:
		mask = mask.pow(power)
	# 0 -> -inf を厳密に作る
	neg_inf = torch.tensor(float('-inf'), device=mask.device, dtype=mask.dtype)
	log_m = torch.where(mask > 0, mask.log(), neg_inf).unsqueeze(1)  # (B,1,H,W)
	return logits + log_m.to(dtype=logits.dtype)


def apply_velocity_filt_prob(
	prob: Tensor,  # (B,C,H,W)
	mask: Tensor,  # (B,H,W)
	*,
	power: float = 1.0,
	time_dim: int = -1,
	renorm: bool = True,
	eps: float = 1e-12,
) -> Tensor:
	"""softmax後の確率に mask^power を乗算。必要なら再正規化（分布の総和=1）。"""
	m = (
		(mask.pow(power) if power != 1.0 else mask).unsqueeze(1).to(prob.dtype)
	)  # (B,1,H,W)
	y = prob * m
	if renorm:
		denom = y.sum(dim=time_dim, keepdim=True).clamp_min(
			torch.as_tensor(eps, device=y.device, dtype=y.dtype)
		)
		y = y / denom
	return y


# -------------------- Compose 連携用の Op（(B,C,H,W) 専用） --------------------


@dataclass(frozen=True)
class VelocityFiltConfig:
	# 適用先（学習=logits 推奨。可視化/後段整形で prob を使うなら "prob"）
	operate_on: Literal['logits', 'prob'] = 'logits'
	# 速度制約
	vmin: float = 1500.0
	vmax: float = 4500.0
	t0_lo_ms: float = 0.0
	t0_hi_ms: float = 0.0
	taper_ms: float = 8.0
	power: float = 1.0
	# 形状軸
	time_dim: int = -1  # W 軸
	# バッチから参照するキー
	offsets_key: str = 'offsets'  # (B,H)
	dt_key: str = 'dt_sec'  # (B,) / (B,1) / (B,1,1) / scalar
	# 複数チャンネル対応
	channels: int | Iterable[int] = 0  # 例: 0, [0,2], range(C)
	# prob用のオプション
	renorm: bool = True
	eps: float = 1e-12
	# auxキー接頭辞
	aux_key: str = 'velocity_mask'


class ApplyVelocityFiltOp:
	"""PostprocessCompose 互換の op（非in-place）。
	(logits, batch) -> (logits, aux)。入力は (B,C,H,W) のみを受け付ける。
	指定チャンネルのみ (B,H,W) に落としてマスク適用→(B,C,H,W) に戻す。
	"""

	def __init__(self, cfg: VelocityMaskConfig) -> None:
		self.cfg = cfg
		chs = [cfg.channels] if isinstance(cfg.channels, int) else list(cfg.channels)
		assert len(chs) > 0
		self._channels = [int(c) for c in chs]

	def _norm_channels(self, C: int) -> list[int]:
		seen: set[int] = set()
		out: list[int] = []
		for c in self._channels:
			assert 0 <= c < C, f'channel out of range: {c} not in [0,{C - 1}]'
			if c not in seen:
				seen.add(c)
				out.append(c)
		return out

	@torch.no_grad()
	def __call__(self, logits: Tensor, batch: Mapping[str, Any]) -> tuple[Tensor, dict]:
		cfg = self.cfg
		# ---- 入力検証（4Dのみ）
		assert isinstance(logits, torch.Tensor) and logits.ndim == 4, (
			'logits must be (B,C,H,W)'
		)
		B, C, H, W = logits.shape
		ch_list = self._norm_channels(C)

		# ---- batch から offsets / dt を取得し検証
		assert cfg.offsets_key in batch and cfg.dt_key in batch, (
			f"batch must contain '{cfg.offsets_key}', '{cfg.dt_key}'"
		)
		offsets = batch[cfg.offsets_key]
		dt_sec = batch[cfg.dt_key]
		assert isinstance(offsets, torch.Tensor) and offsets.shape == (B, H)
		assert isinstance(dt_sec, (torch.Tensor, float))

		# ---- (B,H,W) の連続マスクを生成
		mask = make_velocity_feasible_mask(
			offsets_m=offsets,
			dt_sec=dt_sec,
			W=W,
			vmin=cfg.vmin,
			vmax=cfg.vmax,
			t0_lo_ms=cfg.t0_lo_ms,
			t0_hi_ms=cfg.t0_hi_ms,
			taper_ms=cfg.taper_ms,
			device=logits.device,
			dtype=torch.float32,
		)
		if cfg.power != 1.0:
			mask = mask.pow(cfg.power)

		out = logits.clone()
		aux: dict[str, Any] = {
			f'{cfg.aux_key}_channels': ch_list,
			f'{cfg.aux_key}_mask': mask,
		}

		if cfg.operate_on == 'logits':
			# 0 -> -inf 厳密対応
			neg_inf = torch.tensor(float('-inf'), device=mask.device, dtype=mask.dtype)
			log_m = (
				torch.where(mask > 0, mask.log(), neg_inf)
				.to(dtype=logits.dtype)
				.unsqueeze(1)
			)  # (B,1,H,W)
			for ch in ch_list:
				out[:, ch, :, :] = out[:, ch, :, :] + log_m[:, 0, :, :]
		else:
			# prob 乗算 + 再正規化（必要なら）
			m = mask.unsqueeze(1).to(out.dtype)  # (B,1,H,W)
			for ch in ch_list:
				y = out[:, ch, :, :] * m[:, 0, :, :]  # (B,H,W)
				if cfg.renorm:
					denom = y.sum(dim=cfg.time_dim, keepdim=True).clamp_min(
						torch.as_tensor(cfg.eps, device=y.device, dtype=y.dtype)
					)
					y = y / denom
				out[:, ch, :, :] = y

		return out, aux
