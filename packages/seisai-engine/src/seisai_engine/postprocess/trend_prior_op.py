# %%
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from seisai_pick.trend._time_pick import _argmax_time_parabolic
from seisai_pick.trend.confidence_from_prob import trace_confidence_from_prob
from seisai_pick.trend.gaussian_prior_from_trend import gaussian_prior_from_trend
from seisai_pick.trend.trend_fit_strategy import (
	IRLSStrategy,
	TrendFitStrategy,
)
from torch import Tensor
from torch.amp import autocast


@dataclass(frozen=True)
class TrendPriorConfig:
	# batch keys
	offsets_key: str = 'offsets'  # (B,H) [m]
	fb_idx_key: str = 'fb_idx'  # (B,H) int (valid: >=0)
	dt_key: str = 'dt_sec'  # (B,) or (B,1) or (B,1,1) [s]

	# confidence (entropy-based)
	conf_floor: float = 0.2
	conf_power: float = 0.5

	# trend fit strategy (DI)
	fit: TrendFitStrategy = IRLSStrategy()

	# prior (Gaussian around trend)
	prior_sigma_ms: float = 20.0
	prior_alpha: float = 1.0
	prior_conf_gate: float = 0.5
	prior_log_eps: float = 1e-4

	# numerics
	logit_clip: float = 30.0

	# channels to apply (0 / [0,2] / range(C))
	channels: int | Iterable[int] = 0

	# aux prefix
	aux_key: str = 'trend_prior'

	# debug
	debug: bool = False


def _resolve_channels(ch_spec: int | Iterable[int], C: int) -> Sequence[int]:
	if isinstance(ch_spec, int):
		return [int(ch_spec)]
	return [int(c) for c in ch_spec]


class TrendPriorOp:
	"""(B,C,H,W) logits を受け取り、選択チャネルに trend 中心の Gaussian prior を
	log 空間で合成して返す。prior_mode='logit' のみを提供（損失計算は別モジュール）。
	トレンド推定は Strategy インスタンス（IRLS/RANSAC等）で差し替え。
	"""

	def __init__(self, cfg: TrendPriorConfig) -> None:
		assert cfg.prior_sigma_ms > 0.0 and cfg.prior_alpha >= 0.0
		self.cfg = cfg

	@torch.no_grad()
	def __call__(
		self, logits: Tensor, batch: Mapping[str, Any]
	) -> tuple[Tensor, dict[str, Any]]:
		# ---- 入力検証
		assert isinstance(logits, torch.Tensor) and logits.ndim == 4, (
			'logits must be (B,C,H,W)'
		)
		B, C, H, W = logits.shape

		cfg = self.cfg
		assert (
			(cfg.dt_key in batch)
			and (cfg.offsets_key in batch)
			and (cfg.fb_idx_key in batch)
		), f"batch must contain '{cfg.dt_key}', '{cfg.offsets_key}', '{cfg.fb_idx_key}'"

		offsets: Tensor = batch[cfg.offsets_key]
		fb_idx: Tensor = batch[cfg.fb_idx_key]
		dt_sec: Tensor = batch[cfg.dt_key]
		assert isinstance(offsets, torch.Tensor) and offsets.shape == (B, H)
		assert isinstance(fb_idx, torch.Tensor) and fb_idx.shape == (B, H)

		valid = fb_idx >= 0
		chs = _resolve_channels(cfg.channels, C)

		out = logits.clone()
		aux: dict[str, Any] = {
			'prior_mode': 'logit',
			'prior_alpha': float(cfg.prior_alpha),
			'fit': getattr(cfg.fit, 'name', 'custom'),
		}

		for c in chs:
			# --- logits サニタイズ & prob ---
			logit = torch.nan_to_num(
				logits[:, c], nan=0.0, posinf=cfg.logit_clip, neginf=-cfg.logit_clip
			).clamp_(-cfg.logit_clip, cfg.logit_clip)  # (B,H,W)
			prob = torch.softmax(logit, dim=-1)

			# --- trace confidence (entropy-based) ---
			w_conf = trace_confidence_from_prob(
				prob=prob, floor=cfg.conf_floor, power=cfg.conf_power
			).to(logit)  # (B,H)

			# --- 早期ゲート（中央値） ---
			use_mask = valid
			conf_med = w_conf[use_mask].median() if use_mask.any() else w_conf.median()
			alpha_eff = (
				cfg.prior_alpha
				if float(conf_med) >= float(cfg.prior_conf_gate)
				else 0.0
			)
			aux[f'{cfg.aux_key}_conf_med_ch{c}'] = float(conf_med)
			aux[f'{cfg.aux_key}_alpha_eff_ch{c}'] = float(alpha_eff)

			if alpha_eff == 0.0:
				out[:, c] = logit.to(logits.dtype)
				aux[f'{cfg.aux_key}_skipped_ch{c}'] = True
				continue

			# --- 到達候補: 確率重心 t_mu ---
			t_sec = _argmax_time_parabolic(prob, dt_sec)  # (B,H) [s]

			# --- トレンド推定（Strategy 呼び出し） ---
			trend_t, trend_s, v_trend, w_used, covered = cfg.fit(
				offsets=offsets.to(logit),
				t_sec=t_sec.to(logit),
				valid=valid,
				w_conf=w_conf,
			)

			# --- Gaussian prior（trend中心） ---
			prior = (
				gaussian_prior_from_trend(
					t_trend_sec=trend_t,
					dt_sec=dt_sec,
					W=W,
					sigma_ms=cfg.prior_sigma_ms,
					ref_tensor=logit,
					covered_mask=covered,
				)
				.nan_to_num(0.0)
				.clamp_(min=0.0)
			)  # (B,H,W)

			# --- logit 合成（log prior を加算） ---
			log_prior = torch.log(prior.clamp_min(cfg.prior_log_eps)).to(torch.float32)

			device_type = 'cuda' if logit.is_cuda else 'cpu'
			with autocast(device_type, enabled=False):
				fused32 = logit.to(torch.float32) + float(alpha_eff) * log_prior
				if not torch.isfinite(fused32).all():
					aux[f'{cfg.aux_key}_disabled_reason_ch{c}'] = 'non_finite_after_add'
					fused = logit
				else:
					fused = fused32

			fused = (
				torch.nan_to_num(
					fused, nan=0.0, posinf=cfg.logit_clip, neginf=-cfg.logit_clip
				)
				.clamp_(-cfg.logit_clip, cfg.logit_clip)
				.to(logits.dtype)
			)

			out[:, c] = fused

			# 診断
			aux[f'{cfg.aux_key}_trend_t_ch{c}'] = trend_t
			aux[f'{cfg.aux_key}_v_trend_ch{c}'] = v_trend
			aux[f'{cfg.aux_key}_w_conf_ch{c}'] = w_used
			aux[f'{cfg.aux_key}_covered_ch{c}'] = covered

		return out, aux
