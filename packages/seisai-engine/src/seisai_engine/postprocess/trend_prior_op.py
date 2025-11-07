# TrendPriorOp: (B,C,H,W) 専用 + Configあり + 対象チャンネルだけ(B,H,W)に落として処理→戻す
# ============================================================
# File: packages/seisai-engine/src/seisai_engine/postprocess/trend_prior_op.py
# 目的: prior_mode='logit' 専用の logits 変換器（学習用の損失計算は別モジュールへ）
# 依存: torch, seisai_pick.trend.*
# ============================================================
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from seisai_pick.trend.confidence_from_prob import trace_confidence_from_prob
from seisai_pick.trend.gaussian_prior_from_trend import gaussian_prior_from_trend
from seisai_pick.trend.trend_fit import robust_linear_trend
from torch import Tensor

# -------------------- Config --------------------


@dataclass(frozen=True)
class TrendPriorConfig:
	# batch keys
	offsets_key: str = 'offsets'  # (B,H) [m]
	fb_idx_key: str = 'fb_idx'  # (B,H) int (valid: >=0)
	dt_key: str = 'dt_sec'  # (B,) or (B,1) or (B,1,1) [s]
	# confidence (entropy-based)
	conf_floor: float = 0.2
	conf_power: float = 0.5
	# IRLS (robust linear trend)
	section_len: int = 128
	stride: int = 64
	huber_c: float = 1.345
	iters: int = 3
	vmin: float = 300.0
	vmax: float = 8000.0
	sort_offsets: bool = True
	use_taper: bool = True
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


# -------------------- Op (logit 合成のみ) --------------------


class TrendPriorOp:
	"""(B,C,H,W) logits を受け取り、選択チャネルに Gaussian prior（trend中心）を
	log 空間で合成して返す。prior_mode='logit' のみを提供（損失計算は別モジュール）。
	"""

	def __init__(self, cfg: TrendPriorConfig) -> None:
		assert cfg.vmin > 0.0 and cfg.vmax > cfg.vmin
		assert cfg.section_len >= 4 and cfg.stride >= 1 and cfg.iters >= 1
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

		# 時間グリッド
		if dt_sec.ndim == 1 or dt_sec.ndim == 2:
			dt = dt_sec.view(B, 1, 1).to(logits)
		else:
			assert dt_sec.shape in [(B, 1, 1)], 'dt_sec must be (B,), (B,1), or (B,1,1)'
			dt = dt_sec.to(logits)
		t_idx = torch.arange(W, device=logits.device, dtype=logits.dtype).view(
			1, 1, W
		)  # (1,1,W)
		t_grid = t_idx * dt  # (B,1,W) [s]

		valid = fb_idx >= 0
		chs = _resolve_channels(cfg.channels, C)

		# 出力バッファ（コピーして上書き）
		out = logits.clone()
		aux: dict[str, Any] = {
			'prior_mode': 'logit',
			'prior_alpha': float(cfg.prior_alpha),
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
				# 合成をスキップ（安全のため再サニタイズのみ）
				out[:, c] = logit.to(logits.dtype)
				aux[f'{cfg.aux_key}_skipped_ch{c}'] = True
				continue

			# --- t_mu（確率重心） ---
			t_mu = (prob * t_grid).sum(dim=-1)  # (B,H) [s]

			# --- robust trend fit ---
			trend_t, trend_s, v_trend, w_used, covered = robust_linear_trend(
				offsets=offsets.to(logit),
				t_sec=t_mu.to(logit),
				valid=valid.to(logit.dtype),
				w_conf=w_conf,
				section_len=cfg.section_len,
				stride=cfg.stride,
				huber_c=cfg.huber_c,
				iters=cfg.iters,
				vmin=cfg.vmin,
				vmax=cfg.vmax,
				sort_offsets=cfg.sort_offsets,
				use_taper=cfg.use_taper,
			)

			# --- Gaussian prior（trend中心） ---
			prior = (
				gaussian_prior_from_trend(
					t_trend_sec=trend_t,
					dt_sec=dt_sec,
					W=W,
					sigma_ms=cfg.prior_sigma_ms,
					ref_tensor=logit,  # dtype/device 整合
					covered_mask=covered,  # 未カバーは一様
				)
				.nan_to_num(0.0)
				.clamp_(min=0.0)
			)  # (B,H,W)

			# --- logit 合成（log prior を加算） ---
			log_prior = torch.log(prior.clamp_min(cfg.prior_log_eps)).to(torch.float32)

			device_type = 'cuda' if logit.is_cuda else 'cpu'
			with torch.autocast(device_type=device_type, enabled=False):
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
