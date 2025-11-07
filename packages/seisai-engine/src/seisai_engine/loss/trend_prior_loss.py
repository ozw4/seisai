from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from seisai_pick.trend.confidence_from_prob import trace_confidence_from_prob
from seisai_pick.trend.gaussian_prior_from_trend import gaussian_prior_from_trend
from seisai_pick.trend.trend_fit import robust_linear_trend
from torch import Tensor

# TrendPriorConfig は既存のものを import してください
# from seisai_engine.postprocess.trend_prior_op import TrendPriorConfig


def _resolve_channels(ch_spec: int | Iterable[int], C: int) -> Sequence[int]:
	if isinstance(ch_spec, int):
		return [int(ch_spec)]
	return [int(c) for c in ch_spec]


@torch.no_grad()
def trend_prior_ce(
	logits: Tensor,  # (B,C,H,W)
	batch: Mapping[str, Any],
	cfg,  # TrendPriorConfig と同等の属性を持つ設定
) -> tuple[Tensor, dict[str, Any]]:
	"""TrendPrior の KL/CE 正則化項のみを計算して返す。

	Returns
	-------
	prior_ce : 0-dim Tensor（スカラー）。選択チャネル平均の CE（=KL相当）
	aux      : 診断情報（各 ch の t_trend、被覆、信頼度など）

	"""
	assert logits.ndim == 4, 'logits must be (B,C,H,W)'
	B, C, H, W = logits.shape
	assert (
		cfg.tau > 0.0
		and cfg.prior_sigma_ms > 0.0
		and cfg.vmin > 0
		and cfg.vmax > cfg.vmin
	)

	assert cfg.dt_key in batch and cfg.offsets_key in batch and cfg.fb_idx_key in batch
	offsets: Tensor = batch[cfg.offsets_key]  # (B,H)
	fb_idx: Tensor = batch[cfg.fb_idx_key]  # (B,H)
	dt_sec: Tensor = batch[cfg.dt_key]  # (B,) or (B,1) or (B,1,1)
	assert offsets.shape == (B, H) and fb_idx.shape == (B, H)

	chs = _resolve_channels(cfg.channels, C)
	assert len(chs) >= 1

	# 時間グリッド（確率重心 t_mu に使用）
	if dt_sec.ndim == 1 or dt_sec.ndim == 2:
		dt = dt_sec.view(B, 1, 1).to(logits)
	else:
		assert dt_sec.shape in [(B, 1, 1)], 'dt_sec must be (B,), (B,1), or (B,1,1)'
		dt = dt_sec.to(logits)
	t_idx = torch.arange(W, device=logits.device, dtype=logits.dtype).view(
		1, 1, W
	)  # (1,1,W)
	t_grid = t_idx * dt  # (B,1,W) [s]

	valid = (fb_idx >= 0).to(dtype=torch.bool, device=logits.device)

	prior_ce_acc = logits.new_tensor(0.0)
	aux: dict[str, Any] = {'prior_mode': 'kl', 'tau': float(cfg.tau)}
	num_used_ch = 0

	for c in chs:
		# --- prob と confidence（トレース毎） ---
		logit_c = torch.nan_to_num(
			logits[:, c], nan=0.0, posinf=cfg.logit_clip, neginf=-cfg.logit_clip
		).clamp_(-cfg.logit_clip, cfg.logit_clip)  # (B,H,W)
		prob_c = torch.softmax(logit_c, dim=-1)  # (B,H,W)

		w_conf = trace_confidence_from_prob(
			prob=prob_c, floor=cfg.conf_floor, power=cfg.conf_power
		).to(logit_c)  # (B,H)

		# 早期ゲート（中央値）
		use_mask = valid
		conf_med = w_conf[use_mask].median() if use_mask.any() else w_conf.median()
		alpha_eff = (
			cfg.prior_alpha if float(conf_med) >= float(cfg.prior_conf_gate) else 0.0
		)
		aux[f'{cfg.aux_key}_conf_med_ch{c}'] = float(conf_med)
		aux[f'{cfg.aux_key}_alpha_eff_ch{c}'] = float(alpha_eff)
		if alpha_eff == 0.0:
			continue  # このチャネルは寄与0

		# --- t_mu（確率重心）とロバスト直線トレンド ---
		t_mu = (prob_c * t_grid).sum(dim=-1)  # (B,H) [s]

		trend_t, trend_s, v_trend, w_used, covered = robust_linear_trend(
			offsets=offsets.to(logit_c),
			t_sec=t_mu.to(logit_c),
			valid=valid,
			w_conf=w_conf,
			section_len=cfg.section_len,
			stride=cfg.stride,
			huber_c=cfg.huber_c,
			iters=cfg.iters,
			vmin=cfg.vmin,
			vmax=cfg.vmax,
			sort_offsets=cfg.sort_offsets,
			use_taper=cfg.use_taper,
		)  # trend_t: (B,H)

		# --- ガウス prior（trend 中心） ---
		prior = (
			gaussian_prior_from_trend(
				t_trend_sec=trend_t,
				dt_sec=dt_sec,
				W=W,
				sigma_ms=cfg.prior_sigma_ms,
				ref_tensor=logit_c,  # dtype/device 整合
				covered_mask=covered,  # 未カバーは一様
			)
			.nan_to_num(0.0)
			.clamp_(min=0.0)
		)  # (B,H,W)

		# --- KL/CE（prior を教師分布として CE を計算） ---
		log_p = torch.log_softmax(logit_c / float(cfg.tau), dim=-1)  # (B,H,W)
		if (not torch.isfinite(log_p).all()) or (not torch.isfinite(prior).all()):
			aux[f'{cfg.aux_key}_disabled_reason_ch{c}'] = 'non_finite_in_prior_or_logp'
			continue

		ce_bh = -(prior * log_p).sum(dim=-1)  # (B,H)
		use = valid & covered
		ce = ce_bh[use].mean() if use.any() else ce_bh.mean()

		prior_ce_acc = prior_ce_acc + ce
		num_used_ch += 1

		# 診断保存
		aux[f'{cfg.aux_key}_trend_t_ch{c}'] = trend_t
		aux[f'{cfg.aux_key}_v_trend_ch{c}'] = v_trend
		aux[f'{cfg.aux_key}_w_conf_ch{c}'] = w_used
		aux[f'{cfg.aux_key}_covered_ch{c}'] = covered

	# チャネル平均（有効チャネルのみ）
	if num_used_ch == 0:
		return logits.new_tensor(0.0), aux
	prior_ce = prior_ce_acc / float(num_used_ch)
	return prior_ce, aux
