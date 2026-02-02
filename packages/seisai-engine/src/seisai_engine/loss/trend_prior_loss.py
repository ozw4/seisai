# ============================================================
# File: packages/seisai-engine/src/seisai_engine/postprocess/trend_prior_loss.py
# 依存: torch, seisai_engine.postprocess.trend_fit_strategy, seisai_pick.trend.*
# IF:
#   loss = TrendPriorCELoss(cfg)(logits, batch, reduction='mean', return_aux=False)
#   # cfg は少なくとも以下の属性を持つこと:
#   #  - offsets_key, fb_idx_key, dt_key
#   #  - conf_floor, conf_power
#   #  - fit: TrendFitStrategy インスタンス(IRLSStrategy/RANSACStrategy等)
#   #  - prior_sigma_ms, prior_alpha, prior_conf_gate, prior_log_eps
#   #  - logit_clip, channels, aux_key
#   #  - tau (>0)
# ============================================================
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import torch
from seisai_pick.trend._time_pick import _argmax_time_parabolic
from seisai_pick.trend.confidence_from_prob import trace_confidence_from_prob
from seisai_pick.trend.gaussian_prior_from_trend import gaussian_prior_from_trend
from torch import Tensor

Reduction = Literal['mean', 'sum', 'none']


def _resolve_channels(ch_spec: int | Iterable[int], C: int) -> Sequence[int]:
	if isinstance(ch_spec, int):
		return [int(ch_spec)]
	return [int(c) for c in ch_spec]


class TrendPriorCELoss:
	"""Trend Prior の CE/KL 正則化をクラス化(戦略DI対応, t抽出=放物線argmax)。
	Args:
	  cfg: TrendPrior 用設定。少なくとも本ファイル冒頭の IF コメントで列挙した属性を持つこと。
	Note:
	  - reduction は 'mean' のみ許容(スカラー返却)。'sum'/'none'は未対応。
	  - return_aux=True の場合は (loss, aux) を返す。
	"""

	def __init__(self, cfg: Any):
		self.cfg = cfg

	def __call__(
		self,
		pred: Tensor,
		batch: Mapping[str, Any],
		*,
		reduction: Reduction = 'mean',
		return_aux: bool = False,
	) -> Tensor | tuple[Tensor, dict]:
		assert isinstance(pred, torch.Tensor) and pred.ndim == 4, (
			'pred(logits) must be (B,C,H,W)'
		)
		assert reduction == 'mean', "TrendPriorCELoss only supports reduction='mean'"

		loss, aux = trend_prior_ce(pred, batch, self.cfg)  # スカラーTensor, dict
		return (loss, aux) if return_aux else loss


@torch.no_grad()
def trend_prior_ce(
	logits: Tensor,  # (B,C,H,W)
	batch: Mapping[str, Any],
	cfg: Any,  # 必須属性は冒頭コメントを参照
) -> tuple[Tensor, dict[str, Any]]:
	"""Trend Prior の KL/CE 正則化(教師=prior, 学習対象=logits/τ)。戦略DI対応。
	t抽出は期待値ではなく放物線argmaxを用いる。
	"""
	assert logits.ndim == 4, 'logits must be (B,C,H,W)'
	B, C, H, W = logits.shape

	# ---- cfg 必須属性チェック(フォールバック禁止)
	for k in (
		'offsets_key',
		'fb_idx_key',
		'dt_key',
		'conf_floor',
		'conf_power',
		'fit',
		'prior_sigma_ms',
		'prior_alpha',
		'prior_conf_gate',
		'prior_log_eps',
		'logit_clip',
		'channels',
		'aux_key',
		'tau',
	):
		assert hasattr(cfg, k), f'cfg.{k} is required'
	# StrategyはProtocol想定(@runtime_checkable未使用)なのでcallableでチェック
	assert callable(cfg.fit), 'cfg.fit must be a callable TrendFitStrategy-like object'
	assert float(cfg.tau) > 0.0 and float(cfg.prior_sigma_ms) > 0.0

	assert (
		cfg.dt_key in batch and cfg.offsets_key in batch and cfg.fb_idx_key in batch
	), f"batch must contain '{cfg.dt_key}', '{cfg.offsets_key}', '{cfg.fb_idx_key}'"

	offsets: Tensor = batch[cfg.offsets_key]  # (B,H)
	fb_idx: Tensor = batch[cfg.fb_idx_key]  # (B,H)
	dt_sec: Tensor = batch[cfg.dt_key]  # (B,) or (B,1) or (B,1,1)
	assert offsets.shape == (B, H) and fb_idx.shape == (B, H)

	chs = _resolve_channels(cfg.channels, C)
	assert len(chs) >= 1

	valid = (fb_idx >= 0).to(dtype=torch.bool, device=logits.device)

	prior_ce_acc = logits.new_tensor(0.0)
	aux: dict[str, Any] = {
		'prior_mode': 'kl',
		'tau': float(cfg.tau),
		'fit': getattr(cfg.fit, 'name', 'custom'),
		't_pick': 'parabolic_argmax',
	}
	num_used_ch = 0

	for c in chs:
		# --- prob と confidence(トレース毎) ---
		logit_c = torch.nan_to_num(
			logits[:, c], nan=0.0, posinf=cfg.logit_clip, neginf=-cfg.logit_clip
		).clamp_(-cfg.logit_clip, cfg.logit_clip)  # (B,H,W)
		prob_c = torch.softmax(logit_c, dim=-1)  # (B,H,W)

		w_conf = trace_confidence_from_prob(
			prob=prob_c, floor=float(cfg.conf_floor), power=float(cfg.conf_power)
		).to(logit_c)  # (B,H)

		# --- 早期ゲート(中央値)
		use_mask = valid
		conf_med = w_conf[use_mask].median() if use_mask.any() else w_conf.median()
		alpha_eff = (
			float(cfg.prior_alpha)
			if float(conf_med) >= float(cfg.prior_conf_gate)
			else 0.0
		)
		aux[f'{cfg.aux_key}_conf_med_ch{c}'] = float(conf_med)
		aux[f'{cfg.aux_key}_alpha_eff_ch{c}'] = float(alpha_eff)
		if alpha_eff == 0.0:
			continue  # このチャネルは寄与0

		# --- 到達候補: 放物線argmax t_sec(期待値ではない) ---
		t_sec = _argmax_time_parabolic(prob_c, dt_sec)  # (B,H) [s]

		# --- トレンド推定(Strategy をそのまま呼び出し)
		trend_t, trend_s, v_trend, w_used, covered = cfg.fit(
			offsets=offsets.to(logit_c),
			t_sec=t_sec.to(logit_c),
			valid=valid,
			w_conf=w_conf,
		)  # すべて (B,H)

		# --- ガウス prior(trend 中心)
		prior = (
			gaussian_prior_from_trend(
				t_trend_sec=trend_t,
				dt_sec=dt_sec,
				W=W,
				sigma_ms=float(cfg.prior_sigma_ms),
				ref_tensor=logit_c,  # dtype/device 整合
				covered_mask=covered,  # 未カバーは一様
			)
			.nan_to_num(0.0)
			.clamp_(min=0.0)
		)  # (B,H,W)

		# --- KL/CE(prior を教師分布として CE を計算)
		log_p = torch.log_softmax(logit_c / float(cfg.tau), dim=-1)  # (B,H,W)
		assert torch.isfinite(log_p).all() and torch.isfinite(prior).all(), (
			'non-finite in prior/log_p'
		)

		ce_bh = -(prior * log_p).sum(dim=-1)  # (B,H)
		use = valid & covered
		ce = ce_bh[use].mean() if use.any() else ce_bh.mean()

		prior_ce_acc = prior_ce_acc + ce
		num_used_ch += 1

		# 診断
		aux[f'{cfg.aux_key}_trend_t_ch{c}'] = trend_t
		aux[f'{cfg.aux_key}_v_trend_ch{c}'] = v_trend
		aux[f'{cfg.aux_key}_w_conf_ch{c}'] = w_used
		aux[f'{cfg.aux_key}_covered_ch{c}'] = covered

	assert num_used_ch > 0, (
		'no channel contributed to prior_ce (likely gated by prior_conf_gate)'
	)
	return prior_ce_acc / float(num_used_ch), aux
