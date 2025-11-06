from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

Tensor = torch.Tensor


class PostprocessCompose:
	"""後処理を順次適用（logits -> logits）。各opは (logits, batch) -> logits | (logits, dict) を返す。
	dict は監視用（ログなど）。集約して返す。
	"""

	def __init__(self, ops: Iterable):
		self.ops = tuple(ops)

	def __call__(self, logits: Tensor, batch: Mapping[str, Any]) -> tuple[Tensor, dict]:
		y = logits
		aux: dict[str, Any] = {}
		for op in self.ops:
			out = op(y, batch)
			if isinstance(out, tuple):
				y, info = out
				aux.update(info)
			else:
				y = out
		return y, aux


from __future__ import annotations

import torch

# 依存: ユーザー提供モジュール
from confidence_from_prob import trace_confidence_from_prob
from gaussian_prior_from_trend import gaussian_prior_from_trend

# 変更点:
# - confidence はエントロピー由来のみを使用
# - robust_linear_trend は w_conf を受け取る設計（prob/dt/conf_* を渡さない）
# - 早期ゲート: バッチ代表自信度が閾値未満なら IRLS/prior をスキップ
# - trace_confidence_from_prob は外部パスから import（要求どおり）
from seisai_pick.trend.confidence_from_prob import trace_confidence_from_prob
from torch import Tensor
from trend_fit import robust_linear_trend


class PostprocessCompose:
	"""後処理を順次適用（logits -> logits）。各opは (logits, batch) -> logits | (logits, dict) を返す。
	dict は監視用（ログなど）。集約して返す。
	"""

	def __init__(self, ops: Iterable):
		self.ops = tuple(ops)

	def __call__(self, logits: Tensor, batch: Mapping[str, Any]) -> tuple[Tensor, dict]:
		y = logits
		aux: dict[str, Any] = {}
		for op in self.ops:
			out = op(y, batch)
			if isinstance(out, tuple):
				y, info = out
				aux.update(info)
			else:
				y = out
		return y, aux


class TrendPriorOp:
	"""logits(B,H,W)を受けて:
	1) softmax→prob
	2) probから entropy-based confidence w_conf を構築
	3) IRLSでtrend t(x)を推定（w_conf を基底重みとして渡す）
	4) trend中心のGaussian priorを作成
	5) gate付きで prior を logit に合成（'logit'）または loss 項を返す（'kl'）
	を行う PostprocessCompose 互換の op。

	必須batchキー:
	- offsets: (B,H) [m] 受信点オフセット
	- fb_idx: (B,H) int  有効トレースは >=0
	- dt_sec: (B,) or (B,1) サンプリング間隔[s]（t重心の計算に使用）
	"""

	def __init__(
		self,
		*,
		offsets_key: str = 'offsets',
		fb_idx_key: str = 'fb_idx',
		dt_key: str = 'dt_sec',
		# confidence 設定（entropy のみ）
		conf_floor: float = 0.2,
		conf_power: float = 0.5,
		# IRLS 設定
		section_len: int = 128,
		stride: int = 64,
		huber_c: float = 1.345,
		iters: int = 3,
		vmin: float = 300.0,
		vmax: float = 8000.0,
		sort_offsets: bool = True,
		use_taper: bool = True,
		# prior 設定
		prior_sigma_ms: float = 20.0,
		prior_mode: str = 'logit',  # 'logit' | 'kl'
		prior_alpha: float = 1.0,
		prior_conf_gate: float = 0.5,
		prior_log_eps: float = 1e-4,
		# 合成・数値安定化
		logit_clip: float = 30.0,
		tau: float = 1.0,  # 'kl'用温度
		# デバッグ
		debug: bool = False,
	):
		assert prior_mode in ('logit', 'kl')
		assert prior_sigma_ms > 0.0
		assert section_len >= 4 and stride >= 1 and iters >= 1
		assert vmin > 0 and vmax > vmin
		self.offsets_key = offsets_key
		self.fb_idx_key = fb_idx_key
		self.dt_key = dt_key

		self.conf_floor = float(conf_floor)
		self.conf_power = float(conf_power)

		self.section_len = int(section_len)
		self.stride = int(stride)
		self.huber_c = float(huber_c)
		self.iters = int(iters)
		self.vmin = float(vmin)
		self.vmax = float(vmax)
		self.sort_offsets = bool(sort_offsets)
		self.use_taper = bool(use_taper)

		self.prior_sigma_ms = float(prior_sigma_ms)
		self.prior_mode = prior_mode
		self.prior_alpha = float(prior_alpha)
		self.prior_conf_gate = float(prior_conf_gate)
		self.prior_log_eps = float(prior_log_eps)

		self.logit_clip = float(logit_clip)
		self.tau = float(tau)
		self.debug = bool(debug)

	@torch.no_grad()
	def __call__(
		self, logits: Tensor, batch: Mapping[str, Any]
	) -> tuple[Tensor, dict[str, Any]]:
		# ---- 入力検証
		assert isinstance(logits, torch.Tensor) and logits.ndim == 3, (
			'logits must be (B,H,W)'
		)
		B, H, W = logits.shape
		assert (
			self.dt_key in batch
			and self.offsets_key in batch
			and self.fb_idx_key in batch
		), (
			f"batch must contain '{self.dt_key}', '{self.offsets_key}', '{self.fb_idx_key}'"
		)
		offsets = batch[self.offsets_key]
		fb_idx = batch[self.fb_idx_key]
		dt_sec = batch[self.dt_key]
		assert isinstance(offsets, torch.Tensor) and offsets.shape == (B, H)
		assert isinstance(fb_idx, torch.Tensor) and fb_idx.shape == (B, H)
		assert (
			isinstance(dt_sec, torch.Tensor)
			and dt_sec.ndim in (1, 2)
			and dt_sec.shape[0] == B
		)

		# ---- prob & エントロピー由来の w_conf を構築
		logit = torch.nan_to_num(
			logits, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip
		).clamp_(-self.logit_clip, self.logit_clip)
		prob = torch.softmax(logit, dim=-1)  # (B,H,W)

		# 外部モジュール（エントロピーのみ）で (B,H) の confidence を作成
		w_conf = trace_confidence_from_prob(
			prob=prob,
			floor=self.conf_floor,
			power=self.conf_power,
		).to(logit)  # (B,H)

		# ---- 早期ゲート（中央値で判定）
		use_mask = fb_idx >= 0
		conf_med = w_conf[use_mask].median() if use_mask.any() else w_conf.median()
		alpha_eff = (
			self.prior_alpha if float(conf_med) >= float(self.prior_conf_gate) else 0.0
		)

		# ---- 初期 t_sec（確率重心）
		t_idx = torch.arange(W, device=logits.device, dtype=logits.dtype).view(1, 1, W)
		if dt_sec.ndim == 1:
			dt = dt_sec.view(B, 1, 1).to(logits)
		else:
			assert dt_sec.shape in [(B, 1), (B, 1, 1)]
			dt = dt_sec.view(B, 1, 1).to(logits)
		t_grid = t_idx * dt  # (B,1,W) [s]
		t_mu = (prob * t_grid).sum(dim=-1)  # (B,H)

		valid = (fb_idx >= 0).to(logits)

		aux: dict[str, Any] = {
			'prior_mode': self.prior_mode,
			'prior_alpha_eff': float(alpha_eff),
			'prior_conf_gate': float(self.prior_conf_gate),
			'conf_med': float(conf_med),
		}

		# 低自信なら heavy 処理をスキップ
		if alpha_eff == 0.0:
			aux.update({'skipped_prior': True})
			# softmax前の最終ガードだけして返す
			logit = torch.nan_to_num(
				logit, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip
			).clamp_(-self.logit_clip, self.logit_clip)
			return logit, aux

		# ---- IRLSで trend 推定（w_conf を渡す）
		trend_t, trend_s, v_trend, w_used, covered = robust_linear_trend(
			offsets=offsets.to(logits),
			t_sec=t_mu.to(logits),
			valid=valid,
			w_conf=w_conf,
			section_len=self.section_len,
			stride=self.stride,
			huber_c=self.huber_c,
			iters=self.iters,
			vmin=self.vmin,
			vmax=self.vmax,
			sort_offsets=self.sort_offsets,
			use_taper=self.use_taper,
		)

		# ---- trend中心の Gaussian prior
		prior = gaussian_prior_from_trend(
			t_trend_sec=trend_t,
			dt_sec=dt_sec,
			W=W,
			sigma_ms=self.prior_sigma_ms,
			ref_tensor=logits,  # dtype/device 合わせ
			covered_mask=covered,  # 未カバーは一様に置換
		)  # (B,H,W)
		# gaussian_prior_from_trend は正規化済みだが、最後の砦として有限化
		prior = torch.nan_to_num(prior, nan=0.0).clamp_(min=0.0)

		# ---- prior 合成/正則化
		if self.prior_mode == 'logit':
			log_prior = torch.log(prior.clamp_min(self.prior_log_eps)).to(torch.float32)

			# 保存: prior 合成前の安全な logit
			logit_base = logit.clone()

			device_type = 'cuda' if logit.is_cuda else 'cpu'
			with torch.autocast(device_type=device_type, enabled=False):
				tmp32 = logit.to(torch.float32) + float(alpha_eff) * log_prior
				# 非有限チェック（ここで一度だけ）
				if not torch.isfinite(tmp32).all():
					if self.debug:
						print(
							'[trend prior] non-finite after add -> disable prior for this batch'
						)
					aux['prior_disabled_reason'] = 'non_finite_after_add'
					logit_work = logit_base
				else:
					logit_work = tmp32

			# 最終サニタイズは1回だけ
			logit = (
				torch.nan_to_num(
					logit_work, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip
				)
				.clamp_(-self.logit_clip, self.logit_clip)
				.to(logits.dtype)
			)

			# 監視用
			aux.update(
				{
					'trend_t': trend_t,
					'v_trend': v_trend,
					'w_conf': w_used,
					'covered': covered,
				}
			)
			return logit, aux

		if self.prior_mode == 'kl':
			# logits は変更せず、prior に近づける正則化項を返す
			log_p = torch.log_softmax(logit / self.tau, dim=-1)
			if (not torch.isfinite(log_p).all()) or (not torch.isfinite(prior).all()):
				if self.debug:
					print(
						'[trend prior KL] non-finite in prior/log_p -> disable prior for this batch'
					)
				aux['prior_alpha_eff'] = 0.0
				prior_ce = logit.new_tensor(0.0)
			else:
				kl_bh = -(prior * log_p).sum(dim=-1)  # (B,H)
				use = (fb_idx >= 0) & covered
				prior_ce = kl_bh[use].mean() if use.any() else kl_bh.mean()

			aux.update(
				{
					'prior_ce': prior_ce,
					'trend_t': trend_t,
					'v_trend': v_trend,
					'w_conf': w_used,
					'covered': covered,
				}
			)
			return logits, aux

		raise ValueError(f'Unknown prior_mode: {self.prior_mode}')
