"""Two-piece physical-center fitting helpers."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from seisai_pick.trend.trend_fit_strategy import (
    PiecewiseLinearTrend,
    TwoPieceIRLSAutoBreakStrategy,
    TwoPieceRansacAutoBreakStrategy,
)

from .physical_center_observation import _indices_key
from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_FIT_FAILED,
    PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
)
from .progress import build_progress_reporter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import PhysicsLiteConfig
    from .physical_center_observation import _ObservationPlan
    from .runtime_diagnostics import PhysicalRuntimeDiagnostics

MODEL_TYPE_TWO_PIECE = 'two_piece'
MODEL_TYPE_SINGLE_LINE = 'single_line'

_MODEL_CODE_BY_TYPE = {
    MODEL_TYPE_TWO_PIECE: 1,
    MODEL_TYPE_SINGLE_LINE: 2,
}
_MODEL_TYPE_BY_CODE = {value: key for key, value in _MODEL_CODE_BY_TYPE.items()}


@dataclass(frozen=True)
class _SelectedPhysicalTrend:
    base_model: object
    model_type: str
    selected_model: str
    relative_improvement: float
    single_line_cost: float
    two_piece_cost: float
    single_line_slope: float
    single_line_t0_sec: float
    two_piece_slope_near: float
    two_piece_slope_far: float
    two_piece_break_offset_m: float

    @property
    def edges(self) -> object:
        return self.base_model.edges

    @property
    def coef(self) -> object:
        return self.base_model.coef

    def predict(self, x_abs: torch.Tensor) -> torch.Tensor:
        return self.base_model.predict(x_abs)


@dataclass(frozen=True)
class _ModelSelectionFitStrategy:
    two_piece_strategy: TwoPieceRansacAutoBreakStrategy | TwoPieceIRLSAutoBreakStrategy
    fit_kind: str
    candidate_models: tuple[str, ...]
    prefer_two_piece_min_relative_improvement: float
    fallback_to_single_line: bool
    vmin_m_s: float
    vmax_m_s: float
    t0_lo_ms: float
    t0_hi_ms: float
    irls_huber_c: float
    irls_iters: int
    min_pts: int


_PhysicalFitStrategy = (
    TwoPieceRansacAutoBreakStrategy
    | TwoPieceIRLSAutoBreakStrategy
    | _ModelSelectionFitStrategy
)


@dataclass(frozen=True)
class _FitCacheEntry:
    model: object | None
    diagnostics: tuple[float, ...] | None
    fit_failed: bool
    diagnostics_computed: bool = False
    failure_reason: int | None = None


@dataclass(frozen=True)
class _FitTaskCfgValues:
    fit_kind: str
    n_iter: int
    inlier_th_ms: float
    irls_huber_c: float
    irls_iters: int
    min_pts: int
    n_break_cand: int
    q_lo: float
    q_hi: float
    seed: int
    slope_eps: float
    sort_offsets: bool
    candidate_models: tuple[str, ...]
    prefer_two_piece_min_relative_improvement: float
    fallback_to_single_line: bool
    vmin_m_s: float
    vmax_m_s: float
    t0_lo_ms: float
    t0_hi_ms: float
    min_offset_spread_m: float
    torch_num_threads_per_worker: int


@dataclass(frozen=True)
class _FitTask:
    fit_key: tuple[int, ...]
    x_obs: np.ndarray
    y_obs: np.ndarray
    w_obs: np.ndarray
    obs_count_before_sampling: int
    cfg_values: _FitTaskCfgValues


@dataclass(frozen=True)
class _FitTaskResult:
    fit_key: tuple[int, ...]
    trend_model: object | None
    diagnostics: tuple[float, ...] | None
    fit_failed: bool
    failure_reason: int | None
    elapsed_sec: float
    obs_count: int
    obs_count_before_sampling: int
    fit_attempted: bool


def _tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _is_irls_fit_kind(fit_kind: str) -> bool:
    return fit_kind in {'auto_irls', 'two_piece_irls_autobreak'}


def _candidate_models_for_fit_kind(
    fit_kind: str,
    candidate_models: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if candidate_models is not None:
        return tuple(candidate_models)
    if fit_kind == 'auto_irls':
        return (MODEL_TYPE_TWO_PIECE, MODEL_TYPE_SINGLE_LINE)
    return (MODEL_TYPE_TWO_PIECE,)


def _fit_min_obs_required(cfg: PhysicsLiteConfig) -> int:
    candidates = _candidate_models_for_fit_kind(
        str(cfg.physical_trend.fit_kind),
        cfg.physical_trend.candidate_models,
    )
    min_pts = _fit_min_pts(cfg)
    required: list[int] = []
    if MODEL_TYPE_TWO_PIECE in candidates:
        required.append(2 * min_pts)
    if MODEL_TYPE_SINGLE_LINE in candidates:
        required.append(min_pts)
    if not required:
        return 2 * min_pts
    return min(required)


def _fit_min_pts(cfg: PhysicsLiteConfig) -> int:
    if _is_irls_fit_kind(str(cfg.physical_trend.fit_kind)):
        return int(cfg.two_piece_irls.min_pts)
    return int(cfg.two_piece_ransac.min_pts)


def _fit_strategy(cfg: PhysicsLiteConfig) -> _PhysicalFitStrategy:
    fit_kind = str(cfg.physical_trend.fit_kind)
    if _is_irls_fit_kind(fit_kind):
        two_piece_strategy = TwoPieceIRLSAutoBreakStrategy(
            huber_c=float(cfg.two_piece_irls.huber_c),
            iters=int(cfg.two_piece_irls.iters),
            min_pts=int(cfg.two_piece_irls.min_pts),
            n_break_cand=int(cfg.two_piece_irls.n_break_cand),
            q_lo=float(cfg.two_piece_irls.q_lo),
            q_hi=float(cfg.two_piece_irls.q_hi),
            slope_eps=float(cfg.two_piece_irls.slope_eps),
            sort_offsets=bool(cfg.two_piece_irls.sort_offsets),
        )
    else:
        two_piece_strategy = TwoPieceRansacAutoBreakStrategy(
            n_iter=int(cfg.two_piece_ransac.n_iter),
            inlier_th_ms=float(cfg.two_piece_ransac.inlier_th_ms),
            min_pts=int(cfg.two_piece_ransac.min_pts),
            n_break_cand=int(cfg.two_piece_ransac.n_break_cand),
            q_lo=float(cfg.two_piece_ransac.q_lo),
            q_hi=float(cfg.two_piece_ransac.q_hi),
            seed=int(cfg.two_piece_ransac.seed),
            slope_eps=float(cfg.two_piece_ransac.slope_eps),
            sort_offsets=bool(cfg.two_piece_ransac.sort_offsets),
        )
    candidate_models = _candidate_models_for_fit_kind(
        fit_kind,
        cfg.physical_trend.candidate_models,
    )
    if candidate_models == (MODEL_TYPE_TWO_PIECE,):
        return two_piece_strategy
    model_selection = cfg.physical_trend.model_selection
    return _ModelSelectionFitStrategy(
        two_piece_strategy=two_piece_strategy,
        fit_kind=fit_kind,
        candidate_models=candidate_models,
        prefer_two_piece_min_relative_improvement=float(
            model_selection.prefer_two_piece_min_relative_improvement
        ),
        fallback_to_single_line=bool(model_selection.fallback_to_single_line),
        vmin_m_s=float(cfg.physical_prefilter.vmin_m_s),
        vmax_m_s=float(cfg.physical_prefilter.vmax_m_s),
        t0_lo_ms=float(cfg.physical_prefilter.t0_lo_ms),
        t0_hi_ms=float(cfg.physical_prefilter.t0_hi_ms),
        irls_huber_c=float(cfg.two_piece_irls.huber_c),
        irls_iters=int(cfg.two_piece_irls.iters),
        min_pts=_fit_min_pts(cfg),
    )


def _confidence_weights_for_obs(coarse_pmax_obs: np.ndarray) -> np.ndarray:
    w = np.asarray(coarse_pmax_obs, dtype=np.float32)
    if w.ndim != 1:
        w = w.reshape(-1)
    good = np.isfinite(w) & (w > np.float32(0.0))
    if not bool(np.any(good)):
        return np.ones_like(w, dtype=np.float32)
    fill = np.float32(np.median(w[good].astype(np.float64, copy=False)))
    out = np.where(good, w, fill).astype(np.float32, copy=False)
    return np.clip(out, np.float32(1.0e-6), None)


def _fit_line_irls_huber_np(
    x: np.ndarray,
    y: np.ndarray,
    w_base: np.ndarray,
    *,
    huber_c: float,
    iters: int,
) -> tuple[float, float] | None:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ww0 = np.asarray(w_base, dtype=np.float64)
    valid = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ww0) & (ww0 > 0.0)
    xx = xx[valid]
    yy = yy[valid]
    ww0 = ww0[valid]
    if int(xx.size) < 2:
        return None
    w = ww0.copy()
    slope = 0.0
    intercept = float(np.mean(yy))
    eps = 1.0e-12
    for _ in range(int(iters)):
        sw = float(np.sum(w))
        if sw <= eps:
            return None
        sx = float(np.sum(w * xx))
        sy = float(np.sum(w * yy))
        sxx = float(np.sum(w * xx * xx))
        sxy = float(np.sum(w * xx * yy))
        denom = sw * sxx - sx * sx
        if denom <= eps:
            slope = 0.0
            intercept = sy / sw
            break
        slope = (sw * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / sw
        residual = yy - (slope * xx + intercept)
        scale = max(float(1.4826 * np.median(np.abs(residual))), 1.0e-6)
        r = residual / (float(huber_c) * scale)
        abs_r = np.abs(r)
        huber_w = np.where(abs_r <= 1.0, 1.0, 1.0 / np.maximum(abs_r, eps))
        w = ww0 * np.minimum(huber_w, 10.0)
    return float(slope), float(intercept)


def _model_cost(
    trend_model: object | None,
    *,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
) -> float:
    if trend_model is None:
        return float('inf')
    try:
        pred = _tensor_to_numpy(
            trend_model.predict(torch.as_tensor(x_obs, dtype=torch.float32))
        ).astype(np.float64, copy=False)
    except (TypeError, ValueError, RuntimeError):
        return float('inf')
    residual = np.asarray(y_obs, dtype=np.float64) - pred
    residual = residual[np.isfinite(residual)]
    if int(residual.size) == 0:
        return float('inf')
    return float(np.mean(np.abs(residual)))


def _build_single_line_model(
    *,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    w_obs: np.ndarray,
    strategy: _ModelSelectionFitStrategy,
) -> PiecewiseLinearTrend | None:
    x = np.asarray(x_obs, dtype=np.float32)
    y = np.asarray(y_obs, dtype=np.float32)
    w = np.asarray(w_obs, dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if int(np.count_nonzero(valid)) < int(strategy.min_pts):
        return None
    fit = _fit_line_irls_huber_np(
        x[valid],
        y[valid],
        w[valid],
        huber_c=float(strategy.irls_huber_c),
        iters=int(strategy.irls_iters),
    )
    if fit is None:
        return None
    slope, intercept = fit
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return None
    slope_lo = 1.0 / float(strategy.vmax_m_s)
    slope_hi = 1.0 / float(strategy.vmin_m_s)
    if not (slope_lo <= float(slope) <= slope_hi):
        return None
    t0_ms = float(intercept) * 1000.0
    if not float(strategy.t0_lo_ms) <= t0_ms <= float(strategy.t0_hi_ms):
        return None
    finite_x = x[valid]
    xmin = float(np.min(finite_x))
    xmax = float(np.max(finite_x))
    edges = torch.tensor([xmin, xmax, xmax], dtype=torch.float32)
    coef = torch.tensor(
        [[float(slope), float(intercept)], [float(slope), float(intercept)]],
        dtype=torch.float32,
    )
    return PiecewiseLinearTrend(edges=edges, coef=coef)


def _wrap_selected_model(
    model: object,
    *,
    selected_model: str,
    relative_improvement: float = np.nan,
    single_line_cost: float = np.nan,
    two_piece_cost: float = np.nan,
    single_line_model: object | None = None,
    two_piece_model: object | None = None,
) -> _SelectedPhysicalTrend:
    single_coef = (
        _tensor_to_numpy(single_line_model.coef).astype(np.float32, copy=False)
        if single_line_model is not None
        else None
    )
    two_piece_coef = (
        _tensor_to_numpy(two_piece_model.coef).astype(np.float32, copy=False)
        if two_piece_model is not None
        else None
    )
    two_piece_edges = (
        _tensor_to_numpy(two_piece_model.edges).astype(np.float32, copy=False)
        if two_piece_model is not None
        else None
    )
    return _SelectedPhysicalTrend(
        base_model=model,
        model_type=selected_model,
        selected_model=selected_model,
        relative_improvement=float(relative_improvement),
        single_line_cost=float(single_line_cost),
        two_piece_cost=float(two_piece_cost),
        single_line_slope=(
            float(single_coef[0, 0]) if single_coef is not None else np.nan
        ),
        single_line_t0_sec=(
            float(single_coef[0, 1]) if single_coef is not None else np.nan
        ),
        two_piece_slope_near=(
            float(two_piece_coef[0, 0]) if two_piece_coef is not None else np.nan
        ),
        two_piece_slope_far=(
            float(two_piece_coef[1, 0]) if two_piece_coef is not None else np.nan
        ),
        two_piece_break_offset_m=(
            float(two_piece_edges[1]) if two_piece_edges is not None else np.nan
        ),
    )


def _fit_model_selection_strategy(
    strategy: _ModelSelectionFitStrategy,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    w_tensor: torch.Tensor,
) -> object | None:
    x_obs = _tensor_to_numpy(x_tensor).astype(np.float32, copy=False)
    y_obs = _tensor_to_numpy(y_tensor).astype(np.float32, copy=False)
    w_obs = _tensor_to_numpy(w_tensor).astype(np.float32, copy=False)
    two_piece_model = None
    single_line_model = None
    if MODEL_TYPE_TWO_PIECE in strategy.candidate_models:
        two_piece_model = _fit_strategy_model(
            strategy.two_piece_strategy,
            x_tensor,
            y_tensor,
            w_tensor,
        )
    if (
        MODEL_TYPE_SINGLE_LINE in strategy.candidate_models
        and bool(strategy.fallback_to_single_line)
    ):
        single_line_model = _build_single_line_model(
            x_obs=x_obs,
            y_obs=y_obs,
            w_obs=w_obs,
            strategy=strategy,
        )
    two_piece_cost = _model_cost(two_piece_model, x_obs=x_obs, y_obs=y_obs)
    single_line_cost = _model_cost(single_line_model, x_obs=x_obs, y_obs=y_obs)
    relative_improvement = (
        1.0 - two_piece_cost / single_line_cost
        if (
            two_piece_model is not None
            and single_line_model is not None
            and np.isfinite(two_piece_cost)
            and np.isfinite(single_line_cost)
            and single_line_cost > 0.0
        )
        else np.nan
    )
    if two_piece_model is not None and single_line_model is not None:
        if (
            np.isfinite(relative_improvement)
            and relative_improvement
            >= float(strategy.prefer_two_piece_min_relative_improvement)
        ):
            return _wrap_selected_model(
                two_piece_model,
                selected_model=MODEL_TYPE_TWO_PIECE,
                relative_improvement=relative_improvement,
                single_line_cost=single_line_cost,
                two_piece_cost=two_piece_cost,
                single_line_model=single_line_model,
                two_piece_model=two_piece_model,
            )
        return _wrap_selected_model(
            single_line_model,
            selected_model=MODEL_TYPE_SINGLE_LINE,
            relative_improvement=relative_improvement,
            single_line_cost=single_line_cost,
            two_piece_cost=two_piece_cost,
            single_line_model=single_line_model,
            two_piece_model=two_piece_model,
        )
    if two_piece_model is not None:
        return _wrap_selected_model(
            two_piece_model,
            selected_model=MODEL_TYPE_TWO_PIECE,
            single_line_cost=single_line_cost,
            two_piece_cost=two_piece_cost,
            single_line_model=single_line_model,
            two_piece_model=two_piece_model,
        )
    if single_line_model is not None:
        return _wrap_selected_model(
            single_line_model,
            selected_model=MODEL_TYPE_SINGLE_LINE,
            single_line_cost=single_line_cost,
            two_piece_cost=two_piece_cost,
            single_line_model=single_line_model,
            two_piece_model=two_piece_model,
        )
    return None


def _fit_strategy_model(
    strategy: _PhysicalFitStrategy,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    w_tensor: torch.Tensor,
) -> object | None:
    if isinstance(strategy, _ModelSelectionFitStrategy):
        return _fit_model_selection_strategy(strategy, x_tensor, y_tensor, w_tensor)
    if isinstance(strategy, TwoPieceIRLSAutoBreakStrategy):
        return strategy.fit(x_tensor, y_tensor, w_tensor)
    return strategy.fit(x_tensor, y_tensor)


def _median_time_position(local_positions: np.ndarray, y_obs: np.ndarray) -> int:
    positions = np.asarray(local_positions, dtype=np.int64)
    if positions.size == 0:
        msg = 'local_positions must be non-empty'
        raise ValueError(msg)
    y_values = np.asarray(y_obs, dtype=np.float32)[positions]
    finite = np.isfinite(y_values)
    if not np.any(finite):
        return int(positions[0])
    finite_positions = positions[finite]
    finite_y = y_values[finite]
    median_y = float(np.median(finite_y.astype(np.float64, copy=False)))
    return int(finite_positions[int(np.argmin(np.abs(finite_y - median_y)))])


def _stable_observation_seed(seed: int, values: np.ndarray, *, bin_id: int) -> int:
    acc = int(seed) & 0xFFFFFFFF
    for value in np.asarray(values, dtype=np.int64).tolist():
        acc = (acc * 1664525 + int(value) + 1013904223) & 0xFFFFFFFF
    return int((acc + int(bin_id) * 374761393) & 0xFFFFFFFF)


def _bin_representative_position(  # noqa: PLR0913
    *,
    local_positions: np.ndarray,
    obs_indices: np.ndarray,
    y_obs: np.ndarray,
    p_obs: np.ndarray | None,
    bin_pick: str,
    random_seed: int,
    bin_id: int,
) -> int:
    positions = np.asarray(local_positions, dtype=np.int64)
    if positions.size == 0:
        msg = 'local_positions must be non-empty'
        raise ValueError(msg)
    if bin_pick == 'median_time':
        return _median_time_position(positions, y_obs)
    if bin_pick == 'random':
        seed = _stable_observation_seed(
            random_seed,
            np.asarray(obs_indices, dtype=np.int64)[positions],
            bin_id=int(bin_id),
        )
        rng = np.random.default_rng(seed)
        return int(positions[int(rng.integers(0, int(positions.size)))])

    if p_obs is not None:
        p_values = np.asarray(p_obs, dtype=np.float32)[positions]
        finite = np.isfinite(p_values)
        if np.any(finite):
            finite_positions = positions[finite]
            finite_p = p_values[finite]
            return int(finite_positions[int(np.argmax(finite_p))])
    return _median_time_position(positions, y_obs)


def _evenly_spaced_positions(length: int, count: int) -> np.ndarray:
    n = int(length)
    k = int(count)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    raw = np.linspace(0.0, float(n - 1), num=k)
    used: set[int] = set()
    out: list[int] = []
    for value in raw.tolist():
        pos = int(np.rint(float(value)))
        if pos in used:
            for delta in range(1, n):
                left = pos - delta
                right = pos + delta
                if left >= 0 and left not in used:
                    pos = left
                    break
                if right < n and right not in used:
                    pos = right
                    break
        used.add(pos)
        out.append(pos)
    return np.asarray(sorted(out), dtype=np.int64)


def _limit_selected_positions(
    selected_count: int,
    *,
    max_count: int,
    preserve_edge_bins: bool,
) -> np.ndarray:
    n = int(selected_count)
    max_n = int(max_count)
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    if bool(preserve_edge_bins) and max_n >= 2 and n >= 2:
        interior_count = max_n - 2
        interior = _evenly_spaced_positions(n - 2, interior_count) + 1
        return np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                interior,
                np.asarray([n - 1], dtype=np.int64),
            ]
        )
    return _evenly_spaced_positions(n, max_n)


def _sample_observation_indices_for_fit(  # noqa: PLR0911, PLR0913
    *,
    obs_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    pick_t_sec: np.ndarray,
    coarse_pmax: np.ndarray | None,
    cfg: PhysicsLiteConfig,
    min_required_obs: int = 0,
) -> np.ndarray:
    sampling = cfg.physical_runtime.observation_sampling
    obs = np.asarray(obs_indices, dtype=np.int64)
    insufficient = np.zeros((0,), dtype=np.int64)
    if not bool(sampling.enabled):
        return obs
    max_obs = int(sampling.max_obs_per_fit)
    if int(obs.size) <= max_obs:
        return obs

    x_obs = np.asarray(offset_abs_m, dtype=np.float32)[obs]
    y_obs = np.asarray(pick_t_sec, dtype=np.float32)[obs]
    finite = np.isfinite(x_obs) & np.isfinite(y_obs)
    finite_positions = np.flatnonzero(finite).astype(np.int64, copy=False)
    if int(finite_positions.size) == 0:
        return obs

    finite_x = x_obs[finite_positions]
    x_min = float(np.min(finite_x))
    x_max = float(np.max(finite_x))
    if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or x_max <= x_min:
        return obs

    n_bins = min(int(sampling.n_offset_bins), int(finite_positions.size))
    edges = np.linspace(x_min, x_max, num=n_bins + 1, dtype=np.float64)
    bin_ids = np.searchsorted(edges, finite_x.astype(np.float64), side='right') - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1).astype(np.int64, copy=False)

    p_obs = (
        None
        if coarse_pmax is None
        else np.asarray(coarse_pmax, dtype=np.float32)[obs]
    )
    selected_positions: list[int] = []
    for bin_id in range(n_bins):
        in_bin = finite_positions[bin_ids == int(bin_id)]
        if int(in_bin.size) == 0:
            continue
        selected_positions.append(
            _bin_representative_position(
                local_positions=in_bin,
                obs_indices=obs,
                y_obs=y_obs,
                p_obs=p_obs,
                bin_pick=str(sampling.bin_pick),
                random_seed=int(cfg.two_piece_ransac.seed),
                bin_id=int(bin_id),
            )
        )

    selected = obs[np.asarray(selected_positions, dtype=np.int64)]
    min_after = max(
        int(sampling.min_obs_per_fit_after_sampling),
        int(min_required_obs),
    )
    if int(selected.size) < min_after:
        return insufficient
    if int(selected.size) > max_obs:
        keep = _limit_selected_positions(
            int(selected.size),
            max_count=max_obs,
            preserve_edge_bins=bool(sampling.preserve_edge_bins),
        )
        selected = selected[keep]
    if int(selected.size) < min_after:
        return insufficient
    return np.asarray(selected, dtype=np.int64)


def _model_diagnostics(
    trend_model: object,
    *,
    obs_offsets_m: np.ndarray,
    obs_times_sec: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    edges = _tensor_to_numpy(trend_model.edges).astype(np.float32, copy=False)
    coef = _tensor_to_numpy(trend_model.coef).astype(np.float32, copy=False)
    if edges.shape != (3,) or coef.shape != (2, 2):
        msg = 'trend model must expose edges (3,) and coef (2,2)'
        raise ValueError(msg)

    slope_near = float(coef[0, 0])
    slope_far = float(coef[1, 0])
    velocity_near = (
        1.0 / slope_near
        if np.isfinite(slope_near) and slope_near > 0.0
        else np.nan
    )
    velocity_far = (
        1.0 / slope_far
        if np.isfinite(slope_far) and slope_far > 0.0
        else np.nan
    )

    x_obs = torch.as_tensor(obs_offsets_m, dtype=torch.float32)
    pred = _tensor_to_numpy(trend_model.predict(x_obs)).astype(np.float64, copy=False)
    residual_sec = np.abs(np.asarray(obs_times_sec, dtype=np.float64) - pred)
    residual_sec = residual_sec[np.isfinite(residual_sec)]
    if residual_sec.size == 0:
        resid_p50 = np.nan
        resid_p90 = np.nan
        selected_cost = float('inf')
    else:
        residual_ms = residual_sec * 1000.0
        resid_p50 = float(np.percentile(residual_ms, 50.0))
        resid_p90 = float(np.percentile(residual_ms, 90.0))
        selected_cost = float(np.mean(residual_sec))

    model_type = str(getattr(trend_model, 'model_type', MODEL_TYPE_TWO_PIECE))
    selected_model = str(getattr(trend_model, 'selected_model', model_type))
    single_line_cost = float(
        getattr(
            trend_model,
            'single_line_cost',
            selected_cost if model_type == MODEL_TYPE_SINGLE_LINE else np.nan,
        )
    )
    two_piece_cost = float(
        getattr(
            trend_model,
            'two_piece_cost',
            selected_cost if model_type == MODEL_TYPE_TWO_PIECE else np.nan,
        )
    )
    return (
        float(edges[1]),
        slope_near,
        slope_far,
        velocity_near,
        velocity_far,
        resid_p50,
        resid_p90,
        float(_MODEL_CODE_BY_TYPE.get(model_type, 0)),
        float(_MODEL_CODE_BY_TYPE.get(selected_model, 0)),
        float(getattr(trend_model, 'relative_improvement', np.nan)),
        single_line_cost,
        two_piece_cost,
        float(
            getattr(
                trend_model,
                'single_line_slope',
                slope_near if model_type == MODEL_TYPE_SINGLE_LINE else np.nan,
            )
        ),
        float(
            getattr(
                trend_model,
                'single_line_t0_sec',
                float(coef[0, 1]) if model_type == MODEL_TYPE_SINGLE_LINE else np.nan,
            )
        ),
        float(getattr(trend_model, 'two_piece_slope_near', slope_near)),
        float(getattr(trend_model, 'two_piece_slope_far', slope_far)),
        float(getattr(trend_model, 'two_piece_break_offset_m', float(edges[1]))),
    )


def _fit_key_for_obs(
    obs_indices: np.ndarray,
    *,
    precomputed_key: tuple[int, ...] | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    after_sampling: bool = False,
    count_missing_precomputed: bool = True,
) -> tuple[int, ...]:
    if precomputed_key is not None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_precomputed_fit_key_used')
        return precomputed_key

    if runtime_diagnostics is not None:
        if bool(count_missing_precomputed):
            runtime_diagnostics.inc('n_fit_key_missing_precomputed')
        runtime_diagnostics.inc('n_fit_key_built_from_indices')
        if bool(after_sampling):
            runtime_diagnostics.inc('n_fit_key_built_after_sampling')
    return _indices_key(obs_indices)


def _fit_cache_key(plan: _ObservationPlan) -> tuple[int, ...]:
    return _fit_key_for_obs(plan.obs_indices, precomputed_key=plan.obs_key)


def _offset_spread_failure_reason(
    x_obs: np.ndarray,
    *,
    min_pts: int,
    min_offset_spread_m: float,
) -> int | None:
    finite_x = np.asarray(x_obs, dtype=np.float64)
    finite_x = finite_x[np.isfinite(finite_x)]
    if int(finite_x.size) < int(min_pts):
        return PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS
    if float(np.ptp(finite_x)) < float(min_offset_spread_m):
        return PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS
    return None


def _fit_task_cfg_values(cfg: PhysicsLiteConfig) -> _FitTaskCfgValues:
    executor = cfg.physical_runtime.fit_executor
    fit_kind = str(cfg.physical_trend.fit_kind)
    is_irls = _is_irls_fit_kind(fit_kind)
    model_selection = cfg.physical_trend.model_selection
    return _FitTaskCfgValues(
        fit_kind=fit_kind,
        n_iter=int(cfg.two_piece_ransac.n_iter),
        inlier_th_ms=float(cfg.two_piece_ransac.inlier_th_ms),
        irls_huber_c=float(cfg.two_piece_irls.huber_c),
        irls_iters=int(cfg.two_piece_irls.iters),
        min_pts=_fit_min_pts(cfg),
        n_break_cand=(
            int(cfg.two_piece_irls.n_break_cand)
            if is_irls
            else int(cfg.two_piece_ransac.n_break_cand)
        ),
        q_lo=(
            float(cfg.two_piece_irls.q_lo)
            if is_irls
            else float(cfg.two_piece_ransac.q_lo)
        ),
        q_hi=(
            float(cfg.two_piece_irls.q_hi)
            if is_irls
            else float(cfg.two_piece_ransac.q_hi)
        ),
        seed=int(cfg.two_piece_ransac.seed),
        slope_eps=(
            float(cfg.two_piece_irls.slope_eps)
            if is_irls
            else float(cfg.two_piece_ransac.slope_eps)
        ),
        sort_offsets=(
            bool(cfg.two_piece_irls.sort_offsets)
            if is_irls
            else bool(cfg.two_piece_ransac.sort_offsets)
        ),
        candidate_models=_candidate_models_for_fit_kind(
            fit_kind,
            cfg.physical_trend.candidate_models,
        ),
        prefer_two_piece_min_relative_improvement=float(
            model_selection.prefer_two_piece_min_relative_improvement
        ),
        fallback_to_single_line=bool(model_selection.fallback_to_single_line),
        vmin_m_s=float(cfg.physical_prefilter.vmin_m_s),
        vmax_m_s=float(cfg.physical_prefilter.vmax_m_s),
        t0_lo_ms=float(cfg.physical_prefilter.t0_lo_ms),
        t0_hi_ms=float(cfg.physical_prefilter.t0_hi_ms),
        min_offset_spread_m=float(cfg.physical_trend.min_offset_spread_m),
        torch_num_threads_per_worker=int(executor.torch_num_threads_per_worker),
    )


def _fit_task_from_work_item(
    work_item: object,
    *,
    cfg_values: _FitTaskCfgValues,
) -> _FitTask:
    return _FitTask(
        fit_key=work_item.fit_key,
        x_obs=np.asarray(work_item.x_obs, dtype=np.float32),
        y_obs=np.asarray(work_item.y_obs, dtype=np.float32),
        w_obs=np.asarray(work_item.w_obs, dtype=np.float32),
        obs_count_before_sampling=int(work_item.obs_count_before_sampling),
        cfg_values=cfg_values,
    )


def _strategy_from_fit_task_cfg(
    cfg_values: _FitTaskCfgValues,
) -> _PhysicalFitStrategy:
    if _is_irls_fit_kind(str(cfg_values.fit_kind)):
        two_piece_strategy = TwoPieceIRLSAutoBreakStrategy(
            huber_c=float(cfg_values.irls_huber_c),
            iters=int(cfg_values.irls_iters),
            min_pts=int(cfg_values.min_pts),
            n_break_cand=int(cfg_values.n_break_cand),
            q_lo=float(cfg_values.q_lo),
            q_hi=float(cfg_values.q_hi),
            slope_eps=float(cfg_values.slope_eps),
            sort_offsets=bool(cfg_values.sort_offsets),
        )
    else:
        two_piece_strategy = TwoPieceRansacAutoBreakStrategy(
            n_iter=int(cfg_values.n_iter),
            inlier_th_ms=float(cfg_values.inlier_th_ms),
            min_pts=int(cfg_values.min_pts),
            n_break_cand=int(cfg_values.n_break_cand),
            q_lo=float(cfg_values.q_lo),
            q_hi=float(cfg_values.q_hi),
            seed=int(cfg_values.seed),
            slope_eps=float(cfg_values.slope_eps),
            sort_offsets=bool(cfg_values.sort_offsets),
        )
    if tuple(cfg_values.candidate_models) == (MODEL_TYPE_TWO_PIECE,):
        return two_piece_strategy
    return _ModelSelectionFitStrategy(
        two_piece_strategy=two_piece_strategy,
        fit_kind=str(cfg_values.fit_kind),
        candidate_models=tuple(cfg_values.candidate_models),
        prefer_two_piece_min_relative_improvement=float(
            cfg_values.prefer_two_piece_min_relative_improvement
        ),
        fallback_to_single_line=bool(cfg_values.fallback_to_single_line),
        vmin_m_s=float(cfg_values.vmin_m_s),
        vmax_m_s=float(cfg_values.vmax_m_s),
        t0_lo_ms=float(cfg_values.t0_lo_ms),
        t0_hi_ms=float(cfg_values.t0_hi_ms),
        irls_huber_c=float(cfg_values.irls_huber_c),
        irls_iters=int(cfg_values.irls_iters),
        min_pts=int(cfg_values.min_pts),
    )


def _run_fit_task(task: _FitTask) -> _FitTaskResult:
    cfg_values = task.cfg_values
    x_obs = np.asarray(task.x_obs, dtype=np.float32)
    y_obs = np.asarray(task.y_obs, dtype=np.float32)
    w_obs = np.asarray(task.w_obs, dtype=np.float32)
    spread_failure_reason = _offset_spread_failure_reason(
        x_obs,
        min_pts=int(cfg_values.min_pts),
        min_offset_spread_m=float(cfg_values.min_offset_spread_m),
    )
    if spread_failure_reason is not None:
        return _FitTaskResult(
            fit_key=task.fit_key,
            trend_model=None,
            diagnostics=None,
            fit_failed=False,
            failure_reason=spread_failure_reason,
            elapsed_sec=0.0,
            obs_count=int(x_obs.size),
            obs_count_before_sampling=int(task.obs_count_before_sampling),
            fit_attempted=False,
        )

    strategy = _strategy_from_fit_task_cfg(cfg_values)
    start = time.perf_counter()
    trend_model = _fit_strategy_model(
        strategy,
        torch.as_tensor(x_obs, dtype=torch.float32),
        torch.as_tensor(y_obs, dtype=torch.float32),
        torch.as_tensor(w_obs, dtype=torch.float32),
    )
    elapsed = time.perf_counter() - start
    if trend_model is None:
        return _FitTaskResult(
            fit_key=task.fit_key,
            trend_model=None,
            diagnostics=None,
            fit_failed=True,
            failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
            elapsed_sec=elapsed,
            obs_count=int(x_obs.size),
            obs_count_before_sampling=int(task.obs_count_before_sampling),
            fit_attempted=True,
        )
    try:
        diagnostics = _model_diagnostics(
            trend_model,
            obs_offsets_m=x_obs,
            obs_times_sec=y_obs,
        )
    except (TypeError, ValueError, RuntimeError):
        diagnostics = None
    return _FitTaskResult(
        fit_key=task.fit_key,
        trend_model=trend_model,
        diagnostics=diagnostics,
        fit_failed=False,
        failure_reason=None,
        elapsed_sec=elapsed,
        obs_count=int(x_obs.size),
        obs_count_before_sampling=int(task.obs_count_before_sampling),
        fit_attempted=True,
    )


def _set_fit_worker_torch_num_threads(num_threads: int) -> None:
    torch.set_num_threads(int(num_threads))


def _fit_progress_fields(
    *,
    done: int,
    total: int,
    start_sec: float,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    force: bool = False,
) -> dict[str, object]:
    elapsed = max(0.0, time.perf_counter() - float(start_sec))
    rate = (float(done) / elapsed) if elapsed > 0.0 else 0.0
    remaining = max(0, int(total) - int(done))
    eta = (float(remaining) / rate) if rate > 0.0 else 0.0
    fields: dict[str, object] = {
        'done': int(done),
        'total': int(total),
        'elapsed': elapsed,
        'rate': rate,
        'eta': eta,
        'force': force,
    }
    if runtime_diagnostics is not None:
        fields.update(
            {
                'cache_hit': int(runtime_diagnostics.n_cache_hits),
                'cache_miss': int(runtime_diagnostics.n_cache_misses),
                'n_fit_calls': int(runtime_diagnostics.n_fit_calls),
                'fit_total_sec': float(runtime_diagnostics.ransac_fit_total_sec),
            }
        )
    return fields


def _run_fit_tasks_with_executor(  # noqa: PLR0913
    tasks: list[_FitTask],
    *,
    cfg: PhysicsLiteConfig,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    progress_start_done: int = 0,
    progress_total: int | None = None,
    progress_start_sec: float | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    progress_cache_miss_base: int = 0,
    progress_fit_calls_base: int = 0,
    progress_fit_total_sec_base: float = 0.0,
) -> dict[tuple[int, ...], _FitTaskResult]:
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(cfg.physical_runtime.progress)
    )
    context = dict(progress_context or {})
    total = len(tasks) if progress_total is None else int(progress_total)
    start_sec = (
        time.perf_counter()
        if progress_start_sec is None
        else progress_start_sec
    )
    fit_calls_done = 0
    fit_total_sec = float(progress_fit_total_sec_base)
    executor_cfg = cfg.physical_runtime.fit_executor
    if str(executor_cfg.backend) == 'thread':
        with ThreadPoolExecutor(max_workers=executor_cfg.max_workers) as executor:
            futures = [executor.submit(_run_fit_task, task) for task in tasks]
            out: dict[tuple[int, ...], _FitTaskResult] = {}
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                out[result.fit_key] = result
                if bool(result.fit_attempted):
                    fit_calls_done += 1
                    fit_total_sec += float(result.elapsed_sec)
                fields = _fit_progress_fields(
                    done=int(progress_start_done) + completed,
                    total=total,
                    start_sec=start_sec,
                    runtime_diagnostics=runtime_diagnostics,
                    force=int(progress_start_done) + completed >= total,
                )
                fields.update(
                    {
                        'cache_miss': int(progress_cache_miss_base) + completed,
                        'n_fit_calls': int(progress_fit_calls_base) + fit_calls_done,
                        'fit_total_sec': fit_total_sec,
                    }
                )
                reporter.emit(
                    'fit.progress',
                    **context,
                    **fields,
                )
            return out

    with ProcessPoolExecutor(
        max_workers=executor_cfg.max_workers,
        initializer=_set_fit_worker_torch_num_threads,
        initargs=(int(executor_cfg.torch_num_threads_per_worker),),
    ) as executor:
        futures = [executor.submit(_run_fit_task, task) for task in tasks]
        out: dict[tuple[int, ...], _FitTaskResult] = {}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            out[result.fit_key] = result
            if bool(result.fit_attempted):
                fit_calls_done += 1
                fit_total_sec += float(result.elapsed_sec)
            fields = _fit_progress_fields(
                done=int(progress_start_done) + completed,
                total=total,
                start_sec=start_sec,
                runtime_diagnostics=runtime_diagnostics,
                force=int(progress_start_done) + completed >= total,
            )
            fields.update(
                {
                    'cache_miss': int(progress_cache_miss_base) + completed,
                    'n_fit_calls': int(progress_fit_calls_base) + fit_calls_done,
                    'fit_total_sec': fit_total_sec,
                }
            )
            reporter.emit(
                'fit.progress',
                **context,
                **fields,
            )
        return out


def _fit_model_for_plan(  # noqa: PLR0913
    *,
    strategy: _PhysicalFitStrategy,
    plan: _ObservationPlan,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    w_obs: np.ndarray,
    min_pts: int,
    min_offset_spread_m: float,
    cache: dict[tuple[int, ...], _FitCacheEntry],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    obs_count_before_sampling: int | None = None,
) -> tuple[
    object | None,
    tuple[float, ...] | None,
    int | None,
]:
    spread_failure_reason = _offset_spread_failure_reason(
        x_obs,
        min_pts=int(min_pts),
        min_offset_spread_m=float(min_offset_spread_m),
    )
    if spread_failure_reason is not None:
        return None, None, spread_failure_reason

    cache_key = _fit_cache_key(plan)
    entry = cache.get(cache_key)
    if entry is None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.record_cache_miss()
        try:
            x_tensor = torch.as_tensor(x_obs, dtype=torch.float32)
            y_tensor = torch.as_tensor(y_obs, dtype=torch.float32)
            w_tensor = torch.as_tensor(w_obs, dtype=torch.float32)
            if runtime_diagnostics is None:
                trend_model = _fit_strategy_model(
                    strategy,
                    x_tensor,
                    y_tensor,
                    w_tensor,
                )
            else:
                with runtime_diagnostics.time_ransac_fit(
                    obs_count=int(np.asarray(x_obs).size),
                    obs_count_before=obs_count_before_sampling,
                ):
                    trend_model = _fit_strategy_model(
                        strategy,
                        x_tensor,
                        y_tensor,
                        w_tensor,
                    )
        except (TypeError, ValueError, RuntimeError):
            trend_model = None

        if trend_model is None:
            entry = _FitCacheEntry(
                model=None,
                diagnostics=None,
                fit_failed=True,
                failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
            )
        else:
            entry = _FitCacheEntry(
                model=trend_model,
                diagnostics=None,
                fit_failed=False,
            )
        cache[cache_key] = entry
    elif runtime_diagnostics is not None:
        runtime_diagnostics.record_cache_hit()

    if entry.failure_reason is not None:
        return None, None, entry.failure_reason
    if bool(entry.fit_failed):
        return None, None, PHYSICAL_MODEL_FAILURE_FIT_FAILED
    return entry.model, entry.diagnostics, None
