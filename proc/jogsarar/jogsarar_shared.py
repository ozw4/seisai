#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from seisai_pick.residual_statics import refine_firstbreak_residual_statics
from seisai_pick.score.confidence_from_residual_statics import (
    trace_confidence_from_residual_statics,
)
from seisai_pick.score.confidence_from_trend_resid import (
    trace_confidence_from_trend_resid_gaussian,
    trace_confidence_from_trend_resid_var,
)


def find_segy_files(
    root: Path | str,
    *,
    exts: Iterable[str] = ('.sgy', '.segy'),
    recursive: bool = True,
) -> list[Path]:
    """Find SEG-Y files under root.

    Args:
        root: directory path.
        exts: file extensions (case-insensitive). ".sgy", ".segy" etc.
        recursive: if True use rglob, else glob only in root.

    Returns:
        Sorted list of paths.

    """
    r = Path(root)
    if not r.exists() or not r.is_dir():
        msg = f'root must be an existing directory: {r}'
        raise FileNotFoundError(msg)

    exts_l = []
    for e in exts:
        ee = str(e).lower()
        if not ee.startswith('.'):
            ee = '.' + ee
        exts_l.append(ee)
    exts_t = tuple(exts_l)

    if bool(recursive):
        cand = r.rglob('*')
    else:
        cand = r.glob('*')

    out: list[Path] = []
    for p in cand:
        if p.is_file() and p.suffix.lower() in exts_t:
            out.append(p)

    out.sort()
    return out


def require_npz_key(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    context: str | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper of common.npz_io.require_npz_key."""
    from common.npz_io import require_npz_key as _require_npz_key

    return _require_npz_key(z, key, context=context)


def read_trace_field(
    src,
    field,
    *,
    dtype,
    name: str = 'trace_field',
) -> np.ndarray:
    """Backward-compatible wrapper of common.segy_io.read_trace_field."""
    from common.segy_io import read_trace_field as _read_trace_field

    return _read_trace_field(src, field, dtype=dtype, name=name)


def valid_pick_mask(
    picks: np.ndarray,
    *,
    n_samples: int | None = None,
    zero_is_invalid: bool = True,
) -> np.ndarray:
    """Return a boolean mask for valid picks.

    Rules:
    - finite
    - >0 if zero_is_invalid else >=0
    - < n_samples if provided
    """
    pk = np.asarray(picks)
    m = np.isfinite(pk)
    if bool(zero_is_invalid):
        m &= pk > 0
    else:
        m &= pk >= 0
    if n_samples is not None:
        ns = int(n_samples)
        if ns <= 0:
            msg = f'n_samples must be positive, got {n_samples}'
            raise ValueError(msg)
        m &= pk < ns
    return m


def build_pick_aligned_window(
    wave_hw: np.ndarray,
    picks: np.ndarray,
    pre: int,
    post: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Extract per-trace windows aligned on pick indices.

    Args:
      wave_hw: (H, W)
      picks: (H,) pick indices. Non-finite or <=0 picks are treated as invalid.
      pre/post: number of samples before/after the pick (post is exclusive).
      fill: fill value for out-of-range samples.

    """
    wave = np.asarray(wave_hw, dtype=np.float32)
    pk = np.asarray(picks)

    if wave.ndim != 2:
        msg = f'wave_hw must be 2D (H,W), got {wave.shape}'
        raise ValueError(msg)
    if pk.ndim != 1 or pk.shape[0] != wave.shape[0]:
        msg = f'picks must be 1D length H={wave.shape[0]}, got {pk.shape}'
        raise ValueError(msg)
    if int(pre) < 0 or int(post) <= 0:
        msg = f'pre must be >=0 and post must be >0, got pre={pre}, post={post}'
        raise ValueError(msg)

    h, w = wave.shape
    length = int(pre + post)
    out = np.full((h, length), np.float32(fill), dtype=np.float32)

    for i in range(h):
        p = float(pk[i])
        if (not np.isfinite(p)) or p <= 0.0:
            continue

        c = int(np.rint(p))
        src_l = c - int(pre)
        src_r = c + int(post)

        ov_l = max(0, src_l)
        ov_r = min(w, src_r)
        if ov_l >= ov_r:
            continue

        dst_l = ov_l - src_l
        dst_r = dst_l + (ov_r - ov_l)
        out[i, dst_l:dst_r] = wave[i, ov_l:ov_r]

    return out


def build_groups_by_key(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Group trace indices by a 1D integer key array.

    Returns:
      uniq: unique keys sorted ascending
      inv:  per-element group id (0..len(uniq)-1)
      groups: list of index arrays (each group is sorted in original order)

    """
    v = np.asarray(values)
    if v.ndim != 1:
        msg = f'values must be 1D, got {v.shape}'
        raise ValueError(msg)

    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='mergesort')
    splits = np.cumsum(counts)[:-1]
    groups = [np.asarray(g, dtype=np.int64) for g in np.split(order, splits)]
    return uniq.astype(v.dtype, copy=False), inv.astype(np.int64, copy=False), groups


def build_key_to_indices(values: np.ndarray) -> dict[int, np.ndarray]:
    """Return dict[key] -> indices for each unique key in values."""
    uniq, _, groups = build_groups_by_key(np.asarray(values))
    return {
        int(k): np.asarray(g, dtype=np.int64)
        for k, g in zip(uniq.tolist(), groups, strict=False)
    }


@dataclass(frozen=True)
class TilePerTraceStandardize:
    """Tile transform: per-trace standardization (same signature as seisai tile_transform)."""

    eps_std: float = 1e-8

    @torch.no_grad()
    def __call__(self, patch: torch.Tensor, *, return_meta: bool = False):
        from seisai_transforms.signal_ops.scaling.standardize import (
            standardize_per_trace_torch,
        )

        out = standardize_per_trace_torch(patch, eps=float(self.eps_std))
        return (out, {}) if return_meta else out


# =========================
# QC metrics
# =========================
def compute_conf_trend_gaussian_var(
    *,
    t_pick_sec: np.ndarray,
    t_trend_sec: np.ndarray,
    valid_mask: np.ndarray,
    sigma_ms: float,
    half_win_traces: int,
    sigma_std_ms: float,
    min_count: int,
    zero_invalid: bool,
) -> np.ndarray:
    t_pick = np.asarray(t_pick_sec, dtype=np.float32)
    t_trend = np.asarray(t_trend_sec, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)

    if t_pick.ndim != 1 or t_trend.ndim != 1 or valid.ndim != 1:
        msg = (
            't_pick_sec/t_trend_sec/valid_mask must be 1D, got '
            f'{t_pick.shape}, {t_trend.shape}, {valid.shape}'
        )
        raise ValueError(msg)
    if t_pick.shape != t_trend.shape or t_pick.shape != valid.shape:
        msg = (
            'shape mismatch: '
            f't_pick_sec={t_pick.shape}, t_trend_sec={t_trend.shape}, valid_mask={valid.shape}'
        )
        raise ValueError(msg)

    conf_gaussian = trace_confidence_from_trend_resid_gaussian(
        t_pick,
        t_trend,
        valid,
        sigma_ms=float(sigma_ms),
    )
    conf_var = trace_confidence_from_trend_resid_var(
        t_pick,
        t_trend,
        valid,
        half_win_traces=int(half_win_traces),
        sigma_std_ms=float(sigma_std_ms),
        min_count=int(min_count),
    )
    conf = (
        np.asarray(conf_gaussian, dtype=np.float32)
        * np.asarray(conf_var, dtype=np.float32)
    ).astype(np.float32, copy=False)
    if bool(zero_invalid):
        conf[~valid] = 0.0
    return conf


def compute_conf_trend1_from_trend(
    *,
    pick_final_i: np.ndarray,
    trend_center_i: np.ndarray,
    n_samples_in: int,
    dt_sec_in: float,
    sigma_ms: float,
    half_win_traces: int,
    sigma_std_ms: float,
    min_count: int,
) -> np.ndarray:
    p = np.asarray(pick_final_i, dtype=np.int64)
    c = np.asarray(trend_center_i, dtype=np.float32)
    if p.ndim != 1 or c.ndim != 1 or p.shape != c.shape:
        msg = f'pick_final_i/trend shape mismatch: {p.shape}, {c.shape}'
        raise ValueError(msg)

    t_pick_sec = p.astype(np.float32, copy=False) * float(dt_sec_in)
    t_trend_sec = c.astype(np.float32, copy=False) * float(dt_sec_in)
    trend_ok = np.isfinite(c) & (c > 0.0)
    valid = valid_pick_mask(p, n_samples=int(n_samples_in)) & trend_ok

    return compute_conf_trend_gaussian_var(
        t_pick_sec=t_pick_sec,
        t_trend_sec=t_trend_sec,
        valid_mask=valid,
        sigma_ms=float(sigma_ms),
        half_win_traces=int(half_win_traces),
        sigma_std_ms=float(sigma_std_ms),
        min_count=int(min_count),
        zero_invalid=True,
    )


@dataclass(frozen=True)
class ResidualStaticsMetrics:
    delta_pick: np.ndarray
    cmax: np.ndarray
    valid_mask: np.ndarray
    score: np.ndarray
    history: list[dict[str, Any]]


def compute_residual_statics_metrics(
    *,
    wave_hw: np.ndarray,
    picks: np.ndarray,
    pre: int,
    post: int,
    fill: float,
    max_lag: int,
    k_neighbors: int,
    n_iter: int,
    mode: str,
    c_th: float,
    smooth_method: str,
    lam: float,
    subsample: bool,
    propagate_low_corr: bool,
    taper: str,
    taper_power: float,
    lag_penalty: float,
    lag_penalty_power: float,
) -> ResidualStaticsMetrics:
    wave = np.asarray(wave_hw, dtype=np.float32)
    pk = np.asarray(picks)
    if wave.ndim != 2:
        msg = f'wave_hw must be 2D, got {wave.shape}'
        raise ValueError(msg)
    if pk.ndim != 1 or pk.shape[0] != wave.shape[0]:
        msg = f'picks must be 1D length H={wave.shape[0]}, got {pk.shape}'
        raise ValueError(msg)

    x_rs = build_pick_aligned_window(
        wave,
        picks=pk,
        pre=int(pre),
        post=int(post),
        fill=float(fill),
    )
    rs_res = refine_firstbreak_residual_statics(
        x_rs,
        max_lag=int(max_lag),
        k_neighbors=int(k_neighbors),
        n_iter=int(n_iter),
        mode=str(mode),
        c_th=float(c_th),
        smooth_method=str(smooth_method),
        lam=float(lam),
        subsample=bool(subsample),
        propagate_low_corr=bool(propagate_low_corr),
        taper=str(taper),
        taper_power=float(taper_power),
        lag_penalty=float(lag_penalty),
        lag_penalty_power=float(lag_penalty_power),
    )

    delta_pick = np.asarray(rs_res['delta_pick'], dtype=np.float32)
    cmax = np.asarray(rs_res['cmax'], dtype=np.float32)
    valid_mask = np.asarray(rs_res['valid_mask'], dtype=bool)
    score = np.asarray(rs_res['score'], dtype=np.float32)

    h = int(wave.shape[0])
    expected = (h,)
    if delta_pick.shape != expected:
        msg = f'delta_pick shape mismatch: {delta_pick.shape}, expected {expected}'
        raise ValueError(msg)
    if cmax.shape != expected:
        msg = f'cmax shape mismatch: {cmax.shape}, expected {expected}'
        raise ValueError(msg)
    if valid_mask.shape != expected:
        msg = f'valid_mask shape mismatch: {valid_mask.shape}, expected {expected}'
        raise ValueError(msg)
    if score.shape != expected:
        msg = f'score shape mismatch: {score.shape}, expected {expected}'
        raise ValueError(msg)

    history_obj = rs_res.get('history', [])
    if not isinstance(history_obj, list):
        msg = f'history must be list, got {type(history_obj).__name__}'
        raise TypeError(msg)
    history = list(history_obj)

    return ResidualStaticsMetrics(
        delta_pick=delta_pick.astype(np.float32, copy=False),
        cmax=cmax.astype(np.float32, copy=False),
        valid_mask=valid_mask.astype(bool, copy=False),
        score=score.astype(np.float32, copy=False),
        history=history,
    )


def compute_conf_rs_from_residual_statics(
    *,
    delta_pick: np.ndarray,
    cmax: np.ndarray,
    valid_mask: np.ndarray,
    c_th: float,
    max_lag: float,
) -> np.ndarray:
    delta = np.asarray(delta_pick, dtype=np.float32)
    corr = np.asarray(cmax, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if delta.ndim != 1 or corr.ndim != 1 or valid.ndim != 1:
        msg = (
            'delta_pick/cmax/valid_mask must be 1D, got '
            f'{delta.shape}, {corr.shape}, {valid.shape}'
        )
        raise ValueError(msg)
    if delta.shape != corr.shape or delta.shape != valid.shape:
        msg = (
            'shape mismatch: '
            f'delta_pick={delta.shape}, cmax={corr.shape}, valid_mask={valid.shape}'
        )
        raise ValueError(msg)
    return trace_confidence_from_residual_statics(
        delta,
        corr,
        valid,
        c_th=float(c_th),
        max_lag=float(max_lag),
    )
