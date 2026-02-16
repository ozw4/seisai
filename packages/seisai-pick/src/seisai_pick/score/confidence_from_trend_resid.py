from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from seisai_utils.convert import to_bool_mask_torch, to_numpy, to_torch
from seisai_utils.validator import (
    require_all_numpy,
    require_boolint_array,
    require_float_array,
    require_same_shape_and_backend,
    validate_array,
)
from torch import Tensor


def _as_2d_bh(x: Tensor) -> tuple[Tensor, bool]:
    if x.ndim == 1:
        return x.unsqueeze(0), True
    return x, False


def _conv1d_same_bh(x_bh: Tensor, kernel_len: int) -> Tensor:
    if kernel_len <= 0:
        msg = f'kernel_len must be > 0, got {kernel_len}'
        raise ValueError(msg)
    if x_bh.ndim != 2:
        msg = f'x_bh must be 2D (B,H), got {tuple(x_bh.shape)}'
        raise ValueError(msg)

    pad = kernel_len // 2
    k = x_bh.new_ones((1, 1, int(kernel_len)))
    y = F.conv1d(x_bh.unsqueeze(1), k, padding=int(pad))
    return y.squeeze(1)


def _local_resid_var_bh(
    t_pick_bh: Tensor,
    t_trend_bh: Tensor,
    valid_bh: Tensor,
    *,
    half_win_traces: int,
    min_count: int,
) -> tuple[Tensor, Tensor]:
    if half_win_traces < 0:
        msg = f'half_win_traces must be >= 0, got {half_win_traces}'
        raise ValueError(msg)
    if min_count < 1:
        msg = f'min_count must be >= 1, got {min_count}'
        raise ValueError(msg)
    if t_pick_bh.ndim != 2 or t_trend_bh.ndim != 2 or valid_bh.ndim != 2:
        msg = 'inputs must be (B,H)'
        raise ValueError(msg)
    if t_pick_bh.shape != t_trend_bh.shape or t_pick_bh.shape != valid_bh.shape:
        msg = (
            f'shape mismatch: pick={tuple(t_pick_bh.shape)}, '
            f'trend={tuple(t_trend_bh.shape)}, valid={tuple(valid_bh.shape)}'
        )
        raise ValueError(msg)

    resid = t_pick_bh.to(dtype=torch.float32) - t_trend_bh.to(dtype=torch.float32)
    resid = torch.where(valid_bh, resid, torch.zeros_like(resid))

    w = valid_bh.to(dtype=torch.float32)
    L = int(2 * half_win_traces + 1)
    if L <= 0:
        msg = f'kernel length must be > 0, got {L}'
        raise ValueError(msg)

    sum_w = _conv1d_same_bh(w, L)
    sum_r = _conv1d_same_bh(resid * w, L)
    sum_r2 = _conv1d_same_bh((resid * resid) * w, L)

    den = sum_w.clamp_min(1e-6)
    mean = sum_r / den
    var = (sum_r2 / den) - (mean * mean)
    var = var.clamp_min(0.0)

    ok = sum_w >= float(min_count)
    return var, ok


@torch.no_grad()
def trace_confidence_from_trend_resid_gaussian(
    t_pick_sec: Tensor | np.ndarray,
    t_trend_sec: Tensor | np.ndarray,
    valid: Tensor | np.ndarray | None = None,
    *,
    sigma_ms: float = 6.0,
) -> Tensor | np.ndarray:
    """Per-trace confidence from residual to trend.

    conf = exp(-(resid/sigma)^2), resid = t_pick - t_trend.
    - valid is optional; if None, finite mask is used.
    - Returns NumPy if all inputs are NumPy, otherwise returns Torch.
    """
    if sigma_ms <= 0.0:
        msg = f'sigma_ms must be > 0, got {sigma_ms}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(t_pick_sec, t_trend_sec, valid)

    t_pick = to_torch(t_pick_sec)
    t_trend = to_torch(t_trend_sec, like=t_pick)

    validate_array(
        t_pick,
        allowed_ndims=(1, 2),
        name='t_pick_sec',
        backend='torch',
        shape_hint='(B,H)',
    )
    validate_array(
        t_trend,
        allowed_ndims=(1, 2),
        name='t_trend_sec',
        backend='torch',
        shape_hint='(B,H)',
    )
    require_float_array(t_pick, name='t_pick_sec', backend='torch')
    require_float_array(t_trend, name='t_trend_sec', backend='torch')

    t_pick, squeezed = _as_2d_bh(t_pick)
    t_trend, _ = _as_2d_bh(t_trend)
    require_same_shape_and_backend(
        t_pick, t_trend, name_a='t_pick_sec', name_b='t_trend_sec', backend='torch'
    )

    if valid is None:
        v = torch.isfinite(t_pick) & torch.isfinite(t_trend)
    else:
        t_valid = to_torch(valid, like=t_pick)
        t_valid = to_bool_mask_torch(valid, like=t_pick)
        validate_array(
            t_valid,
            allowed_ndims=(1, 2),
            name='valid',
            backend='torch',
            shape_hint='(B,H)',
        )
        require_boolint_array(t_valid, name='valid', backend='torch')
        t_valid, _ = _as_2d_bh(t_valid)
        require_same_shape_and_backend(
            t_pick, t_valid, name_a='t_pick_sec', name_b='valid', backend='torch'
        )
        v = t_valid.to(torch.bool)

    resid = t_pick.to(dtype=torch.float32) - t_trend.to(dtype=torch.float32)

    sigma_sec = float(sigma_ms) * 1e-3
    inv = 1.0 / float(sigma_sec)

    conf = torch.exp(-((resid * inv) ** 2))
    conf = torch.where(v, conf, torch.zeros_like(conf))

    if squeezed:
        conf = conf.squeeze(0)

    return to_numpy(conf) if all_numpy else conf


@torch.no_grad()
def trace_confidence_from_trend_resid_var(
    t_pick_sec: Tensor | np.ndarray,
    t_trend_sec: Tensor | np.ndarray,
    valid: Tensor | np.ndarray | None = None,
    *,
    half_win_traces: int = 8,
    sigma_std_ms: float = 6.0,
    min_count: int = 2,
) -> Tensor | np.ndarray:
    """Zigzag-penalty confidence using local variance of residuals.

    For each trace i, look at a trace-window [i-half_win, i+half_win].
    Let resid = (t_pick - t_trend). Compute variance over valid samples in the window.

    conf = exp(-var / sigma_std^2)

    Notes
    -----
    - valid is optional; if None, finite mask is used.
    - If the window has < min_count valid traces, confidence becomes 0.
    - Returns NumPy if all inputs are NumPy, otherwise returns Torch.

    """
    if half_win_traces < 0:
        msg = f'half_win_traces must be >= 0, got {half_win_traces}'
        raise ValueError(msg)
    if sigma_std_ms <= 0.0:
        msg = f'sigma_std_ms must be > 0, got {sigma_std_ms}'
        raise ValueError(msg)
    if min_count < 1:
        msg = f'min_count must be >= 1, got {min_count}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(t_pick_sec, t_trend_sec, valid)

    t_pick = to_torch(t_pick_sec)
    t_trend = to_torch(t_trend_sec, like=t_pick)

    validate_array(
        t_pick,
        allowed_ndims=(1, 2),
        name='t_pick_sec',
        backend='torch',
        shape_hint='(B,H)',
    )
    validate_array(
        t_trend,
        allowed_ndims=(1, 2),
        name='t_trend_sec',
        backend='torch',
        shape_hint='(B,H)',
    )
    require_float_array(t_pick, name='t_pick_sec', backend='torch')
    require_float_array(t_trend, name='t_trend_sec', backend='torch')

    t_pick, squeezed = _as_2d_bh(t_pick)
    t_trend, _ = _as_2d_bh(t_trend)
    require_same_shape_and_backend(
        t_pick, t_trend, name_a='t_pick_sec', name_b='t_trend_sec', backend='torch'
    )

    if valid is None:
        v = torch.isfinite(t_pick) & torch.isfinite(t_trend)
    else:
        t_valid = to_torch(valid, like=t_pick)
        t_valid = to_bool_mask_torch(valid, like=t_pick)
        validate_array(
            t_valid,
            allowed_ndims=(1, 2),
            name='valid',
            backend='torch',
            shape_hint='(B,H)',
        )
        require_boolint_array(t_valid, name='valid', backend='torch')
        t_valid, _ = _as_2d_bh(t_valid)
        require_same_shape_and_backend(
            t_pick, t_valid, name_a='t_pick_sec', name_b='valid', backend='torch'
        )
        v = t_valid.to(torch.bool)

    if int(2 * half_win_traces + 1) == 1:
        out = v.to(dtype=torch.float32)
        if squeezed:
            out = out.squeeze(0)
        return to_numpy(out) if all_numpy else out

    var, ok = _local_resid_var_bh(
        t_pick,
        t_trend,
        v,
        half_win_traces=int(half_win_traces),
        min_count=int(min_count),
    )

    sigma_sec = float(sigma_std_ms) * 1e-3
    inv_sigma2 = 1.0 / (sigma_sec * sigma_sec)
    conf = torch.exp(-(var * inv_sigma2))
    conf = torch.where(ok & v, conf, torch.zeros_like(conf))

    if squeezed:
        conf = conf.squeeze(0)

    return to_numpy(conf) if all_numpy else conf


@torch.no_grad()
def trace_trend_residual_variance(
    t_pick_sec: Tensor | np.ndarray,
    t_trend_sec: Tensor | np.ndarray,
    valid: Tensor | np.ndarray | None = None,
    *,
    half_win_traces: int = 8,
    min_count: int = 2,
    fill: float = float('nan'),
) -> Tensor | np.ndarray:
    """Local variance of residuals (t_pick - t_trend) within a trace-window.

    - The unit is [sec^2].
    - For traces with insufficient valid samples in the window (< min_count), output is `fill`.
    - Returns NumPy if all inputs are NumPy, otherwise returns Torch.
    """
    all_numpy = require_all_numpy(t_pick_sec, t_trend_sec, valid)

    t_pick = to_torch(t_pick_sec)
    t_trend = to_torch(t_trend_sec, like=t_pick)

    validate_array(
        t_pick,
        allowed_ndims=(1, 2),
        name='t_pick_sec',
        backend='torch',
        shape_hint='(B,H)',
    )
    validate_array(
        t_trend,
        allowed_ndims=(1, 2),
        name='t_trend_sec',
        backend='torch',
        shape_hint='(B,H)',
    )
    require_float_array(t_pick, name='t_pick_sec', backend='torch')
    require_float_array(t_trend, name='t_trend_sec', backend='torch')

    t_pick, squeezed = _as_2d_bh(t_pick)
    t_trend, _ = _as_2d_bh(t_trend)
    require_same_shape_and_backend(
        t_pick, t_trend, name_a='t_pick_sec', name_b='t_trend_sec', backend='torch'
    )

    if valid is None:
        v = torch.isfinite(t_pick) & torch.isfinite(t_trend)
    else:
        t_valid = to_torch(valid, like=t_pick)
        validate_array(
            t_valid,
            allowed_ndims=(1, 2),
            name='valid',
            backend='torch',
            shape_hint='(B,H)',
        )
        require_boolint_array(t_valid, name='valid', backend='torch')
        t_valid, _ = _as_2d_bh(t_valid)
        require_same_shape_and_backend(
            t_pick, t_valid, name_a='t_pick_sec', name_b='valid', backend='torch'
        )
        v = t_valid.to(torch.bool)

    if int(2 * half_win_traces + 1) == 1:
        var0 = torch.zeros_like(t_pick, dtype=torch.float32)
        out = torch.where(v, var0, var0.new_full(var0.shape, float(fill)))
    else:
        var, ok = _local_resid_var_bh(
            t_pick,
            t_trend,
            v,
            half_win_traces=int(half_win_traces),
            min_count=int(min_count),
        )
        out = torch.where(ok & v, var, var.new_full(var.shape, float(fill)))

    if squeezed:
        out = out.squeeze(0)

    return to_numpy(out) if all_numpy else out
