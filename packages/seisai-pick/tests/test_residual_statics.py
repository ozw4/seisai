from __future__ import annotations

import numpy as np
import pytest
from seisai_pick.residual_statics import (
    apply_shift_linear,
    estimate_shift_ncc,
    refine_firstbreak_residual_statics,
    smooth_shifts,
)


def _ricker_wavelet(n_samples: int, center: float, f0: float) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32) - np.float32(center)
    a = (np.pi * np.float32(f0) * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w.astype(np.float32, copy=False)


def test_estimate_shift_ncc_sign_positive_is_later_pick() -> None:
    n = 256
    ref = _ricker_wavelet(n, center=96.0, f0=0.08)
    trace = apply_shift_linear(ref[None, :], np.array([3.0], dtype=np.float32))[0]

    lag, cmax = estimate_shift_ncc(
        trace, ref, max_lag=12, mode="diff", subsample=True
    )
    assert lag == pytest.approx(3.0, abs=0.25)
    assert cmax > 0.8


def test_refine_firstbreak_residual_statics_recovers_known_residuals() -> None:
    rng = np.random.default_rng(7)
    n_traces = 64
    n_samples = 280
    base = _ricker_wavelet(n_samples, center=110.0, f0=0.075)
    x = np.repeat(base[None, :], n_traces, axis=0)

    e_true = np.zeros(n_traces, dtype=np.float32)
    e_true[20] = 3.0
    e_true[21] = 2.5
    e_true[22] = 3.0
    e_true[40] = -2.0
    e_true[41] = -2.5

    x = apply_shift_linear(x, e_true)
    x += 0.03 * rng.standard_normal(size=x.shape).astype(np.float32, copy=False)

    x[5] = 0.0
    x[33] = 0.0

    res = refine_firstbreak_residual_statics(
        x,
        max_lag=10,
        k_neighbors=8,
        n_iter=5,
        mode="diff",
        c_th=0.25,
        smooth_method="wls",
        lam=4.0,
        subsample=True,
    )

    delta = res["delta_pick"]
    valid = res["valid_mask"]
    delta_trace = res["delta_trace"]

    assert delta.dtype == np.float32
    assert delta.shape == (n_traces,)
    assert valid.shape == (n_traces,)
    assert delta_trace.shape == (n_traces,)
    assert not bool(valid[5])
    assert not bool(valid[33])
    assert delta[5] == 0.0
    assert delta[33] == 0.0

    mask_eval = valid.copy()
    mask_eval[5] = False
    mask_eval[33] = False
    rmse = float(np.sqrt(np.mean((delta[mask_eval] - e_true[mask_eval]) ** 2)))

    assert rmse < 0.9
    assert delta[20] == pytest.approx(3.0, abs=0.8)
    assert delta[20] > 0.0
    assert np.allclose(delta_trace[valid], -delta[valid], atol=1e-4)


def test_smooth_shifts_wls_propagate_false_blocks_zero_weight() -> None:
    delta_raw = np.array([2.0, 0.0, 2.0], dtype=np.float32)
    weights = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    out_propagate = smooth_shifts(
        delta_raw, weights, method="wls", lam=5.0, propagate=True
    )
    out_block = smooth_shifts(
        delta_raw, weights, method="wls", lam=5.0, propagate=False
    )

    assert out_block[0] == pytest.approx(2.0, abs=1e-7)
    assert out_block[1] == pytest.approx(0.0, abs=1e-7)
    assert out_block[2] == pytest.approx(2.0, abs=1e-7)
    assert abs(float(out_propagate[1])) > 1e-3


def test_refine_propagate_low_corr_false_blocks_update_on_low_corr_trace() -> None:
    rng = np.random.default_rng(13)
    n_traces = 48
    n_samples = 260
    base = _ricker_wavelet(n_samples, center=100.0, f0=0.075)
    x = np.repeat(base[None, :], n_traces, axis=0)

    e_true = np.zeros(n_traces, dtype=np.float32)
    e_true[20] = 3.0
    e_true[22] = 3.0
    x = apply_shift_linear(x, e_true)
    x[21] = rng.standard_normal(n_samples).astype(np.float32, copy=False)

    c_th = 0.45
    common_kwargs = dict(
        max_lag=10,
        k_neighbors=6,
        n_iter=1,
        mode="diff",
        c_th=c_th,
        smooth_method="wls",
        lam=6.0,
        subsample=True,
    )

    res_block = refine_firstbreak_residual_statics(
        x, propagate_low_corr=False, **common_kwargs
    )
    res_propagate = refine_firstbreak_residual_statics(
        x, propagate_low_corr=True, **common_kwargs
    )

    delta_block = res_block["delta_pick"]
    delta_propagate = res_propagate["delta_pick"]

    assert float(res_block["cmax"][21]) < c_th
    assert abs(float(delta_block[21])) < 1e-7
    assert delta_block[20] > 0.5
    assert delta_block[22] > 0.5
    assert abs(float(delta_propagate[21])) > abs(float(delta_block[21])) + 1e-3


def test_estimate_shift_ncc_lag_penalty_prefers_smaller_abs_lag() -> None:
    n = 256
    ref = _ricker_wavelet(n, center=96.0, f0=0.08)
    tr_far = apply_shift_linear(ref[None, :], np.array([8.0], dtype=np.float32))[0]
    tr_near = apply_shift_linear(ref[None, :], np.array([2.0], dtype=np.float32))[0]
    trace = (tr_far + tr_near).astype(np.float32, copy=False)

    lag_no_penalty, _ = estimate_shift_ncc(
        trace,
        ref,
        max_lag=12,
        mode="raw",
        subsample=False,
        taper="hann",
        lag_penalty=0.0,
    )
    lag_penalty, _ = estimate_shift_ncc(
        trace,
        ref,
        max_lag=12,
        mode="raw",
        subsample=False,
        taper="hann",
        lag_penalty=0.25,
        lag_penalty_power=1.0,
    )

    assert abs(lag_penalty) < abs(lag_no_penalty)


def test_estimate_shift_ncc_taper_hann_accepts_and_returns_finite() -> None:
    n = 220
    ref = _ricker_wavelet(n, center=88.0, f0=0.09)
    trace = apply_shift_linear(ref[None, :], np.array([2.0], dtype=np.float32))[0]

    lag_hann, c_hann = estimate_shift_ncc(
        trace, ref, max_lag=10, mode="diff", taper="hann", taper_power=1.0
    )
    lag_none, c_none = estimate_shift_ncc(
        trace, ref, max_lag=10, mode="diff", taper=None, taper_power=1.0
    )
    assert np.isfinite(lag_hann)
    assert np.isfinite(c_hann)
    assert np.isfinite(lag_none)
    assert np.isfinite(c_none)

    with pytest.raises(ValueError):
        estimate_shift_ncc(trace, ref, max_lag=10, mode="diff", taper="bad_taper")
    with pytest.raises(ValueError):
        estimate_shift_ncc(trace, ref, max_lag=10, mode="diff", taper_power=0.0)


def test_refine_returns_score_array() -> None:
    n_traces = 24
    n_samples = 220
    base = _ricker_wavelet(n_samples, center=90.0, f0=0.08)
    x = np.repeat(base[None, :], n_traces, axis=0)
    x = apply_shift_linear(x, np.linspace(-1.5, 1.5, n_traces, dtype=np.float32))
    x[3] = 0.0

    res = refine_firstbreak_residual_statics(
        x,
        max_lag=8,
        k_neighbors=5,
        n_iter=2,
        mode="diff",
        c_th=0.25,
        smooth_method="wls",
        lam=4.0,
        subsample=True,
        lag_penalty=0.2,
        lag_penalty_power=1.0,
    )

    score = res["score"]
    valid = res["valid_mask"]

    assert score.dtype == np.float32
    assert score.shape == (n_traces,)
    assert np.all(np.isfinite(score))
    assert score[3] == 0.0
    assert not bool(valid[3])
