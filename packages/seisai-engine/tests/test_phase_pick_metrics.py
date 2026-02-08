import math

import torch
from seisai_engine.metrics.phase_pick_metrics import (
    compute_ps_metrics_from_batch,
    masked_abs_error_1d,
    pick_argmax_w,
    summarize_abs_error,
)


def test_pick_argmax_w_shape_and_dtype() -> None:
    B, H, W = 2, 3, 5
    p = torch.randn((B, H, W), dtype=torch.float32)
    idx = pick_argmax_w(p)
    assert isinstance(idx, torch.Tensor)
    assert tuple(idx.shape) == (B, H)
    assert idx.dtype is torch.int64


def test_masked_abs_error_1d_extracts_by_valid() -> None:
    pred = torch.tensor([[10, 20, 30], [1, 2, 3]], dtype=torch.int64)
    gt = torch.tensor([[11, 10, 30], [2, 2, 0]], dtype=torch.int64)
    valid = torch.tensor([[True, False, True], [True, True, False]], dtype=torch.bool)
    err = masked_abs_error_1d(pred, gt, valid)
    assert tuple(err.shape) == (4,)
    assert err.dtype is torch.int64
    assert torch.equal(err, torch.tensor([1, 0, 1, 0], dtype=torch.int64))


def test_summarize_abs_error_empty_is_nan() -> None:
    out = summarize_abs_error(
        torch.zeros((0,), dtype=torch.int64), thresholds=(5, 10, 20)
    )
    assert math.isnan(out['mean'])
    assert math.isnan(out['median'])
    assert math.isnan(out['p_le_5'])
    assert math.isnan(out['p_le_10'])
    assert math.isnan(out['p_le_20'])


def test_compute_ps_metrics_from_batch_smoke_and_empty_valid() -> None:
    B, C, H, W = 2, 3, 4, 6
    logits = torch.zeros((B, C, H, W), dtype=torch.float32)

    # Construct a clear argmax in time for P/S by shaping class logits per pixel.
    # For P: highest at w=2 on trace 0 only
    logits[:, 0] = -5.0
    logits[:, 1] = -5.0
    logits[:, 2] = 0.0  # noise baseline
    logits[:, 0, 0, 2] = 5.0
    # For S: highest at w=4 on trace 1 only
    logits[:, 1, 1, 4] = 5.0

    trace_valid = torch.ones((B, H), dtype=torch.bool)
    label_valid = torch.ones((B, H), dtype=torch.bool)

    p_gt = torch.full((B, H), -1, dtype=torch.int64)
    s_gt = torch.full((B, H), -1, dtype=torch.int64)
    p_gt[:, 0] = 2
    s_gt[:, 1] = 4

    batch = {
        'trace_valid': trace_valid,
        'label_valid': label_valid,
        'meta': {
            'p_idx_view': p_gt,
            's_idx_view': s_gt,
        },
    }

    m = compute_ps_metrics_from_batch(logits, batch, thresholds=(5, 10, 20))
    assert 'p_mean' in m
    assert 's_mean' in m
    assert m['p_mean'] == 0.0
    assert m['s_mean'] == 0.0

    # Empty valid (label_valid all False) should not crash and should return NaNs.
    batch_empty = {
        'trace_valid': trace_valid,
        'label_valid': torch.zeros((B, H), dtype=torch.bool),
        'meta': {
            'p_idx_view': p_gt,
            's_idx_view': s_gt,
        },
    }
    m2 = compute_ps_metrics_from_batch(logits, batch_empty, thresholds=(5, 10, 20))
    assert math.isnan(m2['p_mean'])
    assert math.isnan(m2['s_mean'])
