from __future__ import annotations

import math


def parse_trace_decimation_cfg(train_cfg: dict) -> tuple[float, tuple[int, int]]:
    raw = train_cfg.get('trace_decimation')
    if raw is None:
        return 0.0, (1, 1)
    if not isinstance(raw, dict):
        msg = 'train.trace_decimation must be dict'
        raise TypeError(msg)
    unknown = set(raw) - {'prob', 'stride_range'}
    if unknown:
        msg = (
            'train.trace_decimation has unsupported keys: '
            f'{sorted(unknown)}'
        )
        raise ValueError(msg)

    prob_raw = raw.get('prob', 0.0)
    if isinstance(prob_raw, bool) or not isinstance(prob_raw, (int, float)):
        msg = 'train.trace_decimation.prob must be float in [0, 1]'
        raise TypeError(msg)
    prob = float(prob_raw)
    if not math.isfinite(prob):
        msg = 'train.trace_decimation.prob must be finite'
        raise ValueError(msg)
    if prob < 0.0 or prob > 1.0:
        msg = 'train.trace_decimation.prob must be in [0, 1]'
        raise ValueError(msg)

    stride_range_raw = raw.get('stride_range', (1, 1))
    if not isinstance(stride_range_raw, (list, tuple)) or len(stride_range_raw) != 2:
        msg = 'train.trace_decimation.stride_range must be [min_int, max_int]'
        raise TypeError(msg)
    min_stride_raw, max_stride_raw = stride_range_raw
    if isinstance(min_stride_raw, bool) or not isinstance(min_stride_raw, int):
        msg = 'train.trace_decimation.stride_range[0] must be int'
        raise TypeError(msg)
    if isinstance(max_stride_raw, bool) or not isinstance(max_stride_raw, int):
        msg = 'train.trace_decimation.stride_range[1] must be int'
        raise TypeError(msg)
    min_stride = int(min_stride_raw)
    max_stride = int(max_stride_raw)
    if min_stride < 1:
        msg = 'train.trace_decimation.stride_range[0] must be >= 1'
        raise ValueError(msg)
    if min_stride > max_stride:
        msg = 'train.trace_decimation.stride_range requires min <= max'
        raise ValueError(msg)
    return prob, (min_stride, max_stride)
