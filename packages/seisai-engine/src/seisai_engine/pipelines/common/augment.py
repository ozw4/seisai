from __future__ import annotations

from seisai_transforms import (
    FreqAugConfig,
    RandomFreqFilter,
    RandomHFlip,
    RandomPolarityFlip,
    RandomSpatialStretchSameH,
    RandomTimeStretch,
    SpaceAugConfig,
    TimeAugConfig,
)
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_tuple2_float,
    optional_value,
)

__all__ = ['build_train_augment_ops']


_FREQ_KINDS_ALLOWED = ('bandpass', 'lowpass', 'highpass')


def _require_dict_or_empty(value: object | None, key: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f'augment.{key} must be dict'
        raise TypeError(msg)
    return value


def _validate_tuple2_positive(name: str, v: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(v[0]), float(v[1])
    if lo <= 0.0 or hi <= 0.0:
        msg = f'{name} must be > 0'
        raise ValueError(msg)
    if lo > hi:
        msg = f'{name} must have min <= max'
        raise ValueError(msg)
    return (lo, hi)


def _validate_tuple2_01(name: str, v: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(v[0]), float(v[1])
    if lo < 0.0 or hi < 0.0 or lo > 1.0 or hi > 1.0:
        msg = f'{name} must be within [0, 1]'
        raise ValueError(msg)
    if lo > hi:
        msg = f'{name} must have min <= max'
        raise ValueError(msg)
    return (lo, hi)


def _validate_roll_01(name: str, v: float) -> float:
    vv = float(v)
    if vv < 0.0 or vv > 1.0:
        msg = f'{name} must be within [0, 1]'
        raise ValueError(msg)
    return vv


def _validate_prob_01(name: str, v: float) -> float:
    vv = float(v)
    if vv < 0.0 or vv > 1.0:
        msg = f'{name} must be within [0, 1]'
        raise ValueError(msg)
    return vv


def _validate_kinds(name: str, kinds: list[str]) -> tuple[str, ...]:
    if len(kinds) == 0:
        msg = f'{name} must be non-empty'
        raise ValueError(msg)
    for k in kinds:
        if k not in _FREQ_KINDS_ALLOWED:
            msg = f'{name} must be one of {list(_FREQ_KINDS_ALLOWED)}'
            raise ValueError(msg)
    return tuple(str(k) for k in kinds)


def build_train_augment_ops(augment_cfg: dict | None) -> tuple[list, list]:
    if augment_cfg is None:
        augment_cfg = {}
    if not isinstance(augment_cfg, dict):
        msg = 'augment must be dict'
        raise TypeError(msg)

    hflip_prob = _validate_prob_01(
        'augment.hflip_prob', optional_float(augment_cfg, 'hflip_prob', 0.0)
    )
    polarity_prob = _validate_prob_01(
        'augment.polarity_prob', optional_float(augment_cfg, 'polarity_prob', 0.0)
    )

    space_cfg = _require_dict_or_empty(augment_cfg.get('space'), 'space')
    space_prob = _validate_prob_01(
        'augment.space.prob', optional_float(space_cfg, 'prob', 0.0)
    )
    space_factor_range = optional_tuple2_float(space_cfg, 'factor_range', (0.90, 1.10))
    space_factor_range = _validate_tuple2_positive(
        'augment.space.factor_range', space_factor_range
    )

    time_cfg = _require_dict_or_empty(augment_cfg.get('time'), 'time')
    time_prob = _validate_prob_01(
        'augment.time.prob', optional_float(time_cfg, 'prob', 0.0)
    )
    time_factor_range = optional_tuple2_float(time_cfg, 'factor_range', (0.95, 1.05))
    time_factor_range = _validate_tuple2_positive(
        'augment.time.factor_range', time_factor_range
    )

    freq_cfg = _require_dict_or_empty(augment_cfg.get('freq'), 'freq')
    freq_prob = _validate_prob_01(
        'augment.freq.prob', optional_float(freq_cfg, 'prob', 0.0)
    )
    freq_kinds = optional_value(
        freq_cfg,
        'kinds',
        list(_FREQ_KINDS_ALLOWED),
        list,
        validator=lambda key, value: _validate_kinds('augment.freq.kinds', value),
        type_message='config.augment.freq.kinds must be list[str]',
    )
    freq_band = optional_tuple2_float(freq_cfg, 'band', (0.05, 0.45))
    freq_width = optional_tuple2_float(freq_cfg, 'width', (0.10, 0.35))
    freq_roll = optional_float(freq_cfg, 'roll', 0.02)
    freq_restandardize = optional_bool(freq_cfg, 'restandardize', default=False)

    freq_band = _validate_tuple2_01('augment.freq.band', freq_band)
    freq_width = _validate_tuple2_01('augment.freq.width', freq_width)
    freq_roll = _validate_roll_01('augment.freq.roll', freq_roll)

    geom_ops = [
        RandomHFlip(prob=float(hflip_prob)),
        RandomSpatialStretchSameH(
            SpaceAugConfig(prob=float(space_prob), factor_range=space_factor_range)
        ),
        RandomTimeStretch(
            TimeAugConfig(prob=float(time_prob), factor_range=time_factor_range)
        ),
    ]

    post_ops = [
        RandomFreqFilter(
            FreqAugConfig(
                prob=float(freq_prob),
                kinds=tuple(freq_kinds),
                band=freq_band,
                width=freq_width,
                roll=float(freq_roll),
                restandardize=bool(freq_restandardize),
            )
        ),
        RandomPolarityFlip(prob=float(polarity_prob)),
    ]

    return geom_ops, post_ops
