from __future__ import annotations

from typing import Any

from seisai_utils.config import optional_bool, optional_int, optional_str, require_value

__all__ = ['build_encdec2d_kwargs']


_DEFAULTS: dict[str, Any] = {
    'stage_strides': None,
    'extra_stages': 0,
    'extra_stage_strides': None,
    'extra_stage_channels': None,
    'extra_stage_use_bn': True,
    'pre_stages': 0,
    'pre_stage_strides': None,
    'pre_stage_kernels': None,
    'pre_stage_channels': None,
    'pre_stage_use_bn': True,
    'pre_stage_antialias': False,
    'pre_stage_aa_taps': 3,
    'pre_stage_aa_pad_mode': 'zeros',
    'decoder_channels': (256, 128, 64, 32),
    'decoder_scales': (2, 2, 2, 2),
    'upsample_mode': 'bilinear',
    'attention_type': 'scse',
    'intermediate_conv': True,
    'pretrained': False,
}


def _config_key(key: str) -> str:
    return f'config.model.{key}'


def _normalize_stride_list(
    key: str, value: list[Any] | tuple[Any, ...]
) -> list[tuple[int, int]]:
    if not isinstance(value, (list, tuple)):
        msg = f'{_config_key(key)} must be list[[int, int]]'
        raise TypeError(msg)
    out: list[tuple[int, int]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (list, tuple)):
            msg = f'{_config_key(key)}[{idx}] must be [int, int]'
            raise TypeError(msg)
        if len(item) != 2:
            msg = f'{_config_key(key)}[{idx}] must be [int, int]'
            raise ValueError(msg)
        a, b = item
        if not isinstance(a, int) or not isinstance(b, int):
            msg = f'{_config_key(key)}[{idx}] must be [int, int]'
            raise TypeError(msg)
        out.append((int(a), int(b)))
    return out


def _normalize_int_seq(
    key: str,
    value: list[Any] | tuple[Any, ...],
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        msg = f'{_config_key(key)} must be list[int]'
        raise TypeError(msg)
    if not allow_empty and len(value) == 0:
        msg = f'{_config_key(key)} must be non-empty'
        raise ValueError(msg)
    out: list[int] = []
    for idx, item in enumerate(value):
        if not isinstance(item, int):
            msg = f'{_config_key(key)}[{idx}] must be int'
            raise TypeError(msg)
        out.append(int(item))
    return tuple(out)


def _normalize_optional_int_seq(
    key: str, value: Any, *, allow_empty: bool
) -> tuple[int, ...] | None:
    if value is None:
        return None
    return _normalize_int_seq(key, value, allow_empty=allow_empty)


def _normalize_optional_stride_list(
    key: str, value: Any
) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    return _normalize_stride_list(key, value)


def _optional_value(
    model_cfg: dict,
    key: str,
    default: Any,
    types: type[Any] | tuple[type[Any], ...],
    *,
    allow_none: bool = False,
    validator: Any = None,
    type_message: str | None = None,
) -> Any:
    if key not in model_cfg:
        return default
    return require_value(
        model_cfg,
        key,
        types,
        allow_none=allow_none,
        validator=validator,
        type_message=type_message,
    )


def _default_value(defaults: dict | None, key: str, fallback: Any) -> Any:
    if defaults is None:
        return fallback
    return defaults.get(key, fallback)


def build_encdec2d_kwargs(
    model_cfg: dict,
    *,
    in_chans: int,
    out_chans: int,
    defaults: dict | None = None,
) -> dict:
    if not isinstance(model_cfg, dict):
        raise TypeError('model_cfg must be dict')

    backbone = require_value(
        model_cfg,
        'backbone',
        str,
        type_message='config.model.backbone must be str',
    )

    pretrained_default = _default_value(defaults, 'pretrained', _DEFAULTS['pretrained'])
    pretrained = optional_bool(model_cfg, 'pretrained', default=bool(pretrained_default))

    stage_strides_default = _normalize_optional_stride_list(
        'stage_strides',
        _default_value(defaults, 'stage_strides', _DEFAULTS['stage_strides']),
    )
    stage_strides = _optional_value(
        model_cfg,
        'stage_strides',
        stage_strides_default,
        (list, tuple),
        allow_none=True,
        validator=_normalize_stride_list,
        type_message='config.model.stage_strides must be list[[int, int]] or null',
    )

    extra_stages_default = _default_value(
        defaults, 'extra_stages', _DEFAULTS['extra_stages']
    )
    extra_stages = optional_int(model_cfg, 'extra_stages', int(extra_stages_default))

    extra_stage_strides_default = _normalize_optional_stride_list(
        'extra_stage_strides',
        _default_value(
            defaults, 'extra_stage_strides', _DEFAULTS['extra_stage_strides']
        ),
    )
    extra_stage_strides = _optional_value(
        model_cfg,
        'extra_stage_strides',
        extra_stage_strides_default,
        (list, tuple),
        allow_none=True,
        validator=_normalize_stride_list,
        type_message='config.model.extra_stage_strides must be list[[int, int]] or null',
    )

    extra_stage_channels_default = _normalize_optional_int_seq(
        'extra_stage_channels',
        _default_value(
            defaults, 'extra_stage_channels', _DEFAULTS['extra_stage_channels']
        ),
        allow_empty=True,
    )
    extra_stage_channels = _optional_value(
        model_cfg,
        'extra_stage_channels',
        extra_stage_channels_default,
        (list, tuple),
        allow_none=True,
        validator=lambda k, v: _normalize_int_seq(k, v, allow_empty=True),
        type_message='config.model.extra_stage_channels must be list[int] or null',
    )

    extra_stage_use_bn_default = _default_value(
        defaults, 'extra_stage_use_bn', _DEFAULTS['extra_stage_use_bn']
    )
    extra_stage_use_bn = optional_bool(
        model_cfg, 'extra_stage_use_bn', default=bool(extra_stage_use_bn_default)
    )

    pre_stages_default = _default_value(defaults, 'pre_stages', _DEFAULTS['pre_stages'])
    pre_stages = optional_int(model_cfg, 'pre_stages', int(pre_stages_default))

    pre_stage_strides_default = _normalize_optional_stride_list(
        'pre_stage_strides',
        _default_value(defaults, 'pre_stage_strides', _DEFAULTS['pre_stage_strides']),
    )
    pre_stage_strides = _optional_value(
        model_cfg,
        'pre_stage_strides',
        pre_stage_strides_default,
        (list, tuple),
        allow_none=True,
        validator=_normalize_stride_list,
        type_message='config.model.pre_stage_strides must be list[[int, int]] or null',
    )

    pre_stage_kernels_default = _normalize_optional_int_seq(
        'pre_stage_kernels',
        _default_value(defaults, 'pre_stage_kernels', _DEFAULTS['pre_stage_kernels']),
        allow_empty=True,
    )
    pre_stage_kernels = _optional_value(
        model_cfg,
        'pre_stage_kernels',
        pre_stage_kernels_default,
        (list, tuple),
        allow_none=True,
        validator=lambda k, v: _normalize_int_seq(k, v, allow_empty=True),
        type_message='config.model.pre_stage_kernels must be list[int] or null',
    )

    pre_stage_channels_default = _normalize_optional_int_seq(
        'pre_stage_channels',
        _default_value(
            defaults, 'pre_stage_channels', _DEFAULTS['pre_stage_channels']
        ),
        allow_empty=True,
    )
    pre_stage_channels = _optional_value(
        model_cfg,
        'pre_stage_channels',
        pre_stage_channels_default,
        (list, tuple),
        allow_none=True,
        validator=lambda k, v: _normalize_int_seq(k, v, allow_empty=True),
        type_message='config.model.pre_stage_channels must be list[int] or null',
    )

    pre_stage_use_bn_default = _default_value(
        defaults, 'pre_stage_use_bn', _DEFAULTS['pre_stage_use_bn']
    )
    pre_stage_use_bn = optional_bool(
        model_cfg, 'pre_stage_use_bn', default=bool(pre_stage_use_bn_default)
    )

    pre_stage_antialias_default = _default_value(
        defaults, 'pre_stage_antialias', _DEFAULTS['pre_stage_antialias']
    )
    pre_stage_antialias = optional_bool(
        model_cfg, 'pre_stage_antialias', default=bool(pre_stage_antialias_default)
    )

    pre_stage_aa_taps_default = _default_value(
        defaults, 'pre_stage_aa_taps', _DEFAULTS['pre_stage_aa_taps']
    )
    pre_stage_aa_taps = optional_int(
        model_cfg, 'pre_stage_aa_taps', int(pre_stage_aa_taps_default)
    )
    if int(pre_stage_aa_taps) not in (3, 5):
        raise ValueError('config.model.pre_stage_aa_taps must be 3 or 5')

    pre_stage_aa_pad_mode_default = _default_value(
        defaults, 'pre_stage_aa_pad_mode', _DEFAULTS['pre_stage_aa_pad_mode']
    )
    pre_stage_aa_pad_mode = optional_str(
        model_cfg, 'pre_stage_aa_pad_mode', str(pre_stage_aa_pad_mode_default)
    )
    if str(pre_stage_aa_pad_mode) != 'zeros':
        raise ValueError('config.model.pre_stage_aa_pad_mode must be "zeros"')

    decoder_channels_default = _normalize_int_seq(
        'decoder_channels',
        _default_value(defaults, 'decoder_channels', _DEFAULTS['decoder_channels']),
        allow_empty=False,
    )
    decoder_channels = _optional_value(
        model_cfg,
        'decoder_channels',
        decoder_channels_default,
        (list, tuple),
        allow_none=False,
        validator=lambda k, v: _normalize_int_seq(k, v, allow_empty=False),
        type_message='config.model.decoder_channels must be list[int]',
    )

    decoder_scales_default = _normalize_int_seq(
        'decoder_scales',
        _default_value(defaults, 'decoder_scales', _DEFAULTS['decoder_scales']),
        allow_empty=False,
    )
    decoder_scales = _optional_value(
        model_cfg,
        'decoder_scales',
        decoder_scales_default,
        (list, tuple),
        allow_none=False,
        validator=lambda k, v: _normalize_int_seq(k, v, allow_empty=False),
        type_message='config.model.decoder_scales must be list[int]',
    )

    upsample_mode_default = _default_value(
        defaults, 'upsample_mode', _DEFAULTS['upsample_mode']
    )
    upsample_mode = optional_str(model_cfg, 'upsample_mode', str(upsample_mode_default))

    attention_default = _default_value(
        defaults, 'attention_type', _DEFAULTS['attention_type']
    )
    attention_type = _optional_value(
        model_cfg,
        'attention_type',
        attention_default,
        str,
        allow_none=True,
        type_message='config.model.attention_type must be str or null',
    )

    intermediate_conv_default = _default_value(
        defaults, 'intermediate_conv', _DEFAULTS['intermediate_conv']
    )
    intermediate_conv = optional_bool(
        model_cfg, 'intermediate_conv', default=bool(intermediate_conv_default)
    )

    return {
        'backbone': str(backbone),
        'in_chans': int(in_chans),
        'out_chans': int(out_chans),
        'pretrained': bool(pretrained),
        'stage_strides': stage_strides,
        'extra_stages': int(extra_stages),
        'extra_stage_strides': extra_stage_strides,
        'extra_stage_channels': extra_stage_channels,
        'extra_stage_use_bn': bool(extra_stage_use_bn),
        'pre_stages': int(pre_stages),
        'pre_stage_strides': pre_stage_strides,
        'pre_stage_kernels': pre_stage_kernels,
        'pre_stage_channels': pre_stage_channels,
        'pre_stage_use_bn': bool(pre_stage_use_bn),
        'pre_stage_antialias': bool(pre_stage_antialias),
        'pre_stage_aa_taps': int(pre_stage_aa_taps),
        'pre_stage_aa_pad_mode': str(pre_stage_aa_pad_mode),
        'decoder_channels': decoder_channels,
        'decoder_scales': decoder_scales,
        'upsample_mode': str(upsample_mode),
        'attention_type': attention_type,
        'intermediate_conv': bool(intermediate_conv),
    }
