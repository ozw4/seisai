from __future__ import annotations

__all__ = [
    'format_cfg_key',
    'normalize_endian',
    'raise_if_deprecated_time_len_keys',
]


def format_cfg_key(section: str, key: str) -> str:
    return f'{section}.{key}'


def normalize_endian(*, value: str, key_name: str) -> str:
    endian = str(value).strip().lower()
    if endian not in ('big', 'little'):
        msg = f'{key_name} must be "big" or "little"'
        raise ValueError(msg)
    return endian


def raise_if_deprecated_time_len_keys(
    *,
    train_cfg: object,
    transform_cfg: object,
) -> None:
    if isinstance(train_cfg, dict) and 'time_len' in train_cfg:
        msg = (
            f'deprecated key: {format_cfg_key("train", "time_len")}; '
            f'use {format_cfg_key("transform", "time_len")}'
        )
        raise ValueError(msg)
    if isinstance(transform_cfg, dict) and 'target_len' in transform_cfg:
        msg = (
            f'deprecated key: {format_cfg_key("transform", "target_len")}; '
            f'use {format_cfg_key("transform", "time_len")}'
        )
        raise ValueError(msg)
