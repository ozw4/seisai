from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path


def load_yaml_dict(config_path: Path) -> dict[str, object]:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.is_file():
        msg = f'--config not found: {cfg_path}'
        raise FileNotFoundError(msg)

    import yaml

    with cfg_path.open('r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        msg = f'config top-level must be dict, got {type(loaded).__name__}'
        raise TypeError(msg)
    return loaded


def parse_args_with_yaml_defaults(
    parser: argparse.ArgumentParser,
    *,
    load_defaults: Callable[[Path], dict[str, object]],
) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path, default=None)
    pre_args, _unknown = pre.parse_known_args()
    if pre_args.config is not None:
        parser.set_defaults(**load_defaults(pre_args.config))
    return parser.parse_args()


def coerce_path(key: str, value: object, *, allow_none: bool = False) -> Path | None:
    if value is None:
        if allow_none:
            return None
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    msg = f'config[{key}] must be str or Path, got {type(value).__name__}'
    raise TypeError(msg)


def coerce_optional_int(key: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f'config[{key}] must be int, got {type(value).__name__}'
        raise TypeError(msg)
    return int(value)


def coerce_optional_bool(key: str, value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        msg = f'config[{key}] must be bool, got {type(value).__name__}'
        raise TypeError(msg)
    return bool(value)


def coerce_optional_float(key: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f'config[{key}] must be float, got {type(value).__name__}'
        raise TypeError(msg)
    return float(value)


def normalize_segy_exts(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        vals = [x.strip() for x in value.split(',')]
        if len(vals) == 0:
            msg = 'config[segy_exts] must not be empty'
            raise ValueError(msg)

        out: list[str] = []
        for i, v in enumerate(vals):
            if v == '':
                msg = (
                    'config[segy_exts] contains empty value at '
                    f'position {i}: {value!r}'
                )
                raise ValueError(msg)
            e = v.lower()
            if not e.startswith('.'):
                e = '.' + e
            out.append(e)
        return tuple(out)

    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError('config[segy_exts] must not be empty')
        normalized_items: list[str] = []
        for i, raw in enumerate(value):
            if not isinstance(raw, str):
                msg = f'config[segy_exts][{i}] must be str, got {type(raw).__name__}'
                raise TypeError(msg)
            item = raw.strip()
            if item == '':
                msg = f'config[segy_exts][{i}] must not be empty'
                raise ValueError(msg)
            normalized_items.append(item)
        return normalize_segy_exts(','.join(normalized_items))

    msg = f'config[segy_exts] must be str or list[str], got {type(value).__name__}'
    raise TypeError(msg)


def build_yaml_defaults(
    loaded: dict[str, object],
    *,
    allowed_keys: set[str],
    coercers: dict[str, Callable[[object], object]],
) -> dict[str, object]:
    unknown = sorted(set(loaded) - allowed_keys)
    if unknown:
        msg = f'unknown config keys: {unknown}'
        raise ValueError(msg)

    defaults: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            msg = f'config key must be str, got {type(key).__name__}'
            raise TypeError(msg)
        coerce = coercers.get(key)
        if coerce is None:
            msg = f'missing coercer for config key: {key}'
            raise RuntimeError(msg)
        defaults[key] = coerce(value)
    return defaults
