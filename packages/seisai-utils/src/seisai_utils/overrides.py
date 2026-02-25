from __future__ import annotations

from typing import Any

import yaml

__all__ = [
    'deep_merge_dict',
    'parse_override_token',
    'set_nested_key',
]


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge_dict(out[key], value)
            continue
        if isinstance(value, dict):
            out[key] = deep_merge_dict({}, value)
            continue
        if isinstance(value, list):
            out[key] = list(value)
            continue
        out[key] = value
    return out


def parse_override_token(token: str) -> tuple[str, Any]:
    if not isinstance(token, str):
        msg = 'override token must be str'
        raise TypeError(msg)
    raw = token.strip()
    if not raw:
        msg = 'override token must be non-empty'
        raise ValueError(msg)
    if raw.startswith('--'):
        msg = f'unsupported option for this CLI: {raw}'
        raise ValueError(msg)
    if '=' not in raw:
        msg = f'override must be KEY=VALUE, got: {raw}'
        raise ValueError(msg)

    key, value_raw = raw.split('=', 1)
    key = key.strip()
    if not key:
        msg = f'override key is empty: {raw}'
        raise ValueError(msg)
    key_parts = key.split('.')
    if any((not part) for part in key_parts):
        msg = f'override key must be dot-separated non-empty segments: {key}'
        raise ValueError(msg)
    value = yaml.safe_load(value_raw)
    return key, value


def set_nested_key(cfg: dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split('.')
    cur: dict[str, Any] = cfg
    for part in parts[:-1]:
        existing = cur.get(part)
        if existing is None:
            nxt: dict[str, Any] = {}
            cur[part] = nxt
            cur = nxt
            continue
        if not isinstance(existing, dict):
            msg = f'override parent is not dict at "{part}" for key "{key_path}"'
            raise TypeError(msg)
        cur = existing
    cur[parts[-1]] = value
