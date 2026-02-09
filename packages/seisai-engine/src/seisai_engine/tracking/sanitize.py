from __future__ import annotations

import re

__all__ = ['sanitize_key']

_ALLOWED_RE = re.compile(r'[A-Za-z0-9_./-]')


def sanitize_key(key: str) -> str:
    if not isinstance(key, str):
        msg = 'key must be str'
        raise TypeError(msg)
    if not key:
        msg = 'key must be non-empty str'
        raise ValueError(msg)

    sanitized = ''.join(ch if _ALLOWED_RE.match(ch) else '_' for ch in key)

    if not sanitized:
        msg = 'sanitized key is empty'
        raise ValueError(msg)

    if not any(ch.isalnum() or ch == '_' for ch in sanitized):
        msg = 'sanitized key is invalid'
        raise ValueError(msg)

    return sanitized
