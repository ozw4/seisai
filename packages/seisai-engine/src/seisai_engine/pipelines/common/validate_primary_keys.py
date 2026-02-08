from __future__ import annotations

__all__ = ['validate_primary_keys']


def validate_primary_keys(primary_keys: object) -> tuple[str, ...]:
    if not isinstance(primary_keys, (list, tuple)):
        msg = 'dataset.primary_keys must be list[str] or tuple[str]'
        raise TypeError(msg)

    if len(primary_keys) == 0:
        msg = 'dataset.primary_keys must not be empty'
        raise ValueError(msg)

    normalized: list[str] = []
    seen: set[str] = set()

    for key in primary_keys:
        if not isinstance(key, str):
            msg = 'dataset.primary_keys elements must be str'
            raise TypeError(msg)
        stripped = key.strip()
        if not stripped:
            msg = 'dataset.primary_keys must not contain empty values'
            raise ValueError(msg)
        if stripped in seen:
            raise ValueError(f'dataset.primary_keys has duplicate: {stripped}')
        seen.add(stripped)
        normalized.append(stripped)

    return tuple(normalized)
