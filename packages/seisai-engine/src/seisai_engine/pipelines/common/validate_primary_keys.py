from __future__ import annotations

from typing import Iterable

__all__ = ['validate_primary_keys']


def validate_primary_keys(primary_keys: object) -> tuple[str, ...]:
	if not isinstance(primary_keys, (list, tuple)):
		raise TypeError('dataset.primary_keys must be list[str] or tuple[str]')

	if len(primary_keys) == 0:
		raise ValueError('dataset.primary_keys must not be empty')

	normalized: list[str] = []
	seen: set[str] = set()

	for key in primary_keys:
		if not isinstance(key, str):
			raise TypeError('dataset.primary_keys elements must be str')
		stripped = key.strip()
		if not stripped:
			raise ValueError('dataset.primary_keys must not contain empty values')
		if stripped in seen:
			raise ValueError(f'dataset.primary_keys has duplicate: {stripped}')
		seen.add(stripped)
		normalized.append(stripped)

	return tuple(normalized)
