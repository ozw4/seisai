from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

import yaml

if TYPE_CHECKING:
	from collections.abc import Callable

T = TypeVar('T')
R = TypeVar('R')


@overload
def require_value(
	d: dict,
	key: str,
	types: type[T],
	*,
	allow_none: bool = False,
	coerce: Callable[[T], R] | None = None,
	validator: Callable[[str, T], R] | None = None,
	missing_message: str | None = None,
	type_message: str | None = None,
) -> R | T | None: ...


@overload
def require_value(
	d: dict,
	key: str,
	types: tuple[type[Any], ...],
	*,
	allow_none: bool = False,
	coerce: Callable[[Any], Any] | None = None,
	validator: Callable[[str, Any], Any] | None = None,
	missing_message: str | None = None,
	type_message: str | None = None,
) -> Any: ...


def require_value(
	d: dict,
	key: str,
	types: type[Any] | tuple[type[Any], ...],
	*,
	allow_none: bool = False,
	coerce: Callable[[Any], Any] | None = None,
	validator: Callable[[str, Any], Any] | None = None,
	missing_message: str | None = None,
	type_message: str | None = None,
) -> Any:
	if key not in d:
		raise ValueError(missing_message or f'missing config key: {key}')
	return _validate_value(
		key,
		d[key],
		types,
		allow_none=allow_none,
		coerce=coerce,
		validator=validator,
		type_message=type_message,
	)


@overload
def optional_value(
	d: dict,
	key: str,
	default: R,
	types: type[T],
	*,
	allow_none: bool = False,
	coerce: Callable[[T], R] | None = None,
	validator: Callable[[str, T], R] | None = None,
	type_message: str | None = None,
	coerce_default: bool = False,
) -> R | None: ...


@overload
def optional_value(
	d: dict,
	key: str,
	default: Any,
	types: tuple[type[Any], ...],
	*,
	allow_none: bool = False,
	coerce: Callable[[Any], Any] | None = None,
	validator: Callable[[str, Any], Any] | None = None,
	type_message: str | None = None,
	coerce_default: bool = False,
) -> Any: ...


def optional_value(
	d: dict,
	key: str,
	default: Any,
	types: type[Any] | tuple[type[Any], ...],
	*,
	allow_none: bool = False,
	coerce: Callable[[Any], Any] | None = None,
	validator: Callable[[str, Any], Any] | None = None,
	type_message: str | None = None,
	coerce_default: bool = False,
) -> Any:
	if key not in d:
		return coerce(default) if coerce_default and coerce is not None else default
	return _validate_value(
		key,
		d[key],
		types,
		allow_none=allow_none,
		coerce=coerce,
		validator=validator,
		type_message=type_message,
	)


def _types_str(types: type[Any] | tuple[type[Any], ...]) -> str:
	if isinstance(types, tuple):
		return ' or '.join(t.__name__ for t in types)
	return types.__name__


def _validate_value(
	key: str,
	value: Any,
	types: type[Any] | tuple[type[Any], ...],
	*,
	allow_none: bool,
	coerce: Callable[[Any], Any] | None,
	validator: Callable[[str, Any], Any] | None,
	type_message: str | None,
) -> Any:
	if value is None and allow_none:
		return None

	if not isinstance(value, types):
		msg = (
			type_message
			or f'config.{key} must be {_types_str(types)}, got {type(value).__name__}'
		)
		raise TypeError(msg)

	if validator is not None:
		return validator(key, value)

	return coerce(value) if coerce is not None else value


def _validate_list_str(key: str, value: list[Any]) -> list[str]:
	if not all(isinstance(x, str) for x in value):
		raise TypeError(f'config.{key} must be list[str]')
	if len(value) == 0:
		raise ValueError(f'config.{key} must be non-empty')
	return cast('list[str]', value)


def _validate_tuple2_float(key: str, value: list[Any]) -> tuple[float, float]:
	if len(value) != 2:
		raise ValueError(f'config.{key} must be [float, float]')
	if not isinstance(value[0], (int, float)) or not isinstance(value[1], (int, float)):
		raise TypeError(f'config.{key} must be [float, float]')
	return (float(value[0]), float(value[1]))


def require_dict(d: dict, key: str) -> dict:
	return cast(
		'dict', require_value(d, key, dict, type_message=f'config.{key} must be dict')
	)


def require_list_str(d: dict, key: str) -> list[str]:
	return cast(
		'list[str]',
		require_value(
			d,
			key,
			list,
			type_message=f'config.{key} must be list[str]',
			validator=_validate_list_str,
		),
	)


def require_int(d: dict, key: str) -> int:
	return cast(
		'int', require_value(d, key, int, type_message=f'config.{key} must be int')
	)


def require_float(d: dict, key: str) -> float:
	return cast(
		'float',
		require_value(
			d,
			key,
			(int, float),
			type_message=f'config.{key} must be float',
			coerce=float,
		),
	)


def require_bool(d: dict, key: str) -> bool:
	return cast(
		'bool', require_value(d, key, bool, type_message=f'config.{key} must be bool')
	)


def optional_int(d: dict, key: str, default: int) -> int:
	return cast(
		'int',
		optional_value(
			d,
			key,
			default,
			int,
			type_message=f'config.{key} must be int',
			coerce=int,
			coerce_default=True,
		),
	)


def optional_str(d: dict, key: str, default: str) -> str:
	return cast(
		'str',
		optional_value(d, key, default, str, type_message=f'config.{key} must be str'),
	)


def optional_tuple2_float(
	d: dict,
	key: str,
	default: tuple[float, float],
) -> tuple[float, float]:
	return cast(
		'tuple[float, float]',
		optional_value(
			d,
			key,
			default,
			list,
			type_message=f'config.{key} must be [float, float]',
			validator=_validate_tuple2_float,
		),
	)


def optional_float(d: dict, key: str, default: float) -> float:
	return cast(
		'float',
		optional_value(
			d,
			key,
			default,
			(int, float),
			type_message=f'config.{key} must be float',
			coerce=float,
			coerce_default=True,
		),
	)


def optional_bool(d: dict, key: str, default: bool) -> bool:
	return cast(
		'bool',
		optional_value(
			d,
			key,
			default,
			bool,
			type_message=f'config.{key} must be bool',
			coerce=bool,
			coerce_default=True,
		),
	)


def load_config(path: str | Path) -> dict:
	p = Path(path)
	if not p.is_file():
		raise ValueError(f'config file not found: {p}')
	cfg = yaml.safe_load(p.read_text())
	if not isinstance(cfg, dict):
		raise TypeError('config root must be a dict')
	return cfg
