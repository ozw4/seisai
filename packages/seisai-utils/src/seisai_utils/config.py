"""YAML configuration loading and typed access helpers.

This module provides:
- `load_config`: load a YAML file into a dict with basic validation.
- `require_*` helpers: fetch required config values with type checking/coercion.
- `optional_*` helpers: fetch optional config values with defaults and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from .config_yaml import load_yaml

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
    default: object,
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
    default: object,
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
    value: object,
    types: type[object] | tuple[type[object], ...],
    *,
    allow_none: bool,
    coerce: Callable[[object], object] | None,
    validator: Callable[[str, object], object] | None,
    type_message: str | None,
) -> object:
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
        msg = f'config.{key} must be list[str]'
        raise TypeError(msg)
    if len(value) == 0:
        msg = f'config.{key} must be non-empty'
        raise ValueError(msg)
    return cast('list[str]', value)


def _validate_tuple2_float(key: str, value: list[Any]) -> tuple[float, float]:
    if len(value) != 2:
        msg = f'config.{key} must be [float, float]'
        raise ValueError(msg)
    if not isinstance(value[0], (int, float)) or not isinstance(value[1], (int, float)):
        msg = f'config.{key} must be [float, float]'
        raise TypeError(msg)
    return (float(value[0]), float(value[1]))


def require_dict(d: dict, key: str) -> dict:
    """Return a required dict config value.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.

    Returns
    -------
    dict
            The validated dictionary value.

    Raises
    ------
    ValueError
            If the key is missing.
    TypeError
            If the value exists but is not a dict.

    """
    return cast(
        'dict', require_value(d, key, dict, type_message=f'config.{key} must be dict')
    )


def require_list_str(d: dict, key: str) -> list[str]:
    """Return a required list[str] config value.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.

    Returns
    -------
    list[str]
            The validated list of strings.

    Raises
    ------
    ValueError
            If the key is missing.
    TypeError
            If the value exists but is not a list of strings.

    """
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
    """Return a required integer config value.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.

    Returns
    -------
    int
            The validated integer value.

    Raises
    ------
    ValueError
            If the key is missing.
    TypeError
            If the value exists but is not an int.

    """
    return cast(
        'int', require_value(d, key, int, type_message=f'config.{key} must be int')
    )


def require_float(d: dict, key: str) -> float:
    """Return a required float config value.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.

    Returns
    -------
    float
            The validated (and coerced) float value.

    Raises
    ------
    ValueError
            If the key is missing.
    TypeError
            If the value exists but is not an int or float.

    """
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
    """Return a required boolean config value.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.

    Returns
    -------
    bool
            The validated boolean value.

    Raises
    ------
    ValueError
            If the key is missing.
    TypeError
            If the value exists but is not a bool.

    """
    return cast(
        'bool', require_value(d, key, bool, type_message=f'config.{key} must be bool')
    )


def optional_int(d: dict, key: str, default: int) -> int:
    """Return an integer config value if present, otherwise return the provided default.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.
    default : int
            Value to return if the key is not present.

    Returns
    -------
    int
            The validated (and coerced) integer value.

    Raises
    ------
    TypeError
            If the value exists but is not an int.

    """
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
    """Return a string config value if present, otherwise return the provided default.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.
    default : str
            Value to return if the key is not present.

    Returns
    -------
    str
            The validated string value.

    Raises
    ------
    TypeError
            If the value exists but is not a str.

    """
    return cast(
        'str',
        optional_value(d, key, default, str, type_message=f'config.{key} must be str'),
    )


def optional_tuple2_float(
    d: dict,
    key: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Return a (float, float) config value if present, otherwise return the provided default.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.
    default : tuple[float, float]
            Value to return if the key is not present.

    Returns
    -------
    tuple[float, float]
            The validated tuple of two floats.

    Raises
    ------
    TypeError
            If the value exists but is not a list of numbers.
    ValueError
            If the value exists but does not contain exactly two elements.

    """
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
    """Return a float config value if present, otherwise return the provided default.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.
    default : float
            Value to return if the key is not present.

    Returns
    -------
    float
            The validated (and coerced) float value.

    Raises
    ------
    TypeError
            If the value exists but is not an int or float.

    """
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


def optional_bool(d: dict, key: str, *, default: bool) -> bool:
    """Return a boolean config value if present, otherwise return the provided default.

    Parameters
    ----------
    d : dict
            Config dictionary to read from.
    key : str
            Key to look up in the config dictionary.
    default : bool
            Value to return if the key is not present.

    Returns
    -------
    bool
            The validated (and coerced) boolean value.

    Raises
    ------
    TypeError
            If the value exists but is not a bool.

    """
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
    """Load a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    path : str | Path
            Path to the YAML configuration file.

    Returns
    -------
    dict
            The parsed configuration dictionary.

    Raises
    ------
    ValueError
            If the file does not exist.
    TypeError
            If the top-level YAML object is not a dict.
    yaml.YAMLError
            If the YAML content cannot be parsed.

    """
    return load_yaml(path)
