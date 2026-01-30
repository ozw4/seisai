from __future__ import annotations

from pathlib import Path

import yaml


def require_dict(d: dict, key: str) -> dict:
	if key not in d:
		raise ValueError(f'missing config key: {key}')
	v = d[key]
	if not isinstance(v, dict):
		raise TypeError(f'config.{key} must be dict, got {type(v).__name__}')
	return v


def require_list_str(d: dict, key: str) -> list[str]:
	if key not in d:
		raise ValueError(f'missing config key: {key}')
	v = d[key]
	if not isinstance(v, list):
		raise TypeError(f'config.{key} must be list[str], got {type(v).__name__}')
	if not all(isinstance(x, str) for x in v):
		raise TypeError(f'config.{key} must be list[str]')
	if len(v) == 0:
		raise ValueError(f'config.{key} must be non-empty')
	return v


def require_int(d: dict, key: str) -> int:
	if key not in d:
		raise ValueError(f'missing config key: {key}')
	v = d[key]
	if not isinstance(v, int):
		raise TypeError(f'config.{key} must be int, got {type(v).__name__}')
	return v


def require_float(d: dict, key: str) -> float:
	if key not in d:
		raise ValueError(f'missing config key: {key}')
	v = d[key]
	if not isinstance(v, (int, float)):
		raise TypeError(f'config.{key} must be float, got {type(v).__name__}')
	return float(v)


def require_bool(d: dict, key: str) -> bool:
	if key not in d:
		raise ValueError(f'missing config key: {key}')
	v = d[key]
	if not isinstance(v, bool):
		raise TypeError(f'config.{key} must be bool, got {type(v).__name__}')
	return v


def optional_int(d: dict, key: str, default: int) -> int:
	if key not in d:
		return int(default)
	v = d[key]
	if not isinstance(v, int):
		raise TypeError(f'config.{key} must be int, got {type(v).__name__}')
	return v


def optional_str(d: dict, key: str, default: str) -> str:
	if key not in d:
		return default
	v = d[key]
	if not isinstance(v, str):
		raise TypeError(f'config.{key} must be str, got {type(v).__name__}')
	return v


def optional_tuple2_float(
	d: dict, key: str, default: tuple[float, float]
) -> tuple[float, float]:
	if key not in d:
		return default
	v = d[key]
	if not isinstance(v, list):
		raise TypeError(f'config.{key} must be [float, float], got {type(v).__name__}')
	if len(v) != 2:
		raise ValueError(f'config.{key} must be [float, float]')
	if not isinstance(v[0], (int, float)) or not isinstance(v[1], (int, float)):
		raise TypeError(f'config.{key} must be [float, float]')
	return (float(v[0]), float(v[1]))


def optional_float(d: dict, key: str, default: float) -> float:
	if key not in d:
		return float(default)
	v = d[key]
	if not isinstance(v, (int, float)):
		raise TypeError(f'config.{key} must be float, got {type(v).__name__}')
	return float(v)


def optional_bool(d: dict, key: str, default: bool) -> bool:
	if key not in d:
		return bool(default)
	v = d[key]
	if not isinstance(v, bool):
		raise TypeError(f'config.{key} must be bool, got {type(v).__name__}')
	return v


def load_config(path: str | Path) -> dict:
	p = Path(path)
	if not p.is_file():
		raise ValueError(f'config file not found: {p}')
	cfg = yaml.safe_load(p.read_text())
	if not isinstance(cfg, dict):
		raise TypeError('config root must be a dict')
	return cfg
