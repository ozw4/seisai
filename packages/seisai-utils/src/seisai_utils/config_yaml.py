from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_TARGET_PATH_KEYS = frozenset(
    {
        'segy_files',
        'phase_pick_files',
        'infer_segy_files',
        'infer_phase_pick_files',
        'input_segy_files',
        'target_segy_files',
        'infer_input_segy_files',
        'infer_target_segy_files',
    }
)


def _load_yaml_doc(yaml_path: Path) -> dict:
    cfg = yaml.safe_load(yaml_path.read_text())
    if not isinstance(cfg, dict):
        msg = 'config root must be a dict'
        raise TypeError(msg)
    return cfg


def _resolve_path_str(base_dir: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve_target_value(*, key: str, value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        return _resolve_path_str(base_dir, value)
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            msg = f'config.paths.{key} must be list[str]'
            raise TypeError(msg)
        return [_resolve_path_str(base_dir, item) for item in value]
    msg = f'config.paths.{key} must be str or list[str]'
    raise TypeError(msg)


def _resolve_target_paths(node: Any, *, base_dir: Path) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _TARGET_PATH_KEYS:
                node[key] = _resolve_target_value(
                    key=key,
                    value=value,
                    base_dir=base_dir,
                )
                continue
            _resolve_target_paths(value, base_dir=base_dir)
        return
    if isinstance(node, list):
        for item in node:
            _resolve_target_paths(item, base_dir=base_dir)


def _normalize_base_list(base: Any) -> list[str]:
    if isinstance(base, str):
        if not base.strip():
            msg = 'config.base must be non-empty str or list[str]'
            raise ValueError(msg)
        return [base]
    if isinstance(base, list):
        if len(base) == 0:
            msg = 'config.base must be non-empty str or list[str]'
            raise ValueError(msg)
        if not all(isinstance(item, str) and item.strip() for item in base):
            msg = 'config.base must be non-empty str or list[str]'
            raise TypeError(msg)
        return list(base)
    msg = 'config.base must be str or list[str]'
    raise TypeError(msg)


def _resolve_base_yaml_path(*, yaml_path: Path, base: str) -> Path:
    base_path = Path(base).expanduser()
    if not base_path.is_absolute():
        base_path = yaml_path.parent / base_path
    base_path = base_path.resolve()
    if not base_path.is_file():
        msg = f'base config file not found: {base_path}'
        raise ValueError(msg)
    return base_path


def _merge_cfg(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_cfg(merged[key], value)
            continue
        merged[key] = value
    return merged


def _load_yaml_recursive(yaml_path: Path, *, stack: tuple[Path, ...]) -> dict:
    if yaml_path in stack:
        chain = ' -> '.join(str(p) for p in (*stack, yaml_path))
        msg = f'circular base reference: {chain}'
        raise ValueError(msg)

    cfg = _load_yaml_doc(yaml_path)
    _resolve_target_paths(cfg, base_dir=yaml_path.parent)

    base = cfg.pop('base', None)
    if base is None:
        return cfg

    merged_base: dict = {}
    for base_entry in _normalize_base_list(base):
        base_yaml_path = _resolve_base_yaml_path(yaml_path=yaml_path, base=base_entry)
        base_cfg = _load_yaml_recursive(base_yaml_path, stack=(*stack, yaml_path))
        merged_base = _merge_cfg(merged_base, base_cfg)

    return _merge_cfg(merged_base, cfg)


def load_yaml(path: str | Path) -> dict:
    yaml_path = Path(path).expanduser()
    if not yaml_path.is_file():
        msg = f'config file not found: {yaml_path}'
        raise ValueError(msg)
    yaml_path = yaml_path.resolve()

    return _load_yaml_recursive(yaml_path, stack=())
