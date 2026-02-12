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


def load_yaml(path: str | Path) -> dict:
    yaml_path = Path(path).expanduser()
    if not yaml_path.is_file():
        msg = f'config file not found: {yaml_path}'
        raise ValueError(msg)
    yaml_path = yaml_path.resolve()

    cfg = yaml.safe_load(yaml_path.read_text())
    if not isinstance(cfg, dict):
        msg = 'config root must be a dict'
        raise TypeError(msg)

    _resolve_target_paths(cfg, base_dir=yaml_path.parent)
    return cfg
