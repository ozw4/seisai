from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from seisai_utils.config import optional_str, require_dict
from seisai_utils.overrides import deep_merge_dict, parse_override_token, set_nested_key

from seisai_engine.pipelines.common import resolve_relpath

__all__ = [
    'apply_unknown_overrides',
    'build_merged_cfg',
    'build_merged_cfg_with_ckpt_cfg',
    'cfg_hash',
    'get_allow_unsafe_override',
    'is_strict_int',
    'load_ckpt_cfg_for_merge',
    'merge_with_precedence',
    'resolve_ckpt_path',
    'resolve_segy_files',
    'select_state_dict',
    'sig_hash',
]


def is_strict_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def cfg_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def sig_hash(sig: dict[str, Any]) -> str:
    payload = json.dumps(sig, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def get_allow_unsafe_override(
    cfg: dict[str, Any],
    key_path: str = 'infer.allow_unsafe_override',
) -> bool:
    parts = key_path.split('.')
    if len(parts) == 0 or any((not part) for part in parts):
        msg = f'allow_unsafe key_path must be dot-separated non-empty segments: {key_path}'
        raise ValueError(msg)

    cur: dict[str, Any] = cfg
    for i, part in enumerate(parts[:-1]):
        obj = cur.get(part)
        if obj is None:
            return False
        if not isinstance(obj, dict):
            parent = '.'.join(parts[: i + 1])
            msg = f'{parent} must be dict'
            raise TypeError(msg)
        cur = obj

    leaf = parts[-1]
    value = cur.get(leaf, False)
    if not isinstance(value, bool):
        msg = f'{key_path} must be bool'
        raise TypeError(msg)
    return bool(value)


def apply_unknown_overrides(
    cfg: dict[str, Any],
    unknown_overrides: list[str],
    safe_paths: frozenset[str],
    allow_unsafe_key_path: str = 'infer.allow_unsafe_override',
) -> dict[str, Any]:
    out = deep_merge_dict({}, cfg)
    allow_unsafe = get_allow_unsafe_override(out, key_path=allow_unsafe_key_path)
    for token in unknown_overrides:
        key, value = parse_override_token(token)
        if (key not in safe_paths) and (not allow_unsafe):
            msg = (
                'unsafe override key is not allowed by default: '
                f'{key}. Set {allow_unsafe_key_path}=true to allow.'
            )
            raise ValueError(msg)
        set_nested_key(out, key, value)
        allow_unsafe = get_allow_unsafe_override(out, key_path=allow_unsafe_key_path)
    return out


def resolve_ckpt_path(
    cfg: dict[str, Any],
    base_dir: Path,
    infer_key: str = 'infer',
    ckpt_key: str = 'ckpt_path',
) -> Path:
    infer_cfg = require_dict(cfg, infer_key)
    ckpt_rel = optional_str(infer_cfg, ckpt_key, '')
    ckpt_rel = ckpt_rel.strip()
    if not ckpt_rel:
        msg = f'{infer_key}.{ckpt_key} must be non-empty'
        raise ValueError(msg)
    ckpt_path = Path(resolve_relpath(base_dir, ckpt_rel))
    if not ckpt_path.is_file():
        msg = f'checkpoint not found: {ckpt_path}'
        raise FileNotFoundError(msg)
    return ckpt_path


def resolve_segy_files(base_dir: Path, segy_files: list[str]) -> list[str]:
    resolved: list[str] = []
    for idx, item in enumerate(segy_files):
        if not isinstance(item, str):
            msg = f'paths.segy_files[{idx}] must be str'
            raise TypeError(msg)
        resolved.append(resolve_relpath(base_dir, item))
    return resolved


def select_state_dict(ckpt: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    use_ema = bool(ckpt.get('infer_used_ema', False))
    if use_ema:
        if 'ema_state_dict' not in ckpt:
            msg = 'checkpoint infer_used_ema=true but ema_state_dict is missing'
            raise KeyError(msg)
        state_dict = ckpt['ema_state_dict']
    else:
        state_dict = ckpt['model_state_dict']
    if not isinstance(state_dict, dict):
        msg = 'checkpoint state_dict must be dict'
        raise TypeError(msg)
    return state_dict, use_ema


def merge_with_precedence(
    default_cfg: dict[str, Any],
    ckpt_cfg: dict[str, Any],
    infer_cfg: dict[str, Any],
) -> dict[str, Any]:
    merged = deep_merge_dict(default_cfg, ckpt_cfg)
    merged = deep_merge_dict(merged, infer_cfg)
    return merged


def build_merged_cfg(
    infer_yaml_cfg: dict[str, Any],
    base_dir: Path,
    unknown_overrides: list[str],
    default_cfg: dict[str, Any],
    safe_paths: frozenset[str],
    *,
    ckpt_cfg_loader: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    pre_cfg = deep_merge_dict(default_cfg, infer_yaml_cfg)
    pre_cfg = apply_unknown_overrides(
        cfg=pre_cfg,
        unknown_overrides=unknown_overrides,
        safe_paths=safe_paths,
    )
    ckpt_cfg = _call_ckpt_cfg_loader(
        ckpt_cfg_loader=ckpt_cfg_loader,
        infer_cfg_for_ckpt=pre_cfg,
        base_dir=base_dir,
    )
    merged = merge_with_precedence(default_cfg, ckpt_cfg, infer_yaml_cfg)
    merged = apply_unknown_overrides(
        cfg=merged,
        unknown_overrides=unknown_overrides,
        safe_paths=safe_paths,
    )
    return merged


def _coerce_ckpt_cfg_to_dict(cfg_obj: Any) -> dict[str, Any] | None:
    if isinstance(cfg_obj, dict):
        return cfg_obj
    if cfg_obj is None:
        return None
    if is_dataclass(cfg_obj):
        asdict_obj = asdict(cfg_obj)
        return asdict_obj if isinstance(asdict_obj, dict) else None
    try:
        out = dict(cfg_obj)
    except (TypeError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def _call_ckpt_cfg_loader(
    *,
    ckpt_cfg_loader: Callable[..., dict[str, Any]],
    infer_cfg_for_ckpt: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    try:
        params = inspect.signature(ckpt_cfg_loader).parameters
    except (TypeError, ValueError):
        return ckpt_cfg_loader(infer_cfg_for_ckpt, base_dir)
    if 'infer_cfg_for_ckpt' in params and 'base_dir' in params:
        return ckpt_cfg_loader(
            infer_cfg_for_ckpt=infer_cfg_for_ckpt,
            base_dir=base_dir,
        )
    return ckpt_cfg_loader(infer_cfg_for_ckpt, base_dir)


def load_ckpt_cfg_for_merge(
    *,
    infer_cfg_for_ckpt: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    ckpt_path = resolve_ckpt_path(infer_cfg_for_ckpt, base_dir=base_dir)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if not isinstance(ckpt, dict):
        msg = 'checkpoint must be dict'
        raise TypeError(msg)
    cfg_from_ckpt = _coerce_ckpt_cfg_to_dict(ckpt.get('cfg'))
    if cfg_from_ckpt is None:
        msg = 'checkpoint must contain dict cfg'
        raise TypeError(msg)
    return cfg_from_ckpt


def build_merged_cfg_with_ckpt_cfg(
    *,
    infer_yaml_cfg: dict[str, Any],
    base_dir: Path,
    unknown_overrides: list[str],
    default_cfg: dict[str, Any],
    safe_paths: set[str] | frozenset[str],
) -> dict[str, Any]:
    return build_merged_cfg(
        infer_yaml_cfg=infer_yaml_cfg,
        base_dir=base_dir,
        unknown_overrides=unknown_overrides,
        default_cfg=default_cfg,
        safe_paths=frozenset(safe_paths),
        ckpt_cfg_loader=load_ckpt_cfg_for_merge,
    )
