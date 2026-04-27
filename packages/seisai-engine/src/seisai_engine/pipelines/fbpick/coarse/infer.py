from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seisai_engine.pipelines.common import load_cfg_with_base_dir, load_checkpoint

from .config import (
    COARSE_CKPT_PIPELINE,
    COARSE_IN_CHANS,
    COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE,
    COARSE_TIME_LEN,
    COARSE_TRACE_LEN,
    CoarseInferConfig,
    load_coarse_infer_config,
)

__all__ = [
    'load_infer_config',
    'load_validated_coarse_checkpoint',
    'run_coarse_infer',
    'validate_coarse_checkpoint_metadata',
]


def _invalid_checkpoint_message(
    *,
    key: str,
    expected: object,
    actual: object,
) -> str:
    msg = (
        'Invalid fbpick-coarse checkpoint: '
        f'expected {key}={expected!r}, got {actual!r}.'
    )
    if key == 'coarse_input_mode' and actual is None:
        msg += ' This checkpoint appears to be from the legacy tiled coarse pipeline.'
    return msg


def _require_ckpt_value(ckpt: Mapping[str, Any], key: str, expected: object) -> None:
    actual = ckpt.get(key)
    if actual != expected:
        raise ValueError(
            _invalid_checkpoint_message(key=key, expected=expected, actual=actual)
        )


def validate_coarse_checkpoint_metadata(ckpt: Mapping[str, Any]) -> None:
    if not isinstance(ckpt, Mapping):
        msg = 'checkpoint must be a mapping'
        raise TypeError(msg)

    _require_ckpt_value(ckpt, 'pipeline', COARSE_CKPT_PIPELINE)
    _require_ckpt_value(
        ckpt,
        'coarse_input_mode',
        COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE,
    )
    _require_ckpt_value(ckpt, 'coarse_trace_len', COARSE_TRACE_LEN)
    _require_ckpt_value(ckpt, 'coarse_time_len', COARSE_TIME_LEN)
    _require_ckpt_value(ckpt, 'coarse_in_chans', COARSE_IN_CHANS)


def load_validated_coarse_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt = load_checkpoint(path)
    validate_coarse_checkpoint_metadata(ckpt)
    return ckpt


def load_infer_config(config_path: str | Path) -> CoarseInferConfig:
    cfg, _ = load_cfg_with_base_dir(Path(config_path))
    return load_coarse_infer_config(cfg)


def run_coarse_infer(
    *,
    model: object,
    cfg: dict[str, Any],
    device: object,
) -> Path:
    _ = (model, device)
    load_coarse_infer_config(cfg)
    msg = (
        'fbpick coarse inference execution is not implemented in this branch; '
        'issue #18 defines the config and checkpoint metadata contract only'
    )
    raise NotImplementedError(msg)
