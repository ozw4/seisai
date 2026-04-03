from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from seisai_engine.pipelines.common import load_checkpoint

__all__ = [
    'build_fine_init_state_dict',
    'load_fine_init_from_coarse_checkpoint',
    'validate_coarse_checkpoint_for_fine',
]


def _require_state_dict(name: str, value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        msg = f'{name} must be dict'
        raise TypeError(msg)
    return value


def _find_first_conv_key(state_dict: dict[str, torch.Tensor]) -> str:
    for key, value in state_dict.items():
        if key.startswith('seg_head.'):
            continue
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            return key
    msg = 'unable to identify first conv weight from state_dict'
    raise RuntimeError(msg)


def _find_first_bn_prefix(state_dict: dict[str, torch.Tensor]) -> str:
    suffix = '.running_mean'
    for key, value in state_dict.items():
        if key.startswith('seg_head.'):
            continue
        if key.endswith(suffix) and isinstance(value, torch.Tensor):
            return key[: -len(suffix)]
    msg = 'unable to identify first batch-norm stats from state_dict'
    raise RuntimeError(msg)


def _normalize_output_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(v) for v in value)
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    msg = 'checkpoint output_ids must be list[str] or tuple[str, ...]'
    raise TypeError(msg)


def validate_coarse_checkpoint_for_fine(
    ckpt: dict[str, Any],
    *,
    fine_model_sig: dict[str, Any] | None = None,
) -> None:
    if not isinstance(ckpt, dict):
        msg = 'ckpt must be dict'
        raise TypeError(msg)

    pipeline = ckpt.get('pipeline')
    if pipeline != 'fbpick':
        msg = f'coarse init checkpoint pipeline must be "fbpick", got {pipeline!r}'
        raise ValueError(msg)

    output_ids = _normalize_output_ids(ckpt.get('output_ids'))
    if output_ids != ('P',):
        msg = f'coarse init checkpoint output_ids must be ["P"], got {output_ids!r}'
        raise ValueError(msg)

    softmax_axis = ckpt.get('softmax_axis')
    if softmax_axis != 'time':
        msg = (
            'coarse init checkpoint softmax_axis must be "time", '
            f'got {softmax_axis!r}'
        )
        raise ValueError(msg)

    model_sig = ckpt.get('model_sig')
    if not isinstance(model_sig, dict):
        msg = 'coarse init checkpoint model_sig must be dict'
        raise TypeError(msg)
    if int(model_sig.get('in_chans', -1)) != 3:
        msg = 'coarse init checkpoint model_sig.in_chans must be 3'
        raise ValueError(msg)
    if int(model_sig.get('out_chans', -1)) != 1:
        msg = 'coarse init checkpoint model_sig.out_chans must be 1'
        raise ValueError(msg)

    if fine_model_sig is not None:
        if not isinstance(fine_model_sig, dict):
            msg = 'fine_model_sig must be dict'
            raise TypeError(msg)
        if int(fine_model_sig.get('in_chans', -1)) != 1:
            msg = 'fine model_sig.in_chans must be 1'
            raise ValueError(msg)
        if int(fine_model_sig.get('out_chans', -1)) != 1:
            msg = 'fine model_sig.out_chans must be 1'
            raise ValueError(msg)


# Layer convention:
# - first conv is the first non-seg_head 4D weight tensor in state_dict order
# - first BN stats are the first non-seg_head *.running_mean/var/num_batches_tracked
# - seg_head.* is always excluded when reset_seg_head=True

def build_fine_init_state_dict(
    coarse_state_dict: dict[str, torch.Tensor],
    fine_state_dict: dict[str, torch.Tensor],
    *,
    reset_seg_head: bool,
    reset_first_bn_stats: bool,
) -> dict[str, torch.Tensor]:
    coarse_sd = _require_state_dict('coarse_state_dict', coarse_state_dict)
    fine_sd = _require_state_dict('fine_state_dict', fine_state_dict)

    coarse_first_conv_key = _find_first_conv_key(coarse_sd)
    fine_first_conv_key = _find_first_conv_key(fine_sd)
    if coarse_first_conv_key != fine_first_conv_key:
        msg = (
            'first conv key mismatch between coarse and fine state_dict: '
            f'{coarse_first_conv_key!r} != {fine_first_conv_key!r}'
        )
        raise RuntimeError(msg)

    coarse_first_conv = coarse_sd[coarse_first_conv_key]
    fine_first_conv = fine_sd[fine_first_conv_key]
    if coarse_first_conv.ndim != 4 or fine_first_conv.ndim != 4:
        msg = 'first conv tensors must be 4D'
        raise RuntimeError(msg)
    if tuple(coarse_first_conv.shape[2:]) != tuple(fine_first_conv.shape[2:]):
        msg = 'first conv kernel shapes must match between coarse and fine'
        raise RuntimeError(msg)
    if int(fine_first_conv.shape[1]) != 1:
        msg = 'fine first conv must accept exactly one waveform channel'
        raise RuntimeError(msg)
    if int(coarse_first_conv.shape[1]) < 1:
        msg = 'coarse first conv must have at least one input channel'
        raise RuntimeError(msg)

    out: dict[str, torch.Tensor] = {}
    for key, fine_value in fine_sd.items():
        if reset_seg_head and key.startswith('seg_head.'):
            continue
        if key == fine_first_conv_key:
            sliced = coarse_first_conv[:, 0:1, :, :].clone()
            if sliced.shape != fine_value.shape:
                msg = (
                    'first conv sliced waveform weight shape mismatch: '
                    f'{tuple(sliced.shape)} != {tuple(fine_value.shape)}'
                )
                raise RuntimeError(msg)
            out[key] = sliced
            continue
        if key not in coarse_sd:
            msg = f'coarse checkpoint missing fine key: {key}'
            raise RuntimeError(msg)
        coarse_value = coarse_sd[key]
        if not isinstance(coarse_value, torch.Tensor):
            msg = f'coarse checkpoint key is not a tensor: {key}'
            raise RuntimeError(msg)
        if coarse_value.shape != fine_value.shape:
            msg = (
                f'shape mismatch for key {key}: '
                f'{tuple(coarse_value.shape)} != {tuple(fine_value.shape)}'
            )
            raise RuntimeError(msg)
        out[key] = coarse_value.clone()

    if reset_first_bn_stats:
        bn_prefix = _find_first_bn_prefix(fine_sd)
        for suffix in ('running_mean', 'running_var', 'num_batches_tracked'):
            key = f'{bn_prefix}.{suffix}'
            if key not in fine_sd:
                msg = f'fine state_dict missing first BN stat key: {key}'
                raise RuntimeError(msg)
            if reset_seg_head and key.startswith('seg_head.'):
                continue
            out[key] = fine_sd[key].clone()

    return out


def load_fine_init_from_coarse_checkpoint(
    model: torch.nn.Module,
    coarse_ckpt_path: str | Path,
    *,
    fine_model_sig: dict[str, Any],
    reset_seg_head: bool,
    reset_first_bn_stats: bool,
) -> None:
    if not isinstance(model, torch.nn.Module):
        msg = 'model must be torch.nn.Module'
        raise TypeError(msg)

    ckpt = load_checkpoint(Path(coarse_ckpt_path))
    validate_coarse_checkpoint_for_fine(ckpt, fine_model_sig=fine_model_sig)

    coarse_state_dict = ckpt.get('model_state_dict')
    if not isinstance(coarse_state_dict, dict):
        msg = 'checkpoint model_state_dict must be dict'
        raise RuntimeError(msg)

    load_state_dict = build_fine_init_state_dict(
        coarse_state_dict,
        model.state_dict(),
        reset_seg_head=reset_seg_head,
        reset_first_bn_stats=reset_first_bn_stats,
    )
    load_result = model.load_state_dict(load_state_dict, strict=False)
    unexpected_keys = list(load_result.unexpected_keys)
    missing_keys = list(load_result.missing_keys)

    if unexpected_keys:
        msg = 'unexpected keys while loading coarse-to-fine init: ' + ', '.join(
            unexpected_keys
        )
        raise RuntimeError(msg)

    if reset_seg_head:
        invalid_missing = [key for key in missing_keys if not key.startswith('seg_head.')]
        if invalid_missing:
            msg = 'non-seg_head missing keys while loading coarse-to-fine init: ' + ', '.join(
                invalid_missing
            )
            raise RuntimeError(msg)
    elif missing_keys:
        msg = 'missing keys while loading coarse-to-fine init: ' + ', '.join(missing_keys)
        raise RuntimeError(msg)
