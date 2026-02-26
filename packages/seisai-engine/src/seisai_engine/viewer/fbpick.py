from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from seisai_dataset.builder import MakeOffsetChannel
from seisai_transforms.augment import PerTraceStandardize
from seisai_utils.validator import validate_array

from seisai_engine.infer.ckpt_meta import resolve_output_ids, resolve_softmax_axis
from seisai_engine.pipelines.common import load_checkpoint
from seisai_engine.predict import infer_tiled_chw

from .model_cache import ViewerModelBundle, get_or_create_model_bundle

__all__ = ['infer_prob_hw']

_CHANNEL_PATTERN = re.compile(r'ch(\d+)')


def _require_strict_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f'{name} must be int'
        raise TypeError(msg)
    return int(value)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        if device.type == 'cpu':
            return torch.device('cpu')
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                msg = 'device="cuda" requested but CUDA is not available'
                raise ValueError(msg)
            if device.index is None:
                return torch.device('cuda')
            idx = int(device.index)
            count = torch.cuda.device_count()
            if idx >= count:
                msg = f'device="{device}" is out of range; device_count={count}'
                raise ValueError(msg)
            return torch.device(f'cuda:{idx}')
        msg = f'unsupported device type: {device.type}'
        raise ValueError(msg)

    normalized = str(device).strip().lower()
    if normalized in ('', 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if normalized == 'cpu':
        return torch.device('cpu')
    if normalized == 'cuda':
        if not torch.cuda.is_available():
            msg = 'device="cuda" requested but CUDA is not available'
            raise ValueError(msg)
        return torch.device('cuda')
    if normalized.startswith('cuda:'):
        idx_text = normalized.split(':', 1)[1].strip()
        if not idx_text:
            msg = 'device="cuda:" is invalid; expected cuda:N'
            raise ValueError(msg)
        try:
            idx = int(idx_text)
        except ValueError as exc:
            msg = f'device="{device}" is invalid; expected cuda:N'
            raise ValueError(msg) from exc
        if idx < 0:
            msg = f'device="{device}" is invalid; expected non-negative index'
            raise ValueError(msg)
        if not torch.cuda.is_available():
            msg = f'device="{device}" requested but CUDA is not available'
            raise ValueError(msg)
        count = torch.cuda.device_count()
        if idx >= count:
            msg = f'device="{device}" is out of range; device_count={count}'
            raise ValueError(msg)
        return torch.device(f'cuda:{idx}')
    msg = f'device="{device}" is invalid; expected auto|cpu|cuda|cuda:N'
    raise ValueError(msg)


def _resolve_channel_index(
    channel: str | int | None, *, output_ids: tuple[str, ...]
) -> int:
    out_chans = len(output_ids)
    if out_chans <= 0:
        msg = 'output_ids must be non-empty'
        raise ValueError(msg)

    if channel is None:
        return output_ids.index('P') if 'P' in output_ids else 0

    if isinstance(channel, bool):
        msg = 'channel must be int, str, or None'
        raise TypeError(msg)

    if isinstance(channel, int):
        if 0 <= channel < out_chans:
            return int(channel)
        msg = f'channel index {channel} is out of range for out_chans={out_chans}'
        raise ValueError(msg)

    if isinstance(channel, str):
        token = channel.strip()
        if token in output_ids:
            return output_ids.index(token)

        match = _CHANNEL_PATTERN.fullmatch(token)
        if match is not None:
            idx = int(match.group(1))
            if 0 <= idx < out_chans:
                return idx
            msg = f'channel "{token}" is out of range for out_chans={out_chans}'
            raise ValueError(msg)

        msg = f'channel "{token}" not found in output_ids={list(output_ids)}'
        raise ValueError(msg)

    msg = 'channel must be int, str, or None'
    raise TypeError(msg)


def _apply_softmax(
    logits_chw: torch.Tensor,
    *,
    softmax_axis: str,
    tau: float,
    out_chans: int,
) -> torch.Tensor:
    if logits_chw.ndim != 3:
        msg = f'logits_chw must be 3D (C,H,W), got {tuple(logits_chw.shape)}'
        raise ValueError(msg)
    if int(logits_chw.shape[0]) != int(out_chans):
        msg = (
            f'logits_chw channel dim {int(logits_chw.shape[0])} '
            f'!= out_chans {out_chans}'
        )
        raise ValueError(msg)

    tau_value = float(tau)
    if not np.isfinite(tau_value) or tau_value <= 0.0:
        msg = f'tau must be finite and > 0, got {tau}'
        raise ValueError(msg)

    scaled = logits_chw / tau_value
    if softmax_axis == 'channel':
        return torch.softmax(scaled, dim=0)
    if softmax_axis == 'time':
        if out_chans != 1:
            msg = 'softmax_axis="time" requires out_chans==1'
            raise ValueError(msg)
        return torch.softmax(scaled, dim=-1)
    msg = f'unsupported softmax_axis: {softmax_axis}'
    raise ValueError(msg)


def _build_input_chw(
    *,
    section_hw: np.ndarray,
    in_chans: int,
    offsets_h: np.ndarray | None,
) -> np.ndarray:
    waveform_hw = np.ascontiguousarray(section_hw, dtype=np.float32)
    standardize = PerTraceStandardize()
    standardized_hw = standardize(waveform_hw)
    if not isinstance(standardized_hw, np.ndarray):
        msg = 'PerTraceStandardize must return numpy.ndarray for numpy input'
        raise TypeError(msg)
    waveform_std_hw = np.ascontiguousarray(standardized_hw, dtype=np.float32)

    h, w = waveform_std_hw.shape
    if in_chans == 1:
        return np.ascontiguousarray(waveform_std_hw[None, :, :], dtype=np.float32)
    if in_chans == 2:
        if offsets_h is None:
            msg = 'offsets_h is required when in_chans==2'
            raise ValueError(msg)
        validate_array(
            offsets_h, allowed_ndims=(1,), name='offsets_h', backend='numpy'
        )
        offsets = np.ascontiguousarray(offsets_h, dtype=np.float32)
        if offsets.shape != (h,):
            msg = f'offsets_h must have shape ({h},), got {offsets.shape}'
            raise ValueError(msg)

        sample = {
            'x_view': waveform_std_hw,
            'meta': {
                'offsets_view': offsets,
                'trace_valid': np.ones((h,), dtype=np.bool_),
            },
        }
        make_offset = MakeOffsetChannel(dst='offset_ch', normalize=True)
        make_offset(sample)
        offset_hw = sample['offset_ch']
        if not isinstance(offset_hw, np.ndarray):
            msg = 'MakeOffsetChannel must produce numpy.ndarray'
            raise TypeError(msg)
        if offset_hw.shape != (h, w):
            msg = f'offset channel must have shape ({h}, {w}), got {offset_hw.shape}'
            raise ValueError(msg)

        x_chw = np.stack(
            [waveform_std_hw, np.ascontiguousarray(offset_hw, dtype=np.float32)],
            axis=0,
        )
        return np.ascontiguousarray(x_chw, dtype=np.float32)

    if in_chans > 2:
        msg = f'in_chans={in_chans} is not supported in Phase1'
        raise ValueError(msg)
    msg = f'in_chans must be positive, got {in_chans}'
    raise ValueError(msg)


def _pad_chw_to_min_tile(
    x_chw: np.ndarray, tile: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    validate_array(x_chw, allowed_ndims=(3,), name='x_chw', backend='numpy')
    x = np.ascontiguousarray(x_chw, dtype=np.float32)

    _, h, w = x.shape
    tile_h, tile_w = tile
    orig_hw = (int(h), int(w))
    target_hw = (max(h, tile_h), max(w, tile_w))

    if target_hw == orig_hw:
        return x, orig_hw, target_hw

    x_pad = np.zeros((x.shape[0], target_hw[0], target_hw[1]), dtype=np.float32)
    x_pad[:, :h, :w] = x
    return x_pad, orig_hw, target_hw


def _crop_logits_chw(
    logits_chw: torch.Tensor, orig_hw: tuple[int, int]
) -> torch.Tensor:
    if logits_chw.ndim != 3:
        msg = f'logits_chw must be (C,H,W), got {tuple(logits_chw.shape)}'
        raise ValueError(msg)

    h, w = orig_hw
    if h > int(logits_chw.shape[1]) or w > int(logits_chw.shape[2]):
        msg = (
            f'orig_hw {orig_hw} exceeds logits spatial shape '
            f'{tuple(logits_chw.shape[1:])}'
        )
        raise ValueError(msg)
    return logits_chw[:, :h, :w].contiguous()


def _build_model_bundle(*, ckpt_path: Path, device: torch.device) -> ViewerModelBundle:
    ckpt = load_checkpoint(ckpt_path)
    model_sig = ckpt['model_sig']
    if not isinstance(model_sig, dict):
        msg = 'checkpoint model_sig must be dict'
        raise TypeError(msg)

    if 'in_chans' not in model_sig:
        msg = 'checkpoint model_sig missing: in_chans'
        raise KeyError(msg)
    if 'out_chans' not in model_sig:
        msg = 'checkpoint model_sig missing: out_chans'
        raise KeyError(msg)
    in_chans = _require_strict_int(model_sig['in_chans'], name='model_sig.in_chans')
    out_chans = _require_strict_int(model_sig['out_chans'], name='model_sig.out_chans')

    model_kwargs = dict(model_sig)
    model_kwargs['pretrained'] = False
    from seisai_models.models.encdec2d import EncDec2D

    model = EncDec2D(**model_kwargs)
    model.use_tta = False

    if ckpt.get('infer_used_ema') is True and 'ema_state_dict' in ckpt:
        state_dict = ckpt['ema_state_dict']
    else:
        state_dict = ckpt['model_state_dict']
    if not isinstance(state_dict, dict):
        msg = 'checkpoint state_dict must be dict'
        raise TypeError(msg)

    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()

    return ViewerModelBundle(
        model=model,
        in_chans=in_chans,
        out_chans=out_chans,
        softmax_axis=resolve_softmax_axis(
            ckpt=ckpt,
            out_chans=out_chans,
            pipeline_name='psn',
        ),
        output_ids=resolve_output_ids(
            ckpt=ckpt,
            out_chans=out_chans,
            pipeline_name='psn',
        ),
    )


def _get_model_bundle(
    *, ckpt_path: str | Path, device: torch.device
) -> ViewerModelBundle:
    return get_or_create_model_bundle(
        ckpt_path=ckpt_path,
        device_str=str(device),
        loader=lambda resolved_ckpt: _build_model_bundle(
            ckpt_path=resolved_ckpt, device=device
        ),
    )


@torch.no_grad()
def infer_prob_hw(
    section_hw: np.ndarray,
    *,
    ckpt_path: str | Path,
    offsets_h: np.ndarray | None = None,
    channel: str | int | None = None,
    device: str | torch.device = 'auto',
    tile: tuple[int, int] = (128, 6016),
    overlap: tuple[int, int] = (32, 32),
    amp: bool = True,
    tiles_per_batch: int = 4,
    tau: float = 1.0,
) -> np.ndarray:
    """Run tiled viewer inference and return one probability map `(H, W)`.

    Notes
    -----
    - This API always returns a single-channel map named "prob", but the
      normalization axis depends on the checkpoint/model settings.
    - If input `(H, W)` is smaller than `tile=(tile_h, tile_w)`, inference is
      run on zero-padded `x_chw` internally and cropped back to original size.
      Padding is applied after per-trace standardization and optional offset
      channel creation.
    - `softmax_axis="channel"`: probabilities sum to 1 across channels at each
      pixel `(H, W)` (class-probability style, e.g. PSN).
    - `softmax_axis="time"`: probabilities sum to 1 across time/width `W` for
      each trace (distribution-over-time style, typically `out_chans==1`).
      Logits are cropped to original `(H, W)` before time-softmax so padded
      area does not affect normalization.
    - `channel` string input is matched after `strip()`; label matching remains
      case-sensitive against `output_ids`.

    """
    validate_array(section_hw, allowed_ndims=(2,), name='section_hw', backend='numpy')
    section = np.ascontiguousarray(section_hw, dtype=np.float32)
    orig_hw = (int(section.shape[0]), int(section.shape[1]))

    resolved_device = _resolve_device(device)
    bundle = _get_model_bundle(ckpt_path=ckpt_path, device=resolved_device)

    x_chw = _build_input_chw(
        section_hw=section,
        in_chans=bundle.in_chans,
        offsets_h=offsets_h,
    )
    x_pad_chw, x_orig_hw, target_hw = _pad_chw_to_min_tile(x_chw, tile=tile)
    if x_orig_hw != orig_hw:
        msg = f'x_chw spatial shape {x_orig_hw} != input shape {orig_hw}'
        raise ValueError(msg)

    logits_chw = infer_tiled_chw(
        bundle.model,
        x_pad_chw,
        tile=tile,
        overlap=overlap,
        amp=amp,
        tiles_per_batch=tiles_per_batch,
        use_tqdm=False,
    )
    if not isinstance(logits_chw, torch.Tensor):
        msg = f'infer_tiled_chw must return torch.Tensor, got {type(logits_chw)}'
        raise TypeError(msg)
    if logits_chw.ndim != 3:
        msg = f'logits_chw must be (C,H,W), got {tuple(logits_chw.shape)}'
        raise ValueError(msg)
    if int(logits_chw.shape[0]) != bundle.out_chans:
        msg = (
            f'logits_chw channel dim {int(logits_chw.shape[0])} '
            f'!= out_chans {bundle.out_chans}'
        )
        raise ValueError(msg)
    if tuple(logits_chw.shape[1:]) != target_hw:
        msg = (
            f'logits_chw spatial shape {tuple(logits_chw.shape[1:])} '
            f'!= padded input shape {target_hw}'
        )
        raise ValueError(msg)
    logits_crop_chw = _crop_logits_chw(logits_chw, orig_hw=orig_hw)

    probs_chw = _apply_softmax(
        logits_chw=logits_crop_chw.to(torch.float32),
        softmax_axis=bundle.softmax_axis,
        tau=tau,
        out_chans=bundle.out_chans,
    )
    channel_idx = _resolve_channel_index(channel, output_ids=bundle.output_ids)

    prob_hw = probs_chw[channel_idx].detach().cpu().numpy()
    if tuple(prob_hw.shape) != orig_hw:
        msg = f'prob_hw shape {tuple(prob_hw.shape)} != input shape {orig_hw}'
        raise ValueError(msg)
    return np.ascontiguousarray(prob_hw, dtype=np.float32)
