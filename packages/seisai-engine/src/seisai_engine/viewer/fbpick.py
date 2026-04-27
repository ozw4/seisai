from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from seisai_utils.validator import validate_array

if TYPE_CHECKING:
	from .model_cache import ViewerModelBundle

__all__ = [
	'infer_prob_hw',
	'render_fbpick_overview',
	'save_fbpick_debug_png',
	'save_fbpick_overview_png',
	'save_fbpick_physics_qc_cdf_png',
	'save_fbpick_physics_qc_gather_png',
]

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
	from seisai_dataset.builder import MakeOffsetChannel
	from seisai_transforms.augment import PerTraceStandardize

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
		validate_array(offsets_h, allowed_ndims=(1,), name='offsets_h', backend='numpy')
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
	from seisai_engine.infer.ckpt_meta import resolve_output_ids, resolve_softmax_axis
	from seisai_engine.pipelines.common import load_checkpoint

	from .model_cache import ViewerModelBundle

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
	from .model_cache import get_or_create_model_bundle

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
	from seisai_engine.predict import infer_tiled_chw

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


def _require_overview_vector(
	final_payload: Mapping[str, np.ndarray],
	name: str,
	*,
	length: int,
) -> np.ndarray:
	if name not in final_payload:
		msg = f'final payload missing key: {name}'
		raise KeyError(msg)
	arr = np.asarray(final_payload[name])
	if arr.ndim != 1 or int(arr.shape[0]) != int(length):
		msg = f'{name} must be 1D with length {length}'
		raise ValueError(msg)
	return arr


def _resolve_overview_clip(raw_wave_hw: np.ndarray, *, clip_percentile: float) -> float:
	percentile = float(clip_percentile)
	if percentile <= 0.0 or percentile > 100.0:
		msg = 'clip_percentile must lie in (0, 100]'
		raise ValueError(msg)

	abs_wave = np.abs(np.asarray(raw_wave_hw, dtype=np.float32))
	finite_abs_wave = abs_wave[np.isfinite(abs_wave)]
	if finite_abs_wave.size == 0:
		return 1.0

	clip_value = float(np.percentile(finite_abs_wave, percentile))
	if np.isfinite(clip_value) and clip_value > 0.0:
		return clip_value

	max_value = float(np.max(finite_abs_wave))
	if np.isfinite(max_value) and max_value > 0.0:
		return max_value
	return 1.0


def _extract_batch_scalar(value: object, *, b: int) -> object:
	if torch.is_tensor(value):
		if value.ndim == 0:
			return value.detach().cpu().item()
		return value[b].detach().cpu().item()
	if isinstance(value, np.ndarray):
		if value.ndim == 0:
			return value.item()
		return value[b].item()
	if isinstance(value, (list, tuple)):
		return value[b]
	return value


def _extract_batch_vector(value: object, *, b: int, name: str) -> np.ndarray:
	if torch.is_tensor(value):
		if value.ndim < 2:
			msg = f'{name} must be batched with shape (B,H), got {tuple(value.shape)}'
			raise ValueError(msg)
		return value[b].detach().cpu().numpy()
	if isinstance(value, np.ndarray):
		if value.ndim < 2:
			msg = f'{name} must be batched with shape (B,H), got {value.shape}'
			raise ValueError(msg)
		return np.asarray(value[b])
	if isinstance(value, (list, tuple)):
		item = value[b]
		return np.asarray(item)
	msg = f'{name} must be tensor, ndarray, list, or tuple'
	raise TypeError(msg)


def _extract_title_field(
	batch: Mapping[str, object],
	*,
	b: int,
	key: str,
) -> str | None:
	value = batch.get(key)
	if value is not None:
		scalar = _extract_batch_scalar(value, b=b)
		text = str(scalar).strip()
		if text:
			return text

	meta_obj = batch.get('meta')
	if isinstance(meta_obj, Mapping) and key in meta_obj:
		scalar = _extract_batch_scalar(meta_obj[key], b=b)
		text = str(scalar).strip()
		if text:
			return text
	return None


def _make_debug_title(batch: Mapping[str, object], *, b: int) -> str | None:
	file_path = _extract_title_field(batch, b=b, key='file_path')
	key_name = _extract_title_field(batch, b=b, key='key_name')
	primary_unique = _extract_title_field(batch, b=b, key='primary_unique')
	secondary_key = _extract_title_field(batch, b=b, key='secondary_key')

	line1: list[str] = []
	if file_path is not None:
		line1.append(Path(file_path).name)

	line2: list[str] = []
	if key_name is not None and primary_unique is not None:
		line2.append(f'{key_name}={primary_unique}')
	elif key_name is not None:
		line2.append(key_name)
	elif primary_unique is not None:
		line2.append(primary_unique)
	if secondary_key is not None:
		line2.append(f'secondary={secondary_key}')

	parts: list[str] = []
	if line1:
		parts.append(' '.join(line1))
	if line2:
		parts.append(' | '.join(line2))
	if not parts:
		return None
	return '\n'.join(parts)


def save_fbpick_debug_png(
	out_png: str | Path,
	*,
	x_bchw: torch.Tensor,
	target_bchw: torch.Tensor,
	pred_bchw: torch.Tensor,
	batch: Mapping[str, object],
	b: int = 0,
	title: str | None = None,
	dpi: int = 150,
	clip_percentile: float = 99.0,
) -> Path:
	import matplotlib.pyplot as plt

	if x_bchw.ndim != 4:
		msg = f'x_bchw must be (B,C,H,W), got {tuple(x_bchw.shape)}'
		raise ValueError(msg)
	if target_bchw.ndim != 4:
		msg = f'target_bchw must be (B,C,H,W), got {tuple(target_bchw.shape)}'
		raise ValueError(msg)
	if pred_bchw.ndim != 4:
		msg = f'pred_bchw must be (B,C,H,W), got {tuple(pred_bchw.shape)}'
		raise ValueError(msg)

	out_path = Path(out_png).expanduser().resolve()
	if not out_path.parent.is_dir():
		msg = f'debug output directory not found: {out_path.parent}'
		raise FileNotFoundError(msg)

	dpi_int = int(dpi)
	if dpi_int <= 0:
		msg = 'dpi must be > 0'
		raise ValueError(msg)

	wave_hw = x_bchw[b, 0].detach().cpu().numpy().astype(np.float32, copy=False)
	target_hw = target_bchw[b, 0].detach().cpu().numpy().astype(np.float32, copy=False)
	pred_logits_hw = pred_bchw[b, 0].detach().cpu().to(torch.float32)
	pred_hw = (
		torch.softmax(pred_logits_hw, dim=-1).numpy().astype(np.float32, copy=False)
	)

	if wave_hw.ndim != 2 or target_hw.ndim != 2 or pred_hw.ndim != 2:
		msg = 'fbpick debug views must be 2D (H,W)'
		raise ValueError(msg)
	if wave_hw.shape != target_hw.shape or wave_hw.shape != pred_hw.shape:
		msg = (
			'fbpick debug tensors must share the same spatial shape, got '
			f'wave={wave_hw.shape} target={target_hw.shape} pred={pred_hw.shape}'
		)
		raise ValueError(msg)

	trace_valid = None
	fb_idx_view = None
	meta_obj = batch.get('meta')
	if isinstance(meta_obj, Mapping):
		if 'trace_valid' in meta_obj:
			trace_valid = np.asarray(
				_extract_batch_vector(
					meta_obj['trace_valid'], b=b, name='meta[trace_valid]'
				),
				dtype=np.bool_,
			)
		if 'fb_idx_view' in meta_obj:
			fb_idx_view = np.asarray(
				_extract_batch_vector(
					meta_obj['fb_idx_view'], b=b, name='meta[fb_idx_view]'
				),
				dtype=np.int64,
			)

	if trace_valid is None:
		trace_valid = np.ones((wave_hw.shape[0],), dtype=np.bool_)
	if fb_idx_view is None:
		fb_idx_view = np.full((wave_hw.shape[0],), -1, dtype=np.int64)
	if trace_valid.shape != (wave_hw.shape[0],):
		msg = f'meta[trace_valid] must have shape ({wave_hw.shape[0]},)'
		raise ValueError(msg)
	if fb_idx_view.shape != (wave_hw.shape[0],):
		msg = f'meta[fb_idx_view] must have shape ({wave_hw.shape[0]},)'
		raise ValueError(msg)

	clip_value = _resolve_overview_clip(wave_hw, clip_percentile=clip_percentile)
	title_text = title if title is not None else _make_debug_title(batch, b=b)
	target_pick_i = np.argmax(target_hw, axis=-1).astype(np.float32, copy=False)
	pred_pick_i = np.argmax(pred_hw, axis=-1).astype(np.float32, copy=False)
	valid_pick_mask = trace_valid & (fb_idx_view >= 0)
	x = np.arange(wave_hw.shape[0], dtype=np.float32)
	heat_vmax = float(max(np.max(target_hw), np.max(pred_hw), 1e-6))

	fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
	axes[0].imshow(
		wave_hw.T,
		cmap='gray',
		aspect='auto',
		interpolation='nearest',
		origin='upper',
		vmin=-clip_value,
		vmax=clip_value,
	)
	axes[0].plot(
		x[valid_pick_mask],
		target_pick_i[valid_pick_mask],
		color='yellow',
		lw=1.0,
		alpha=0.9,
		label='target',
	)
	axes[0].plot(
		x[valid_pick_mask],
		pred_pick_i[valid_pick_mask],
		color='#00d7ff',
		lw=1.0,
		alpha=0.9,
		label='pred',
	)
	axes[0].set_title('wave')
	axes[0].set_xlabel('Trace Index')
	axes[0].set_ylabel('Sample Index')
	axes[0].legend(loc='upper right', fontsize=8)

	axes[1].imshow(
		target_hw.T,
		cmap='magma',
		aspect='auto',
		interpolation='nearest',
		origin='upper',
		vmin=0.0,
		vmax=heat_vmax,
	)
	axes[1].set_title('target')
	axes[1].set_xlabel('Trace Index')

	axes[2].imshow(
		pred_hw.T,
		cmap='magma',
		aspect='auto',
		interpolation='nearest',
		origin='upper',
		vmin=0.0,
		vmax=heat_vmax,
	)
	axes[2].set_title('prediction')
	axes[2].set_xlabel('Trace Index')

	if title_text is not None:
		fig.suptitle(str(title_text))
		fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
	else:
		fig.tight_layout()
	fig.savefig(out_path, dpi=dpi_int)
	plt.close(fig)
	return out_path


def render_fbpick_overview(
	raw_wave_hw: np.ndarray,
	final_payload: Mapping[str, np.ndarray],
	*,
	title: str | None = None,
	clip_percentile: float = 99.0,
):
	import matplotlib.pyplot as plt

	wave = np.ascontiguousarray(np.asarray(raw_wave_hw, dtype=np.float32))
	if wave.ndim != 2:
		msg = f'raw_wave_hw must be 2D, got {wave.shape}'
		raise ValueError(msg)

	n_traces, _ = wave.shape
	clip_value = _resolve_overview_clip(wave, clip_percentile=clip_percentile)
	coarse_pick_i = _require_overview_vector(
		final_payload, 'coarse_pick_i', length=n_traces
	)
	robust_pick_i = _require_overview_vector(
		final_payload, 'robust_pick_i', length=n_traces
	)
	window_start_i = _require_overview_vector(
		final_payload, 'window_start_i', length=n_traces
	)
	window_end_i = _require_overview_vector(
		final_payload, 'window_end_i', length=n_traces
	)
	final_pick_i = _require_overview_vector(
		final_payload, 'final_pick_i', length=n_traces
	)
	high_conf_mask = np.asarray(
		_require_overview_vector(final_payload, 'high_conf_mask', length=n_traces),
		dtype=np.bool_,
	)

	fig_width = max(10.0, float(n_traces) / 16.0)
	fig, ax = plt.subplots(figsize=(fig_width, 8.0))
	ax.imshow(
		wave.T,
		cmap='gray',
		aspect='auto',
		interpolation='nearest',
		origin='upper',
		vmin=-clip_value,
		vmax=clip_value,
	)

	x = np.arange(n_traces, dtype=np.float32)
	ax.plot(
		x,
		coarse_pick_i.astype(np.float32),
		color='#7fd3ff',
		lw=1.0,
		alpha=0.9,
		label='coarse_pick_i',
	)
	ax.plot(
		x,
		robust_pick_i.astype(np.float32),
		color='yellow',
		lw=1.0,
		alpha=0.9,
		label='robust_pick_i',
	)
	ax.plot(
		x,
		window_start_i.astype(np.float32),
		color='yellow',
		lw=0.9,
		ls='--',
		alpha=0.8,
		label='window_start_i',
	)
	ax.plot(
		x,
		window_end_i.astype(np.float32),
		color='yellow',
		lw=0.9,
		ls='--',
		alpha=0.8,
		label='window_end_i',
	)
	ax.plot(
		x,
		final_pick_i.astype(np.float32),
		color='red',
		lw=1.2,
		alpha=0.95,
		label='final_pick_i',
	)
	ax.scatter(
		x[high_conf_mask],
		final_pick_i.astype(np.float32)[high_conf_mask],
		color='lime',
		s=14.0,
		alpha=0.95,
		label='high_conf_final_pick',
		zorder=5,
	)

	ax.set_xlabel('Trace Index')
	ax.set_ylabel('Sample Index')
	ax.set_title('fbpick overview' if title is None else str(title))
	ax.legend(loc='upper right', fontsize=8)
	return fig, ax


def save_fbpick_physics_qc_gather_png(
	out_png: str | Path,
	*,
	raw_wave_hw: np.ndarray,
	gt_pick_i: np.ndarray,
	coarse_pick_i: np.ndarray,
	robust_pick_i: np.ndarray,
	title: str | None = None,
	dpi: int = 150,
	clip_percentile: float = 99.0,
) -> Path:
	import matplotlib.pyplot as plt

	wave = np.ascontiguousarray(np.asarray(raw_wave_hw, dtype=np.float32))
	if wave.ndim != 2:
		msg = f'raw_wave_hw must be 2D, got {wave.shape}'
		raise ValueError(msg)

	n_traces, n_samples = wave.shape
	if n_traces <= 0 or n_samples <= 0:
		msg = 'raw_wave_hw must be non-empty'
		raise ValueError(msg)

	gt = np.asarray(gt_pick_i, dtype=np.int64)
	coarse = np.asarray(coarse_pick_i, dtype=np.int64)
	robust = np.asarray(robust_pick_i, dtype=np.int64)
	for name, arr in (
		('gt_pick_i', gt),
		('coarse_pick_i', coarse),
		('robust_pick_i', robust),
	):
		if arr.ndim != 1 or int(arr.shape[0]) != n_traces:
			msg = f'{name} must be 1D with length {n_traces}'
			raise ValueError(msg)

	dpi_int = int(dpi)
	if dpi_int <= 0:
		msg = 'dpi must be > 0'
		raise ValueError(msg)

	valid_gt = (gt > 0) & (gt < int(n_samples))
	robust_window_start = robust.astype(np.int64) - 128
	robust_window_end = robust.astype(np.int64) + 127
	in_robust_window = (
		valid_gt & (gt >= robust_window_start) & (gt <= robust_window_end)
	)
	coarse_err = coarse.astype(np.float32) - gt.astype(np.float32)
	robust_err = robust.astype(np.float32) - gt.astype(np.float32)
	coarse_err[~valid_gt] = np.nan
	robust_err[~valid_gt] = np.nan

	x = np.arange(n_traces, dtype=np.float32)
	clip_value = _resolve_overview_clip(wave, clip_percentile=clip_percentile)
	fig_width = max(14.0, float(n_traces) / 12.0)
	fig, axes = plt.subplots(
		1,
		3,
		figsize=(fig_width, 6.0),
		gridspec_kw={'width_ratios': [2.2, 1.5, 0.7]},
	)

	axes[0].imshow(
		wave.T,
		cmap='gray',
		aspect='auto',
		interpolation='nearest',
		origin='upper',
		vmin=-clip_value,
		vmax=clip_value,
	)
	axes[0].plot(
		x[valid_gt],
		gt.astype(np.float32)[valid_gt],
		color='lime',
		lw=1.2,
		alpha=0.95,
		label='GT pick',
	)
	axes[0].plot(
		x[valid_gt],
		coarse.astype(np.float32)[valid_gt],
		color='#00a6ff',
		lw=1.0,
		alpha=0.9,
		label='coarse pick',
	)
	axes[0].plot(
		x[valid_gt],
		robust.astype(np.float32)[valid_gt],
		color='yellow',
		lw=1.0,
		alpha=0.95,
		label='robust pick',
	)
	axes[0].plot(
		x[valid_gt],
		np.clip(robust_window_start, 0, n_samples - 1).astype(np.float32)[valid_gt],
		color='yellow',
		lw=0.8,
		ls='--',
		alpha=0.75,
		label='robust window start',
	)
	axes[0].plot(
		x[valid_gt],
		np.clip(robust_window_end, 0, n_samples - 1).astype(np.float32)[valid_gt],
		color='yellow',
		lw=0.8,
		ls=':',
		alpha=0.75,
		label='robust window end',
	)
	axes[0].set_title('waveform and picks')
	axes[0].set_xlabel('Trace Index')
	axes[0].set_ylabel('Sample Index')
	axes[0].legend(loc='upper right', fontsize=8)

	axes[1].plot(x, coarse_err, color='#00a6ff', lw=1.0, label='coarse - GT')
	axes[1].plot(x, robust_err, color='yellow', lw=1.0, label='robust - GT')
	for y in (0, 32, -32, 64, -64, 127, -127):
		if y == 0:
			axes[1].axhline(y, color='black', lw=0.9, alpha=0.8)
		elif abs(y) == 32:
			axes[1].axhline(y, color='gray', lw=0.8, ls='--', alpha=0.7)
		elif abs(y) == 64:
			axes[1].axhline(y, color='gray', lw=0.8, ls=':', alpha=0.7)
		else:
			axes[1].axhline(y, color='red', lw=0.8, ls='--', alpha=0.65)
	axes[1].set_title('pick error')
	axes[1].set_xlabel('Trace Index')
	axes[1].set_ylabel('Sample Error')
	axes[1].legend(loc='upper right', fontsize=8)

	mask_values = in_robust_window.astype(np.float32)[None, :]
	axes[2].imshow(
		mask_values,
		cmap='gray_r',
		aspect='auto',
		interpolation='nearest',
		origin='upper',
		vmin=0.0,
		vmax=1.0,
	)
	axes[2].set_title('GT in robust window')
	axes[2].set_xlabel('Trace Index')
	axes[2].set_yticks([0])
	axes[2].set_yticklabels(['mask'])

	if title is not None:
		fig.suptitle(str(title))
		fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
	else:
		fig.tight_layout()

	out_path = Path(out_png).expanduser().resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=dpi_int)
	plt.close(fig)
	return out_path


def save_fbpick_physics_qc_cdf_png(
	out_png: str | Path,
	*,
	coarse_abs_err: np.ndarray,
	robust_abs_err: np.ndarray,
	title: str | None = None,
	dpi: int = 150,
) -> Path:
	import matplotlib.pyplot as plt

	coarse = np.asarray(coarse_abs_err, dtype=np.float64).reshape(-1)
	robust = np.asarray(robust_abs_err, dtype=np.float64).reshape(-1)
	coarse = coarse[np.isfinite(coarse)]
	robust = robust[np.isfinite(robust)]

	dpi_int = int(dpi)
	if dpi_int <= 0:
		msg = 'dpi must be > 0'
		raise ValueError(msg)

	fig, ax = plt.subplots(figsize=(8.0, 5.5))
	if coarse.size > 0:
		x_coarse = np.sort(coarse)
		y_coarse = np.arange(1, int(x_coarse.size) + 1, dtype=np.float64) / float(
			x_coarse.size
		)
		ax.plot(x_coarse, y_coarse, color='#00a6ff', lw=1.6, label='coarse')
	if robust.size > 0:
		x_robust = np.sort(robust)
		y_robust = np.arange(1, int(x_robust.size) + 1, dtype=np.float64) / float(
			x_robust.size
		)
		ax.plot(x_robust, y_robust, color='orange', lw=1.6, label='robust')

	for threshold in (32, 64, 127):
		ax.axvline(
			threshold,
			color='gray',
			lw=0.8,
			ls='--',
			alpha=0.55,
		)
	if coarse.size == 0 and robust.size == 0:
		ax.text(
			0.5,
			0.5,
			'no valid GT traces',
			ha='center',
			va='center',
			transform=ax.transAxes,
		)

	ax.set_title('pick absolute error CDF' if title is None else str(title))
	ax.set_xlabel('Absolute Error (samples)')
	ax.set_ylabel('CDF')
	ax.set_ylim(0.0, 1.0)
	ax.grid(True, alpha=0.25)
	ax.legend(loc='lower right', fontsize=9)
	fig.tight_layout()

	out_path = Path(out_png).expanduser().resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=dpi_int)
	plt.close(fig)
	return out_path


def save_fbpick_overview_png(
	out_png: str | Path,
	*,
	raw_wave_hw: np.ndarray,
	final_payload: Mapping[str, np.ndarray],
	title: str | None = None,
	dpi: int = 150,
	clip_percentile: float = 99.0,
) -> Path:
	import matplotlib.pyplot as plt

	dpi_int = int(dpi)
	if dpi_int <= 0:
		msg = 'dpi must be > 0'
		raise ValueError(msg)

	out_path = Path(out_png).expanduser().resolve()
	if not out_path.parent.is_dir():
		msg = f'overview output directory not found: {out_path.parent}'
		raise FileNotFoundError(msg)

	fig, _ = render_fbpick_overview(
		raw_wave_hw,
		final_payload,
		title=title,
		clip_percentile=clip_percentile,
	)
	fig.tight_layout()
	fig.savefig(out_path, dpi=dpi_int)
	plt.close(fig)
	return out_path
