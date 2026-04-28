from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from seisai_dataset import InputOnlyPlan
from seisai_engine.pipelines.fbpick.common import (
    build_lineage_payload,
    save_coarse_npz,
)

from .build_dataset import build_raw_infer_dataset, collate_input_meta_list
from .build_plan import build_plan
from .config import (
    COARSE_IN_CHANS,
    COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE,
    COARSE_TIME_LEN,
    COARSE_TRACE_LEN,
    load_coarse_infer_config,
)
from .time_axis import project_coarse_indices_to_raw_time

__all__ = [
    'restore_anchor_predictions_to_full_traces',
    'run_coarse_infer',
    'validate_checkpoint_for_global_anchor_infer',
]


def _softmax_last_axis(logits_hw: np.ndarray) -> np.ndarray:
    shifted = logits_hw - logits_hw.max(axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    denom = exp.sum(axis=-1, keepdims=True)
    return exp / denom


def _extract_anchor_predictions(logits_hw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits_hw, dtype=np.float32)
    if logits.ndim != 2:
        msg = f'anchor logits must be 2D, got shape={logits.shape}'
        raise ValueError(msg)
    prob = _softmax_last_axis(logits)
    return (
        logits.argmax(axis=-1).astype(np.int64),
        prob.max(axis=-1).astype(np.float32),
    )


def validate_checkpoint_for_global_anchor_infer(
    ckpt: dict[str, Any],
    *,
    model_sig: dict[str, Any],
) -> None:
    if not isinstance(ckpt, dict):
        msg = 'ckpt must be dict'
        raise TypeError(msg)
    if not isinstance(model_sig, dict):
        msg = 'model_sig must be dict'
        raise TypeError(msg)

    pipeline = ckpt.get('pipeline')
    if pipeline != 'fbpick':
        msg = f'coarse infer checkpoint pipeline must be "fbpick", got {pipeline!r}'
        raise ValueError(msg)

    ckpt_model_sig = ckpt.get('model_sig')
    if not isinstance(ckpt_model_sig, dict):
        msg = 'checkpoint model_sig must be dict'
        raise TypeError(msg)
    for key, value in ckpt_model_sig.items():
        if key in model_sig and model_sig[key] != value:
            msg = (
                f'checkpoint model_sig mismatch for {key}: '
                f'{value!r} != {model_sig[key]!r}'
            )
            raise ValueError(msg)

    output_ids = ckpt.get('output_ids')
    if output_ids is not None:
        if not isinstance(output_ids, (list, tuple)):
            msg = 'checkpoint output_ids must be list[str] or tuple[str, ...]'
            raise TypeError(msg)
        if list(output_ids) != ['P']:
            msg = (
                'coarse infer checkpoint output_ids must be ["P"], '
                f'got {output_ids!r}'
            )
            raise ValueError(msg)

    softmax_axis = ckpt.get('softmax_axis')
    if softmax_axis is not None and softmax_axis != 'time':
        msg = (
            'coarse infer checkpoint softmax_axis must be "time", '
            f'got {softmax_axis!r}'
        )
        raise ValueError(msg)

    expected_meta = {
        'coarse_input_mode': COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE,
        'coarse_trace_len': COARSE_TRACE_LEN,
        'coarse_time_len': COARSE_TIME_LEN,
        'coarse_in_chans': COARSE_IN_CHANS,
    }
    for key, expected in expected_meta.items():
        actual = ckpt.get(key)
        if actual != expected:
            legacy_hint = ''
            if key == 'coarse_input_mode' and actual is None:
                legacy_hint = (
                    ' This checkpoint appears to be from the legacy tiled coarse '
                    'pipeline.'
                )
            msg = (
                f'Invalid fbpick-coarse checkpoint: expected {key}={expected!r}, '
                f'got {actual!r}.{legacy_hint}'
            )
            raise ValueError(msg)


def restore_anchor_predictions_to_full_traces(
    *,
    raw_trace_indices: np.ndarray,
    anchor_source_pos: np.ndarray,
    trace_valid: np.ndarray,
    segment_id: np.ndarray,
    anchor_bin_start_pos: np.ndarray,
    anchor_bin_stop_pos: np.ndarray,
    anchor_pick_i_raw: np.ndarray,
    anchor_pmax: np.ndarray,
    n_samples_orig: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_indices = np.asarray(raw_trace_indices, dtype=np.int64)
    valid = np.asarray(trace_valid, dtype=np.bool_)
    anchor_pos = np.asarray(anchor_source_pos, dtype=np.int64)
    seg_id = np.asarray(segment_id, dtype=np.int64)
    bin_start = np.asarray(anchor_bin_start_pos, dtype=np.int64)
    bin_stop = np.asarray(anchor_bin_stop_pos, dtype=np.int64)
    pick_raw = np.asarray(anchor_pick_i_raw, dtype=np.float32)
    pmax = np.asarray(anchor_pmax, dtype=np.float32)

    h = int(valid.shape[0])
    for name, arr in (
        ('anchor_source_pos', anchor_pos),
        ('segment_id', seg_id),
        ('anchor_bin_start_pos', bin_start),
        ('anchor_bin_stop_pos', bin_stop),
        ('anchor_pick_i_raw', pick_raw),
        ('anchor_pmax', pmax),
    ):
        if arr.shape != (h,):
            msg = f'{name} must have shape ({h},), got {arr.shape}'
            raise ValueError(msg)

    n_samples = int(n_samples_orig)
    if raw_indices.ndim != 1 or raw_indices.size == 0:
        msg = 'raw_trace_indices must be a non-empty 1D array'
        raise ValueError(msg)
    if n_samples <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    out_raw_indices: list[np.ndarray] = []
    out_pick_parts: list[np.ndarray] = []
    out_pmax_parts: list[np.ndarray] = []

    valid_segment_ids = sorted(int(value) for value in np.unique(seg_id[valid]))
    for sid in valid_segment_ids:
        mask = valid & (seg_id == sid)
        if not np.any(mask):
            continue
        start = int(np.min(bin_start[mask]))
        stop = int(np.max(bin_stop[mask]))
        if start < 0 or stop <= start or stop > int(raw_indices.size):
            msg = f'invalid anchor bin span for segment {sid}: [{start}, {stop})'
            raise ValueError(msg)

        raw_positions = np.arange(start, stop, dtype=np.int64)
        anchor_positions = anchor_pos[mask].astype(np.float64, copy=False)
        order = np.argsort(anchor_positions, kind='mergesort')
        anchor_positions = anchor_positions[order]
        pick_values = pick_raw[mask][order].astype(np.float64, copy=False)
        pmax_values = pmax[mask][order].astype(np.float64, copy=False)

        if anchor_positions.size == 1:
            pick_interp = np.full(
                raw_positions.shape,
                float(pick_values[0]),
                dtype=np.float64,
            )
            pmax_interp = np.full(
                raw_positions.shape,
                float(pmax_values[0]),
                dtype=np.float64,
            )
        else:
            pick_interp = np.interp(
                raw_positions.astype(np.float64),
                anchor_positions,
                pick_values,
            )
            pmax_interp = np.interp(
                raw_positions.astype(np.float64),
                anchor_positions,
                pmax_values,
            )

        out_raw_indices.append(raw_indices[raw_positions])
        out_pick_parts.append(pick_interp.astype(np.float32, copy=False))
        out_pmax_parts.append(pmax_interp.astype(np.float32, copy=False))

    if not out_raw_indices:
        msg = 'no valid anchor predictions to restore'
        raise RuntimeError(msg)

    restored_raw_indices = np.concatenate(out_raw_indices).astype(np.int64, copy=False)
    restored_pick_i = np.rint(np.concatenate(out_pick_parts))
    restored_pick_i = np.clip(restored_pick_i, 0, n_samples - 1).astype(np.int32)
    restored_pmax = np.concatenate(out_pmax_parts).astype(np.float32, copy=False)
    restored_pmax = np.clip(restored_pmax, 0.0, 1.0).astype(np.float32, copy=False)
    return restored_raw_indices, restored_pick_i, restored_pmax


def run_coarse_infer(
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt: dict[str, Any] | None = None,
    source_model_id: str | None = None,
    iter_id: int | None = None,
    repo_root: Path | None = None,
) -> Path:
    typed = load_coarse_infer_config(cfg)
    if ckpt is None:
        msg = (
            'global-anchor coarse inference requires checkpoint metadata validation; '
            "pass ckpt with coarse_input_mode='global_anchor_resize'"
        )
        raise ValueError(msg)
    validate_checkpoint_for_global_anchor_infer(ckpt, model_sig=typed.model_sig)
    if typed.paths.fb_files is not None:
        msg = 'coarse infer expects raw-only input; omit paths.fb_files'
        raise ValueError(msg)
    if len(typed.paths.segy_files) != 1:
        msg = 'coarse infer expects exactly one paths.segy_files entry'
        raise ValueError(msg)
    if typed.dataset.waveform_mode == 'mmap' and typed.infer.num_workers != 0:
        msg = 'dataset.waveform_mode="mmap" requires infer.num_workers=0'
        raise ValueError(msg)

    plan = build_plan(
        sigma_ms=10.0,
        time_ref_sec=typed.norm_refs.time_ref_sec,
        offset_ref_m=typed.norm_refs.offset_ref_m,
    )
    input_plan = InputOnlyPlan.from_build_plan(plan, include_label_ops=False)
    dataset = build_raw_infer_dataset(
        segy_files=list(typed.paths.segy_files),
        plan=input_plan,
        trace_len=typed.transform.trace_len,
        time_len=typed.transform.time_len,
        standardize_eps=typed.transform.standardize_eps,
        gap_ratio=typed.trace_anchor.gap_ratio,
        min_gap_m=typed.trace_anchor.min_gap_m,
        primary_keys=typed.dataset.primary_keys,
        waveform_mode=typed.dataset.waveform_mode,
        segy_endian=typed.dataset.infer_endian,
        use_header_cache=typed.dataset.use_header_cache,
    )

    loader = DataLoader(
        dataset,
        batch_size=typed.infer.batch_size,
        shuffle=False,
        num_workers=typed.infer.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_input_meta_list,
    )

    try:
        info = dataset.file_infos[0]
        n_traces = int(info['n_traces'])
        n_samples_orig = int(info['n_samples'])
        coarse_pick_i = np.full((n_traces,), -1, dtype=np.int32)
        coarse_pmax = np.full((n_traces,), np.nan, dtype=np.float32)
        counts = np.zeros((n_traces,), dtype=np.int32)

        non_blocking = bool(device.type == 'cuda')
        amp_enabled = bool(typed.infer.amp and device.type == 'cuda')
        model.eval()

        with torch.inference_mode():
            for x_bchw, metas in loader:
                expected_input_shape = (
                    int(x_bchw.shape[0]),
                    3,
                    typed.transform.trace_len,
                    typed.transform.time_len,
                )
                if tuple(x_bchw.shape) != expected_input_shape:
                    msg = (
                        'global-anchor coarse infer input must have shape '
                        f'{expected_input_shape}, got {tuple(x_bchw.shape)}'
                    )
                    raise ValueError(msg)
                x_bchw = x_bchw.to(device=device, non_blocking=non_blocking)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(x_bchw)
                if not isinstance(logits, torch.Tensor):
                    msg = 'model must return torch.Tensor'
                    raise TypeError(msg)
                expected_logits_shape = (
                    int(x_bchw.shape[0]),
                    1,
                    typed.transform.trace_len,
                    typed.transform.time_len,
                )
                if tuple(logits.shape) != expected_logits_shape:
                    msg = (
                        'global-anchor coarse infer logits must have shape '
                        f'{expected_logits_shape}, got {tuple(logits.shape)}'
                    )
                    raise ValueError(msg)

                logits_bhw = (
                    logits[:, 0, :, :]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                for batch_idx, meta in enumerate(metas):
                    anchor_pick_j, anchor_pmax = _extract_anchor_predictions(
                        logits_bhw[batch_idx]
                    )
                    trace_valid = np.asarray(meta['trace_valid'], dtype=np.bool_)
                    if trace_valid.shape != anchor_pick_j.shape:
                        msg = (
                            'trace_valid and anchor predictions must have the same '
                            f'shape, got {trace_valid.shape} and {anchor_pick_j.shape}'
                        )
                        raise ValueError(msg)
                    anchor_pick_j_validated = np.full(
                        anchor_pick_j.shape,
                        -1,
                        dtype=np.int64,
                    )
                    anchor_pick_j_validated[trace_valid] = anchor_pick_j[trace_valid]
                    anchor_pick_i_raw = project_coarse_indices_to_raw_time(
                        anchor_pick_j_validated,
                        raw_time_len=int(meta['raw_time_len']),
                        coarse_time_len=int(meta['coarse_time_len']),
                        ignore_index=-1,
                    )
                    restored_raw_idx, restored_pick_i, restored_pmax = (
                        restore_anchor_predictions_to_full_traces(
                            raw_trace_indices=np.asarray(
                                meta['raw_trace_indices'],
                                dtype=np.int64,
                            ),
                            anchor_source_pos=np.asarray(
                                meta['anchor_source_pos'],
                                dtype=np.int64,
                            ),
                            trace_valid=trace_valid,
                            segment_id=np.asarray(meta['segment_id'], dtype=np.int64),
                            anchor_bin_start_pos=np.asarray(
                                meta['anchor_bin_start_pos'],
                                dtype=np.int64,
                            ),
                            anchor_bin_stop_pos=np.asarray(
                                meta['anchor_bin_stop_pos'],
                                dtype=np.int64,
                            ),
                            anchor_pick_i_raw=anchor_pick_i_raw,
                            anchor_pmax=anchor_pmax,
                            n_samples_orig=n_samples_orig,
                        )
                    )
                    for raw_idx, pick_i, pmax in zip(
                        restored_raw_idx,
                        restored_pick_i,
                        restored_pmax,
                        strict=True,
                    ):
                        raw_trace_idx = int(raw_idx)
                        if raw_trace_idx < 0 or raw_trace_idx >= n_traces:
                            msg = f'raw trace index out of range: {raw_trace_idx}'
                            raise ValueError(msg)
                        if counts[raw_trace_idx] > 0:
                            msg = (
                                'duplicate coarse prediction for raw trace '
                                f'{raw_trace_idx}'
                            )
                            raise ValueError(msg)
                        coarse_pick_i[raw_trace_idx] = int(pick_i)
                        coarse_pmax[raw_trace_idx] = np.float32(pmax)
                        counts[raw_trace_idx] += 1

        if np.any(counts <= 0):
            msg = 'some raw traces were not covered by global-anchor coarse inference'
            raise RuntimeError(msg)

        coarse_pick_i = (
            np.rint(coarse_pick_i).clip(0, n_samples_orig - 1).astype(np.int32)
        )
        dt_sec = float(info['dt_sec'])
        coarse_pick_t_sec = coarse_pick_i.astype(np.float32) * np.float32(dt_sec)

        out_dir = Path(typed.paths.out_dir).expanduser().resolve()
        stem = Path(typed.paths.segy_files[0]).stem
        out_path = out_dir / f'{stem}.coarse.npz'
        return save_coarse_npz(
            out_path,
            dt_sec=dt_sec,
            n_samples_orig=n_samples_orig,
            n_traces=n_traces,
            ffid_values=np.asarray(info['ffid_values'], dtype=np.int32),
            chno_values=np.asarray(info['chno_values'], dtype=np.int32),
            offsets_m=np.asarray(info['offsets'], dtype=np.float32),
            trace_indices=np.arange(n_traces, dtype=np.int64),
            coarse_pick_i=coarse_pick_i,
            coarse_pick_t_sec=coarse_pick_t_sec,
            coarse_pmax=coarse_pmax,
            coarse_prob_summary=coarse_pmax.copy(),
            lineage=build_lineage_payload(
                cfg,
                repo_root=repo_root,
                source_model_id=source_model_id,
                iter_id=iter_id,
            ),
        )
    finally:
        dataset.close()
