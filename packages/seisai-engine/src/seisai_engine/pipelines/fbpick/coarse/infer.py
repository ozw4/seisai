from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from seisai_dataset.config import LoaderConfig
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
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
    COARSE_INPUT_CHANNELS,
    COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE,
    CoarseInferConfig,
    CoarseQCCfg,
    COARSE_TIME_LEN,
    COARSE_TRACE_LEN,
    load_coarse_infer_config,
)
from .qc import select_display_indices, write_global_anchor_coarse_qc
from .time_axis import project_coarse_indices_to_raw_time

__all__ = [
    'restore_anchor_predictions_to_full_traces',
    'run_coarse_infer',
    'validate_coarse_npz_payload',
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
        'coarse_input_channels': list(COARSE_INPUT_CHANNELS),
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
    anchor_pick_i_raw: np.ndarray,
    anchor_pmax: np.ndarray,
    trace_valid: np.ndarray,
    anchor_source_pos: np.ndarray,
    segment_id: np.ndarray,
    segments,
    n_traces: int,
    n_samples_orig: int,
) -> tuple[np.ndarray, np.ndarray]:
    if segments is None:
        msg = 'segment metadata is required to restore global-anchor coarse predictions'
        raise ValueError(msg)

    n_trace = int(n_traces)
    n_samples = int(n_samples_orig)
    if n_trace <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    valid = np.asarray(trace_valid, dtype=np.bool_)
    anchor_pos = np.asarray(anchor_source_pos, dtype=np.int64)
    seg_id = np.asarray(segment_id, dtype=np.int64)
    pick_raw = np.asarray(anchor_pick_i_raw, dtype=np.float32)
    pmax = np.asarray(anchor_pmax, dtype=np.float32)

    if valid.ndim != 1:
        msg = f'trace_valid must be 1D, got shape {valid.shape}'
        raise ValueError(msg)
    h = int(valid.shape[0])
    for name, arr in (
        ('anchor_pick_i_raw', pick_raw),
        ('anchor_pmax', pmax),
        ('anchor_source_pos', anchor_pos),
        ('segment_id', seg_id),
    ):
        if arr.shape != (h,):
            msg = f'{name} must have shape ({h},), got {arr.shape}'
            raise ValueError(msg)

    segment_spans: list[tuple[int, int, int]] = []
    covered = np.zeros((n_trace,), dtype=np.bool_)
    seen_segment_ids: set[int] = set()
    for segment in tuple(segments):
        try:
            sid = int(segment.segment_id)
            start = int(segment.start_pos)
            stop = int(segment.stop_pos)
        except AttributeError as exc:
            msg = 'segment metadata is required to restore global-anchor coarse predictions'
            raise ValueError(msg) from exc
        if sid < 0:
            msg = f'invalid segment_id: {sid}'
            raise ValueError(msg)
        if sid in seen_segment_ids:
            msg = f'duplicate segment_id: {sid}'
            raise ValueError(msg)
        seen_segment_ids.add(sid)
        if start < 0 or stop <= start or stop > n_trace:
            msg = f'invalid segment span for segment {sid}: [{start}, {stop})'
            raise ValueError(msg)
        if np.any(covered[start:stop]):
            msg = f'overlapping segment span for segment {sid}: [{start}, {stop})'
            raise ValueError(msg)
        covered[start:stop] = True
        segment_spans.append((sid, start, stop))

    if not segment_spans:
        msg = 'segment metadata is required to restore global-anchor coarse predictions'
        raise ValueError(msg)
    if not np.all(covered):
        msg = 'segment metadata must cover every trace exactly once'
        raise ValueError(msg)

    valid_anchor_mask = valid
    if np.any(valid_anchor_mask):
        if np.any(anchor_pos[valid_anchor_mask] < 0) or np.any(
            anchor_pos[valid_anchor_mask] >= n_trace
        ):
            msg = 'valid anchor_source_pos must lie in [0, n_traces)'
            raise ValueError(msg)
        if np.any(seg_id[valid_anchor_mask] < 0):
            msg = 'valid segment_id must be non-negative'
            raise ValueError(msg)
        if not np.all(np.isfinite(pick_raw[valid_anchor_mask])):
            msg = 'valid anchor_pick_i_raw must be finite'
            raise ValueError(msg)
        if np.any(pick_raw[valid_anchor_mask] < 0) or np.any(
            pick_raw[valid_anchor_mask] > n_samples - 1
        ):
            msg = 'valid anchor_pick_i_raw must lie in [0, n_samples_orig)'
            raise ValueError(msg)
        if not np.all(np.isfinite(pmax[valid_anchor_mask])):
            msg = 'valid anchor_pmax must be finite'
            raise ValueError(msg)

    coarse_pick_f = np.full((n_trace,), np.nan, dtype=np.float64)
    coarse_pmax = np.full((n_trace,), np.nan, dtype=np.float64)

    for sid, start, stop in segment_spans:
        mask = valid_anchor_mask & (seg_id == sid)
        if not np.any(mask):
            msg = f'no valid anchors for segment {sid}'
            raise ValueError(msg)

        anchor_positions = anchor_pos[mask].astype(np.float64, copy=False)
        if np.any(anchor_positions < start) or np.any(anchor_positions >= stop):
            msg = f'anchor_source_pos for segment {sid} lies outside segment span'
            raise ValueError(msg)

        order = np.argsort(anchor_positions, kind='mergesort')
        anchor_positions = anchor_positions[order]
        if anchor_positions.size > 1 and np.any(np.diff(anchor_positions) <= 0.0):
            msg = f'anchor_source_pos for segment {sid} must be unique'
            raise ValueError(msg)

        pick_values = pick_raw[mask][order].astype(np.float64, copy=False)
        pmax_values = pmax[mask][order].astype(np.float64, copy=False)
        raw_positions = np.arange(start, stop, dtype=np.float64)

        if anchor_positions.size == 1:
            coarse_pick_f[start:stop] = float(pick_values[0])
            coarse_pmax[start:stop] = float(pmax_values[0])
        else:
            coarse_pick_f[start:stop] = np.interp(
                raw_positions,
                anchor_positions,
                pick_values,
            )
            coarse_pmax[start:stop] = np.interp(
                raw_positions,
                anchor_positions,
                pmax_values,
            )

    if np.any(~np.isfinite(coarse_pick_f)):
        msg = 'restored coarse_pick_i contains missing values'
        raise RuntimeError(msg)
    if np.any(~np.isfinite(coarse_pmax)):
        msg = 'restored coarse_pmax contains missing values'
        raise RuntimeError(msg)

    coarse_pick_i = np.rint(coarse_pick_f).clip(0, n_samples - 1).astype(np.int32)
    return coarse_pick_i, coarse_pmax.astype(np.float32, copy=False)


def validate_coarse_npz_payload(
    *,
    coarse_pick_i: np.ndarray,
    coarse_pick_t_sec: np.ndarray,
    coarse_pmax: np.ndarray,
    dt_sec: float,
    n_traces: int,
    n_samples_orig: int,
) -> None:
    n_trace = int(n_traces)
    n_samples = int(n_samples_orig)
    if n_trace <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    pick_i = np.asarray(coarse_pick_i)
    pick_t_sec = np.asarray(coarse_pick_t_sec)
    pmax = np.asarray(coarse_pmax)
    expected_shape = (n_trace,)

    if pick_i.shape != expected_shape:
        msg = f'coarse_pick_i shape {pick_i.shape} != {expected_shape}'
        raise ValueError(msg)
    if pick_t_sec.shape != expected_shape:
        msg = f'coarse_pick_t_sec shape {pick_t_sec.shape} != {expected_shape}'
        raise ValueError(msg)
    if pmax.shape != expected_shape:
        msg = f'coarse_pmax shape {pmax.shape} != {expected_shape}'
        raise ValueError(msg)
    if not np.issubdtype(pick_i.dtype, np.integer):
        msg = 'coarse_pick_i dtype must be integer'
        raise TypeError(msg)
    if not np.issubdtype(pick_t_sec.dtype, np.floating):
        msg = 'coarse_pick_t_sec dtype must be float'
        raise TypeError(msg)
    if not np.issubdtype(pmax.dtype, np.floating):
        msg = 'coarse_pmax dtype must be float'
        raise TypeError(msg)
    if np.any(pick_i < 0) or np.any(pick_i >= n_samples):
        msg = 'coarse_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)
    if not np.all(np.isfinite(pick_t_sec)):
        msg = 'coarse_pick_t_sec must be finite'
        raise ValueError(msg)
    if not np.all(np.isfinite(pmax)):
        msg = 'coarse_pmax must be finite'
        raise ValueError(msg)

    dt = np.float32(dt_sec)
    if not np.isfinite(dt):
        msg = 'dt_sec must be finite'
        raise ValueError(msg)
    expected_t_sec = pick_i.astype(np.float32) * dt
    if not np.allclose(pick_t_sec, expected_t_sec, rtol=0.0, atol=1.0e-6):
        msg = 'coarse_pick_t_sec must match coarse_pick_i * dt_sec'
        raise ValueError(msg)


def _make_qc_gather_id(meta: dict[str, Any]) -> str:
    source = Path(str(meta.get('source_file') or meta.get('file_path') or 'gather'))
    parent = source.parent.name
    prefix = f'{parent}__{source.stem}' if parent else source.stem
    key_name = str(meta.get('key_name', 'key'))
    primary_value = meta.get('primary_value', meta.get('primary_unique', '0'))
    return f'{prefix}__{key_name}-{primary_value}'


def _load_qc_label_picks(
    typed: CoarseInferConfig,
    *,
    n_traces: int,
) -> np.ndarray | None:
    if not typed.qc.enabled or not typed.qc.plot_error_if_labels_available:
        return None
    if typed.paths.infer_fb_files is None:
        return None
    if len(typed.paths.infer_fb_files) != 1:
        msg = 'qc labels require exactly one paths.infer_fb_files entry'
        raise ValueError(msg)
    labels = np.asarray(np.load(typed.paths.infer_fb_files[0]), dtype=np.int64)
    if labels.ndim != 1 or int(labels.shape[0]) != int(n_traces):
        msg = (
            'qc labels must be a 1D first-break array with length n_traces, '
            f'got shape={labels.shape}, n_traces={int(n_traces)}'
        )
        raise ValueError(msg)
    return labels


def _load_qc_raw_wave_display(
    *,
    info: dict[str, Any],
    raw_trace_indices: np.ndarray,
    qc: CoarseQCCfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace_positions = select_display_indices(
        int(raw_trace_indices.size),
        int(qc.max_display_traces),
    )
    raw_indices_display = np.asarray(raw_trace_indices, dtype=np.int64)[
        trace_positions
    ]
    loader = TraceSubsetLoader(LoaderConfig(pad_traces_to=1))
    wave = loader.load_traces(info['mmap'], raw_indices_display)
    sample_indices = select_display_indices(
        int(wave.shape[1]),
        int(qc.max_display_samples),
    )
    return (
        np.ascontiguousarray(wave[:, sample_indices], dtype=np.float32),
        trace_positions.astype(np.int64, copy=False),
        sample_indices.astype(np.int64, copy=False),
    )


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

    out_dir = Path(typed.paths.out_dir).expanduser().resolve()
    qc_out_dir = out_dir / typed.qc.out_subdir
    qc_count = 0

    try:
        info = dataset.file_infos[0]
        n_traces = int(info['n_traces'])
        n_samples_orig = int(info['n_samples'])
        coarse_pick_i = np.full((n_traces,), -1, dtype=np.int32)
        coarse_pmax = np.full((n_traces,), np.nan, dtype=np.float32)
        counts = np.zeros((n_traces,), dtype=np.int32)
        qc_label_pick_i = _load_qc_label_picks(typed, n_traces=n_traces)

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
                qc_input_bchw = None
                if typed.qc.enabled and qc_count < int(typed.qc.max_gathers):
                    qc_input_bchw = x_bchw.detach().cpu()
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
                    raw_trace_indices = np.asarray(
                        meta['raw_trace_indices'],
                        dtype=np.int64,
                    )
                    if raw_trace_indices.ndim != 1 or raw_trace_indices.size == 0:
                        msg = 'raw_trace_indices must be a non-empty 1D array'
                        raise ValueError(msg)
                    restored_pick_i, restored_pmax = (
                        restore_anchor_predictions_to_full_traces(
                            anchor_pick_i_raw=anchor_pick_i_raw,
                            anchor_pmax=anchor_pmax,
                            trace_valid=trace_valid,
                            anchor_source_pos=np.asarray(
                                meta['anchor_source_pos'],
                                dtype=np.int64,
                            ),
                            segment_id=np.asarray(meta['segment_id'], dtype=np.int64),
                            segments=meta.get('segments'),
                            n_traces=int(raw_trace_indices.size),
                            n_samples_orig=n_samples_orig,
                        )
                    )
                    if typed.qc.enabled and qc_count < int(typed.qc.max_gathers):
                        if qc_input_bchw is None:
                            msg = 'qc input tensor was not captured'
                            raise RuntimeError(msg)
                        raw_wave_hw = None
                        raw_trace_positions = None
                        raw_sample_indices = None
                        if typed.qc.plot_original_gather:
                            file_idx = int(meta.get('file_idx', 0))
                            raw_wave_hw, raw_trace_positions, raw_sample_indices = (
                                _load_qc_raw_wave_display(
                                    info=dataset.file_infos[file_idx],
                                    raw_trace_indices=raw_trace_indices,
                                    qc=typed.qc,
                                )
                            )
                        fb_pick_i = None
                        if qc_label_pick_i is not None:
                            fb_pick_i = qc_label_pick_i[raw_trace_indices]
                        write_global_anchor_coarse_qc(
                            out_dir=qc_out_dir,
                            gather_id=_make_qc_gather_id(meta),
                            input_waveform_hw=qc_input_bchw[batch_idx, 0]
                            .numpy()
                            .astype(np.float32, copy=False),
                            anchor_pick_j=anchor_pick_j_validated,
                            anchor_pmax=anchor_pmax,
                            trace_valid=trace_valid,
                            segment_id=np.asarray(meta['segment_id'], dtype=np.int64),
                            segments=meta.get('segments'),
                            coarse_pick_i=restored_pick_i,
                            coarse_pmax=restored_pmax,
                            raw_wave_hw=raw_wave_hw,
                            raw_trace_positions=raw_trace_positions,
                            raw_sample_indices=raw_sample_indices,
                            fb_pick_i=fb_pick_i,
                            n_samples_orig=int(meta['n_samples_orig']),
                            dt_sec=float(meta['dt_sec']),
                            cfg=typed.qc,
                        )
                        qc_count += 1
                    for raw_idx, pick_i, pmax in zip(
                        raw_trace_indices,
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
        validate_coarse_npz_payload(
            coarse_pick_i=coarse_pick_i,
            coarse_pick_t_sec=coarse_pick_t_sec,
            coarse_pmax=coarse_pmax,
            dt_sec=dt_sec,
            n_traces=n_traces,
            n_samples_orig=n_samples_orig,
        )

        stem = Path(typed.paths.segy_files[0]).stem
        out_path = out_dir / f'{stem}.coarse.npz'
        geometry_valid_mask = np.asarray(info['geometry_valid_mask'], dtype=np.bool_)
        offset_signed_geom_m = None
        if info.get('offset_signed_geom_m') is not None:
            candidate_signed = np.asarray(
                info['offset_signed_geom_m'],
                dtype=np.float32,
            )
            if candidate_signed.shape == (n_traces,) and np.all(
                np.isfinite(candidate_signed[geometry_valid_mask])
            ):
                offset_signed_geom_m = candidate_signed
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
            source_x_m=np.asarray(info['source_x_m'], dtype=np.float32),
            source_y_m=np.asarray(info['source_y_m'], dtype=np.float32),
            receiver_x_m=np.asarray(info['receiver_x_m'], dtype=np.float32),
            receiver_y_m=np.asarray(info['receiver_y_m'], dtype=np.float32),
            offset_abs_geom_m=np.asarray(info['offset_abs_geom_m'], dtype=np.float32),
            geometry_valid_mask=geometry_valid_mask,
            offset_signed_geom_m=offset_signed_geom_m,
        )
    finally:
        dataset.close()
