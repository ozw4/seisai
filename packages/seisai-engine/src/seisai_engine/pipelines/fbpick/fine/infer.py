from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from seisai_dataset import InputOnlyPlan

from seisai_engine.pipelines.fbpick.common import (
    build_fbpick_final_payload,
    build_lineage_payload,
    load_coarse_npz,
    load_robust_npz,
    save_fbpick_final_npz,
    validate_fine_result_payload,
)

from .build_dataset import build_raw_infer_dataset, collate_input_meta_list
from .build_plan import build_plan
from .config import FineInferConfig, load_fine_infer_config

__all__ = ['run_fine_infer', 'run_fine_local_infer']


def _softmax_last_axis(logits_hw: np.ndarray) -> np.ndarray:
    shifted = logits_hw - logits_hw.max(axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    denom = exp.sum(axis=-1, keepdims=True)
    return exp / denom


def _set_or_validate_int(
    store: np.ndarray,
    *,
    name: str,
    raw_trace_idx: int,
    value: int,
    missing_value: int,
) -> None:
    current = int(store[raw_trace_idx])
    if current == int(missing_value):
        store[raw_trace_idx] = int(value)
        return
    if current != int(value):
        msg = (
            f'inconsistent {name} for raw trace {raw_trace_idx}: '
            f'{current} != {int(value)}'
        )
        raise ValueError(msg)


def _validate_fine_infer_inputs(typed: FineInferConfig) -> None:
    if typed.paths.fb_files is not None:
        msg = 'fine infer expects raw-only input; omit paths.fb_files'
        raise ValueError(msg)
    if typed.paths.robust_npz_files is None:
        msg = 'fine infer requires paths.robust_npz_files'
        raise ValueError(msg)
    if len(typed.paths.segy_files) != 1:
        msg = 'fine infer expects exactly one paths.segy_files entry'
        raise ValueError(msg)
    if len(typed.paths.robust_npz_files) != 1:
        msg = 'fine infer expects exactly one paths.robust_npz_files entry'
        raise ValueError(msg)
    if typed.dataset.waveform_mode == 'mmap' and typed.infer.num_workers != 0:
        msg = 'dataset.waveform_mode="mmap" requires infer.num_workers=0'
        raise ValueError(msg)


def _run_fine_local_infer_impl(
    *,
    model: torch.nn.Module,
    typed: FineInferConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    plan = build_plan(
        sigma_ms=3.0,
        sigma_samples_min=1.5,
        sigma_samples_max=12.0,
    )
    input_plan = InputOnlyPlan.from_build_plan(plan, include_label_ops=False)
    dataset = build_raw_infer_dataset(
        segy_files=list(typed.paths.segy_files),
        robust_npz_files=list(typed.paths.robust_npz_files),
        plan=input_plan,
        trace_len=typed.transform.trace_len,
        overlap_h=typed.infer.overlap_h,
        time_len=typed.transform.time_len,
        center_index=typed.transform.center_index,
        standardize_eps=typed.transform.standardize_eps,
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
        dt_sec = float(info['dt_sec'])
        time_len = int(typed.transform.time_len)

        logits_sum = np.zeros((n_traces, time_len), dtype=np.float32)
        counts = np.zeros((n_traces,), dtype=np.int32)
        window_start_full = np.full((n_traces,), np.iinfo(np.int32).min, dtype=np.int32)
        window_end_full = np.full((n_traces,), np.iinfo(np.int32).min, dtype=np.int32)
        center_raw_full = np.full((n_traces,), np.iinfo(np.int32).min, dtype=np.int32)

        non_blocking = bool(device.type == 'cuda')
        amp_enabled = bool(typed.infer.amp and device.type == 'cuda')
        model.eval()

        with torch.inference_mode():
            for x_bchw, metas in loader:
                x_bchw = x_bchw.to(device=device, non_blocking=non_blocking)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(x_bchw)
                if not isinstance(logits, torch.Tensor):
                    msg = 'model must return torch.Tensor'
                    raise TypeError(msg)
                if logits.ndim != 4:
                    msg = f'model output must be (B,C,H,W), got {tuple(logits.shape)}'
                    raise ValueError(msg)
                if int(logits.shape[1]) != 1:
                    msg = f'fine infer model output channel must be 1, got {int(logits.shape[1])}'
                    raise ValueError(msg)
                if int(logits.shape[2]) != typed.transform.trace_len:
                    msg = 'fine infer model output H must match transform.trace_len'
                    raise ValueError(msg)
                if int(logits.shape[3]) != time_len:
                    msg = 'fine infer model output W must match transform.time_len'
                    raise ValueError(msg)

                logits_bhw = (
                    logits[:, 0, :, :].detach().cpu().numpy().astype(np.float32, copy=False)
                )
                for batch_idx, meta in enumerate(metas):
                    raw_idx_global = np.asarray(meta['raw_idx_global'], dtype=np.int64)
                    trace_valid = np.asarray(meta['trace_valid'], dtype=np.bool_)
                    window_start_i = np.asarray(meta['window_start_i'], dtype=np.int32)
                    window_end_i = np.asarray(meta['window_end_i'], dtype=np.int32)
                    center_raw_i = np.asarray(meta['center_raw_i'], dtype=np.int32)
                    if raw_idx_global.shape != trace_valid.shape:
                        msg = 'raw_idx_global and trace_valid must have the same shape'
                        raise ValueError(msg)
                    if window_start_i.shape != trace_valid.shape:
                        msg = 'window_start_i and trace_valid must have the same shape'
                        raise ValueError(msg)
                    if window_end_i.shape != trace_valid.shape:
                        msg = 'window_end_i and trace_valid must have the same shape'
                        raise ValueError(msg)
                    if center_raw_i.shape != trace_valid.shape:
                        msg = 'center_raw_i and trace_valid must have the same shape'
                        raise ValueError(msg)

                    for row_idx, is_valid in enumerate(trace_valid.tolist()):
                        if not bool(is_valid):
                            continue
                        raw_trace_idx = int(raw_idx_global[row_idx])
                        if raw_trace_idx < 0 or raw_trace_idx >= n_traces:
                            msg = f'raw trace index out of range: {raw_trace_idx}'
                            raise ValueError(msg)
                        row_logits = np.asarray(logits_bhw[batch_idx, row_idx], dtype=np.float32)
                        if row_logits.shape != (time_len,):
                            msg = (
                                'fine infer logits row must have shape '
                                f'({time_len},), got {row_logits.shape}'
                            )
                            raise ValueError(msg)
                        logits_sum[raw_trace_idx] += row_logits
                        counts[raw_trace_idx] += 1
                        _set_or_validate_int(
                            window_start_full,
                            name='window_start_i',
                            raw_trace_idx=raw_trace_idx,
                            value=int(window_start_i[row_idx]),
                            missing_value=np.iinfo(np.int32).min,
                        )
                        _set_or_validate_int(
                            window_end_full,
                            name='window_end_i',
                            raw_trace_idx=raw_trace_idx,
                            value=int(window_end_i[row_idx]),
                            missing_value=np.iinfo(np.int32).min,
                        )
                        _set_or_validate_int(
                            center_raw_full,
                            name='center_raw_i',
                            raw_trace_idx=raw_trace_idx,
                            value=int(center_raw_i[row_idx]),
                            missing_value=np.iinfo(np.int32).min,
                        )

        if np.any(counts <= 0):
            msg = 'some raw traces were not covered by fine inference windows'
            raise RuntimeError(msg)

        avg_logits = logits_sum / counts[:, None].astype(np.float32)
        prob = _softmax_last_axis(avg_logits)
        fine_pick_local_i = prob.argmax(axis=-1).astype(np.int32)
        fine_pick_local_f = fine_pick_local_i.astype(np.float32)
        fine_pmax = prob.max(axis=-1).astype(np.float32)

        final_pick_f = window_start_full.astype(np.float32) + fine_pick_local_f
        final_pick_i = (
            window_start_full.astype(np.int64) + fine_pick_local_i.astype(np.int64)
        ).astype(np.int32)
        final_pick_t_sec = final_pick_f.astype(np.float32) * np.float32(dt_sec)
        final_conf = fine_pmax.copy()

        result = {
            'dt_sec': np.asarray(dt_sec, dtype=np.float32),
            'n_samples_orig': np.asarray(n_samples_orig, dtype=np.int32),
            'n_traces': np.asarray(n_traces, dtype=np.int32),
            'trace_indices': np.arange(n_traces, dtype=np.int64),
            'center_raw_i': center_raw_full.astype(np.int32, copy=False),
            'fine_pick_local_i': fine_pick_local_i,
            'fine_pick_local_f': fine_pick_local_f,
            'fine_pmax': fine_pmax,
            'final_pick_i': final_pick_i,
            'final_pick_f': final_pick_f.astype(np.float32, copy=False),
            'final_pick_t_sec': final_pick_t_sec.astype(np.float32, copy=False),
            'final_conf': final_conf,
            'window_start_i': window_start_full.astype(np.int32, copy=False),
            'window_end_i': window_end_full.astype(np.int32, copy=False),
        }
        validate_fine_result_payload(result)
        return result
    finally:
        dataset.close()


def run_fine_local_infer(
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    typed = load_fine_infer_config(cfg)
    _validate_fine_infer_inputs(typed)
    return _run_fine_local_infer_impl(model=model, typed=typed, device=device)


def _derive_coarse_npz_path_from_robust(robust_npz_path: str | Path) -> Path:
    path = Path(robust_npz_path).expanduser().resolve()
    if not path.name.endswith('.robust.npz'):
        msg = f'expected *.robust.npz input, got {path.name}'
        raise ValueError(msg)
    stem = path.name[: -len('.robust.npz')]
    return path.with_name(f'{stem}.coarse.npz')


def _derive_final_npz_path(*, segy_path: str | Path, out_dir: str | Path) -> Path:
    return Path(out_dir).expanduser().resolve() / f'{Path(segy_path).stem}.fbpick_final.npz'


def run_fine_infer(
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    coarse_payload: dict[str, np.ndarray] | None = None,
    robust_payload: dict[str, np.ndarray] | None = None,
    source_model_id: str | None = None,
    iter_id: int | str | None = None,
    repo_root: Path = Path('/workspace'),
    save_output: bool = True,
) -> dict[str, np.ndarray]:
    typed = load_fine_infer_config(cfg)
    _validate_fine_infer_inputs(typed)

    fine_payload = _run_fine_local_infer_impl(model=model, typed=typed, device=device)

    if robust_payload is None:
        robust_payload = load_robust_npz(typed.paths.robust_npz_files[0])
    if coarse_payload is None:
        coarse_payload = load_coarse_npz(
            _derive_coarse_npz_path_from_robust(typed.paths.robust_npz_files[0])
        )

    final_payload = build_fbpick_final_payload(
        coarse_payload=coarse_payload,
        robust_payload=robust_payload,
        fine_payload=fine_payload,
        high_conf_threshold=typed.infer.high_conf_threshold,
        lineage=build_lineage_payload(
            cfg,
            repo_root=repo_root,
            source_model_id=source_model_id,
            iter_id=iter_id,
        ),
    )

    if save_output and typed.paths.out_dir is not None:
        save_fbpick_final_npz(
            _derive_final_npz_path(
                segy_path=typed.paths.segy_files[0],
                out_dir=typed.paths.out_dir,
            ),
            **final_payload,
        )
    return final_payload
