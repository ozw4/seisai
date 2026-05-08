from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from seisai_dataset import InputOnlyPlan
from seisai_utils.listfiles import expand_cfg_listfiles

from seisai_engine.infer.segy2segy_cli_common import (
    build_merged_cfg,
    load_ckpt_cfg_for_merge,
    resolve_ckpt_path,
    select_state_dict,
)
from seisai_engine.pipelines.common import (
    load_cfg_with_base_dir,
    load_checkpoint,
    resolve_cfg_paths,
    resolve_device,
)
from seisai_engine.pipelines.fbpick.common import (
    build_fbpick_final_payload,
    build_lineage_payload,
    iter_qc_gathers,
    load_coarse_npz,
    load_robust_npz,
    save_fbpick_final_npz,
    validate_fine_result_payload,
)

from .build_dataset import build_raw_infer_dataset, collate_input_meta_list
from .build_model import build_model
from .build_plan import build_plan
from .config import FineInferConfig, FineViewerCfg, load_fine_infer_config

__all__ = [
    'infer_coarse_npz_path_from_robust_npz_path',
    'main',
    'require_existing_coarse_npz_path',
    'resolve_fine_coarse_npz_path',
    'run_fine_infer',
    'run_fine_local_infer',
    'run_infer_and_write',
]


DEFAULT_CONFIG_PATH = Path('examples/config_infer_fbpick_fine.yaml')
_SAFE_OVERRIDE_PATHS = frozenset(
    {
        'paths.segy_files',
        'paths.robust_npz_files',
        'paths.coarse_npz_files',
        'paths.out_dir',
        'dataset.use_header_cache',
        'dataset.verbose',
        'dataset.progress',
        'dataset.waveform_mode',
        'dataset.infer_endian',
        'transform.standardize_eps',
        'infer.ckpt_path',
        'infer.device',
        'infer.batch_size',
        'infer.num_workers',
        'infer.overlap_h',
        'infer.amp',
        'infer.use_tqdm',
        'infer.high_conf_threshold',
        'infer.source_model_id',
        'infer.iter_id',
        'infer.allow_unsafe_override',
        'window_center.npz_key',
        'window_center.fallback_npz_key',
        'viewer.enabled',
        'viewer.save_overview_png',
        'viewer.save_gather_png',
        'viewer.max_gathers_per_file',
        'viewer.skip_gather_keys',
        'viewer.max_traces_per_gather',
        'viewer.waveform_norm',
        'viewer.dpi',
        'viewer.clip_percentile',
    }
)


def _default_cfg() -> dict[str, Any]:
    return {
        'paths': {
            'segy_files': [],
            'robust_npz_files': [],
            'coarse_npz_files': None,
            'out_dir': './_fbpick_fine_infer_out',
        },
        'dataset': {
            'use_header_cache': True,
            'verbose': True,
            'progress': True,
            'primary_keys': ['ffid'],
            'waveform_mode': 'eager',
            'train_endian': 'big',
            'infer_endian': 'big',
        },
        'transform': {
            'trace_len': 128,
            'time_len': 256,
            'center_index': 128,
            'standardize_eps': 1.0e-8,
        },
        'window_center': {
            'npz_key': 'robust_pick_i',
            'fallback_npz_key': None,
        },
        'infer': {
            'ckpt_path': '',
            'device': 'auto',
            'batch_size': 1,
            'num_workers': 0,
            'overlap_h': 96,
            'amp': False,
            'use_tqdm': False,
            'high_conf_threshold': 0.5,
            'source_model_id': None,
            'iter_id': None,
            'allow_unsafe_override': False,
        },
        'viewer': {
            'enabled': False,
            'save_overview_png': False,
            'save_gather_png': False,
            'max_gathers_per_file': 8,
            'skip_gather_keys': {},
            'max_traces_per_gather': 10000,
            'waveform_norm': 'global',
            'dpi': 150,
            'clip_percentile': 99.0,
        },
        'model': {
            'backbone': 'resnet18',
            'pretrained': False,
            'in_chans': 1,
            'out_chans': 1,
        },
    }


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
    if (
        typed.paths.coarse_npz_files is not None
        and len(typed.paths.coarse_npz_files) != 1
    ):
        msg = 'fine infer expects exactly one paths.coarse_npz_files entry'
        raise ValueError(msg)
    if typed.dataset.waveform_mode == 'mmap' and typed.infer.num_workers != 0:
        msg = 'dataset.waveform_mode="mmap" requires infer.num_workers=0'
        raise ValueError(msg)


def _extract_raw_wave_hw(*, info, n_traces: int, n_samples_orig: int) -> np.ndarray:
    raw_wave_hw = np.ascontiguousarray(np.asarray(info['mmap'], dtype=np.float32))
    if raw_wave_hw.shape != (n_traces, n_samples_orig):
        msg = (
            'raw waveform gather must have shape '
            f'({n_traces}, {n_samples_orig}), got {raw_wave_hw.shape}'
        )
        raise ValueError(msg)
    return raw_wave_hw


def _run_fine_local_infer_impl(
    *,
    model: torch.nn.Module,
    typed: FineInferConfig,
    device: torch.device,
    raw_wave_hw_out: dict[str, np.ndarray] | None = None,
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
        window_center_npz_key=typed.window_center.npz_key,
        window_center_fallback_npz_key=typed.window_center.fallback_npz_key,
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

        if raw_wave_hw_out is not None:
            raw_wave_hw_out['raw_wave_hw'] = _extract_raw_wave_hw(
                info=info,
                n_traces=n_traces,
                n_samples_orig=n_samples_orig,
            )

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


def infer_coarse_npz_path_from_robust_npz_path(robust_npz_path: str | Path) -> Path:
    """Infer the legacy same-directory coarse path for a robust NPZ path."""
    path = Path(robust_npz_path).expanduser().resolve()
    if not path.name.endswith('.robust.npz'):
        msg = f'expected *.robust.npz input, got {path.name}'
        raise ValueError(msg)
    stem = path.name[: -len('.robust.npz')]
    return path.with_name(f'{stem}.coarse.npz')


def resolve_fine_coarse_npz_path(
    *,
    robust_npz_path: str | Path,
    explicit_coarse_npz_path: str | Path | None,
) -> Path:
    """Resolve the coarse path, preferring an explicit fine-infer config path."""
    if explicit_coarse_npz_path is not None:
        return Path(explicit_coarse_npz_path).expanduser().resolve()
    return infer_coarse_npz_path_from_robust_npz_path(robust_npz_path)


def require_existing_coarse_npz_path(
    *,
    coarse_npz_path: str | Path,
    robust_npz_path: str | Path,
    was_explicit: bool,
) -> Path:
    """Return an existing coarse path or raise a fine-infer specific error."""
    path = Path(coarse_npz_path).expanduser().resolve()
    if path.is_file():
        return path

    if was_explicit:
        msg = f'coarse npz file not found: {path}'
        raise FileNotFoundError(msg)

    robust_path = Path(robust_npz_path).expanduser().resolve()
    msg = (
        f'coarse npz file not found: {path}\n'
        f'inferred from robust npz: {robust_path}\n'
        'Set paths.coarse_npz_files explicitly if coarse and robust outputs are '
        'in different directories.'
    )
    raise FileNotFoundError(msg)


def _derive_final_npz_path(*, segy_path: str | Path, out_dir: str | Path) -> Path:
    return Path(out_dir).expanduser().resolve() / f'{Path(segy_path).stem}.fbpick_final.npz'


def _derive_overview_png_path(*, segy_path: str | Path, out_dir: str | Path) -> Path:
    return Path(out_dir).expanduser().resolve() / f'{Path(segy_path).stem}.overview.png'


def _derive_qc_tag(segy_path: str | Path) -> str:
    segy = Path(segy_path)
    parent_name = segy.parent.name
    if parent_name:
        return parent_name + '__' + segy.stem
    return segy.stem


def _derive_fine_qc_dir(*, segy_path: str | Path, out_dir: str | Path) -> Path:
    return (
        Path(out_dir).expanduser().resolve()
        / f'{_derive_qc_tag(segy_path)}.fine_qc'
    )


def _build_fine_qc_info(
    *, segy_path: str | Path, typed: FineInferConfig
) -> dict[str, Any]:
    import segyio
    from seisai_dataset.file_info import build_file_info

    return build_file_info(
        str(segy_path),
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        use_header_cache=bool(typed.dataset.use_header_cache),
        include_centroids=False,
        waveform_mode='mmap',
        segy_endian=str(typed.dataset.infer_endian),
    )


def _close_info(info: dict[str, Any]) -> None:
    segy_obj = info.get('segy_obj')
    if segy_obj is not None:
        segy_obj.close()


def _validate_fine_qc_info_against_payload(
    info: dict[str, Any],
    *,
    final_payload: dict[str, np.ndarray],
) -> None:
    n_traces = int(np.asarray(final_payload['n_traces']).item())
    n_samples_orig = int(np.asarray(final_payload['n_samples_orig']).item())
    dt_sec = float(np.asarray(final_payload['dt_sec']).item())
    if int(info['n_traces']) != n_traces:
        msg = f'QC info n_traces {int(info["n_traces"])} != final n_traces {n_traces}'
        raise ValueError(msg)
    if int(info['n_samples']) != n_samples_orig:
        msg = (
            f'QC info n_samples {int(info["n_samples"])} != '
            f'final n_samples_orig {n_samples_orig}'
        )
        raise ValueError(msg)
    if not np.isclose(float(info['dt_sec']), dt_sec, rtol=0.0, atol=1e-9):
        msg = f'QC info dt_sec {float(info["dt_sec"])} != final dt_sec {dt_sec}'
        raise ValueError(msg)


def _save_fine_gather_qc_pngs(
    *,
    info: dict[str, Any],
    segy_path: str | Path,
    out_dir: str | Path,
    final_payload: dict[str, np.ndarray],
    viewer: FineViewerCfg,
    primary_keys: tuple[str, ...],
    save_png_func=None,
) -> list[Path]:
    max_gathers = int(viewer.max_gathers_per_file)
    if max_gathers <= 0:
        return []

    if save_png_func is None:
        from seisai_engine.viewer.fbpick import save_fbpick_fine_qc_gather_png

        save_png_func = save_fbpick_fine_qc_gather_png

    _validate_fine_qc_info_against_payload(info, final_payload=final_payload)
    out_subdir = _derive_fine_qc_dir(segy_path=segy_path, out_dir=out_dir)
    out_paths: list[Path] = []
    for gather_idx, (primary_key, gather_key, trace_indices) in enumerate(
        iter_qc_gathers(
            info,
            primary_keys=primary_keys,
            max_gathers=max_gathers,
            skip_gather_keys=viewer.skip_gather_keys,
            max_traces_per_gather=viewer.max_traces_per_gather,
            segy_path=segy_path,
        )
    ):
        x_hw = np.stack(
            [
                np.asarray(info['mmap'][int(i)], dtype=np.float32)
                for i in trace_indices
            ],
            axis=0,
        )
        out_png = out_subdir / f'gather_{gather_idx:04d}.png'
        title = f'{Path(segy_path).name} {primary_key}={gather_key}'
        out_paths.append(
            save_png_func(
                out_png,
                raw_wave_hw=x_hw,
                final_payload=final_payload,
                trace_indices=trace_indices,
                title=title,
                dpi=viewer.dpi,
                clip_percentile=viewer.clip_percentile,
                waveform_norm=viewer.waveform_norm,
            )
        )
    if not out_paths:
        print(
            f'No fine gather PNGs written for {segy_path}: '
            'all candidates were skipped'
        )
    return out_paths


def _load_fine_ckpt_cfg_for_merge(
    *,
    infer_cfg_for_ckpt: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    ckpt_cfg = deepcopy(
        load_ckpt_cfg_for_merge(
            infer_cfg_for_ckpt=infer_cfg_for_ckpt,
            base_dir=base_dir,
        )
    )
    ckpt_cfg.pop('paths', None)
    return ckpt_cfg


def _prepare_fine_infer_cfg(cfg: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    prepared = deepcopy(cfg)
    paths = prepared.get('paths')
    if not isinstance(paths, dict):
        msg = 'paths must be dict'
        raise TypeError(msg)
    list_path_keys: list[str] = ['paths.segy_files', 'paths.robust_npz_files']
    if paths.get('coarse_npz_files') is not None:
        list_path_keys.append('paths.coarse_npz_files')
    resolve_cfg_paths(prepared, base_dir, keys=list_path_keys)
    expand_cfg_listfiles(prepared, keys=list_path_keys)
    if paths.get('out_dir') is not None:
        resolve_cfg_paths(prepared, base_dir, keys=['paths.out_dir'])
    return prepared


def _get_infer_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    infer_cfg = cfg.get('infer')
    if not isinstance(infer_cfg, dict):
        msg = 'infer must be dict'
        raise TypeError(msg)
    return infer_cfg


def _resolve_cli_device(cfg: dict[str, Any]) -> torch.device:
    infer_cfg = _get_infer_cfg(cfg)
    device_raw = infer_cfg.get('device', 'auto')
    if device_raw is not None and not isinstance(device_raw, str):
        msg = 'infer.device must be str or null'
        raise TypeError(msg)
    return resolve_device(device_raw)


def _resolve_lineage_args(cfg: dict[str, Any]) -> tuple[str | None, int | str | None]:
    infer_cfg = _get_infer_cfg(cfg)

    source_model_id = infer_cfg.get('source_model_id')
    if source_model_id is not None:
        if not isinstance(source_model_id, str):
            msg = 'infer.source_model_id must be str or null'
            raise TypeError(msg)
        source_model_id = source_model_id.strip() or None

    iter_id = infer_cfg.get('iter_id')
    if isinstance(iter_id, bool) or not isinstance(iter_id, (int, str, type(None))):
        msg = 'infer.iter_id must be int, str, or null'
        raise TypeError(msg)

    return source_model_id, iter_id


def _validate_fine_checkpoint_for_infer(
    ckpt: dict[str, Any],
    *,
    model_sig: dict[str, Any],
) -> None:
    pipeline = ckpt.get('pipeline')
    if pipeline != 'fbpick':
        msg = f'fine infer checkpoint pipeline must be "fbpick", got {pipeline!r}'
        raise ValueError(msg)

    stage = ckpt.get('stage')
    if stage != 'fine':
        msg = f'fine infer checkpoint stage must be "fine", got {stage!r}'
        raise ValueError(msg)

    ckpt_model_sig = ckpt.get('model_sig')
    if not isinstance(ckpt_model_sig, dict):
        msg = 'checkpoint model_sig must be dict'
        raise TypeError(msg)
    for key, value in ckpt_model_sig.items():
        if key in model_sig and model_sig[key] != value:
            msg = f'checkpoint model_sig mismatch for {key}: {value!r} != {model_sig[key]!r}'
            raise ValueError(msg)

    output_ids = ckpt.get('output_ids')
    if output_ids is not None:
        if not isinstance(output_ids, (list, tuple)):
            msg = 'checkpoint output_ids must be list[str] or tuple[str, ...]'
            raise TypeError(msg)
        if list(output_ids) != ['P']:
            msg = f'fine infer checkpoint output_ids must be ["P"], got {output_ids!r}'
            raise ValueError(msg)

    softmax_axis = ckpt.get('softmax_axis')
    if softmax_axis is not None and softmax_axis != 'time':
        msg = (
            'fine infer checkpoint softmax_axis must be "time", '
            f'got {softmax_axis!r}'
        )
        raise ValueError(msg)


def run_infer_and_write(
    *,
    cfg: dict[str, Any],
    base_dir: Path,
) -> Path:
    prepared = _prepare_fine_infer_cfg(cfg, base_dir=base_dir)
    typed = load_fine_infer_config(prepared)
    if typed.paths.out_dir is None:
        msg = 'paths.out_dir is required for fbpick fine infer'
        raise ValueError(msg)

    ckpt_path = resolve_ckpt_path(prepared, base_dir=base_dir)
    device = _resolve_cli_device(prepared)
    source_model_id, iter_id = _resolve_lineage_args(prepared)

    ckpt = load_checkpoint(ckpt_path)
    _validate_fine_checkpoint_for_infer(ckpt, model_sig=typed.model_sig)

    model = build_model(dict(typed.model_sig))
    state_dict, _ = select_state_dict(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    run_fine_infer(
        model=model,
        cfg=prepared,
        device=device,
        source_model_id=source_model_id,
        iter_id=iter_id,
    )
    return _derive_final_npz_path(
        segy_path=typed.paths.segy_files[0],
        out_dir=typed.paths.out_dir,
    )


def run_fine_infer(
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    coarse_payload: dict[str, np.ndarray] | None = None,
    robust_payload: dict[str, np.ndarray] | None = None,
    source_model_id: str | None = None,
    iter_id: int | str | None = None,
    repo_root: Path | None = None,
    save_output: bool = True,
) -> dict[str, np.ndarray]:
    typed = load_fine_infer_config(cfg)
    _validate_fine_infer_inputs(typed)

    save_overview = bool(
        save_output and typed.viewer.enabled and typed.viewer.save_overview_png
    )
    save_gather_png = bool(
        save_output and typed.viewer.enabled and typed.viewer.save_gather_png
    )
    if save_overview and typed.paths.out_dir is None:
        msg = 'viewer overview export requires paths.out_dir'
        raise ValueError(msg)
    if save_gather_png and typed.paths.out_dir is None:
        msg = 'viewer gather QC export requires paths.out_dir'
        raise ValueError(msg)

    raw_wave_hw_out: dict[str, np.ndarray] | None
    if save_overview:
        raw_wave_hw_out = {}
    else:
        raw_wave_hw_out = None

    fine_payload = _run_fine_local_infer_impl(
        model=model,
        typed=typed,
        device=device,
        raw_wave_hw_out=raw_wave_hw_out,
    )

    if robust_payload is None:
        robust_payload = load_robust_npz(typed.paths.robust_npz_files[0])
    if coarse_payload is None:
        explicit_coarse_npz_path = (
            None
            if typed.paths.coarse_npz_files is None
            else typed.paths.coarse_npz_files[0]
        )
        coarse_npz_path = resolve_fine_coarse_npz_path(
            robust_npz_path=typed.paths.robust_npz_files[0],
            explicit_coarse_npz_path=explicit_coarse_npz_path,
        )
        coarse_npz_path = require_existing_coarse_npz_path(
            coarse_npz_path=coarse_npz_path,
            robust_npz_path=typed.paths.robust_npz_files[0],
            was_explicit=explicit_coarse_npz_path is not None,
        )
        coarse_payload = load_coarse_npz(coarse_npz_path)

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

    if save_overview:
        if raw_wave_hw_out is None or 'raw_wave_hw' not in raw_wave_hw_out:
            msg = 'raw waveform gather is required for overview export'
            raise RuntimeError(msg)
        from seisai_engine.viewer.fbpick import save_fbpick_overview_png

        save_fbpick_overview_png(
            _derive_overview_png_path(
                segy_path=typed.paths.segy_files[0],
                out_dir=typed.paths.out_dir,
            ),
            raw_wave_hw=raw_wave_hw_out['raw_wave_hw'],
            final_payload=final_payload,
            title=Path(typed.paths.segy_files[0]).stem,
            dpi=typed.viewer.dpi,
            clip_percentile=typed.viewer.clip_percentile,
        )
    if save_gather_png:
        info = _build_fine_qc_info(
            segy_path=typed.paths.segy_files[0],
            typed=typed,
        )
        try:
            _save_fine_gather_qc_pngs(
                info=info,
                segy_path=typed.paths.segy_files[0],
                out_dir=typed.paths.out_dir,
                final_payload=final_payload,
                viewer=typed.viewer,
                primary_keys=typed.dataset.primary_keys,
            )
        finally:
            _close_info(info)
    return final_payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args, unknown = parser.parse_known_args(argv)

    infer_yaml_cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    merged_cfg = build_merged_cfg(
        infer_yaml_cfg=infer_yaml_cfg,
        base_dir=base_dir,
        unknown_overrides=unknown,
        default_cfg=_default_cfg(),
        safe_paths=_SAFE_OVERRIDE_PATHS,
        ckpt_cfg_loader=_load_fine_ckpt_cfg_for_merge,
    )
    out_path = run_infer_and_write(cfg=merged_cfg, base_dir=base_dir)
    print(str(out_path))


if __name__ == '__main__':
    main()
