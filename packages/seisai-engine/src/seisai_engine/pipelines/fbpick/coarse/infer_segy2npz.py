from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import segyio
import torch
from seisai_dataset.file_info import build_file_info
from seisai_transforms.augment import PerTraceStandardize
from seisai_utils.config import optional_bool, optional_str, require_dict, require_list_str
from seisai_utils.listfiles import expand_cfg_listfiles

from seisai_engine.infer.segy2segy_cli_common import (
    build_merged_cfg_with_ckpt_cfg,
    resolve_ckpt_path,
    select_state_dict,
)
from seisai_engine.pipelines.common import (
    load_cfg_with_base_dir,
    load_checkpoint,
    resolve_device,
)
from seisai_engine.pipelines.fbpick.common.io import save_coarse_artifact

from .build_model import build_model
from .build_plan import build_input_only_plan_from_config
from .config import CoarseModelCfg, CoarseTrainConfig, load_coarse_train_config
from .infer import build_tiled_w_cfg, run_infer_batch

__all__ = ['main', 'run_infer_and_write']

DEFAULT_CONFIG_PATH = Path('examples/config_infer_fbpick_coarse.yaml')
_SAFE_OVERRIDE_PATHS = frozenset(
    {
        'paths.segy_files',
        'paths.out_dir',
        'paths.survey_id',
        'infer.ckpt_path',
        'infer.device',
        'infer.allow_unsafe_override',
        'dataset.use_header_cache',
        'dataset.waveform_mode',
        'dataset.infer_endian',
        'thresholds.confidence_min',
        'thresholds.trace_valid_min_fraction',
        'thresholds.qc_reject_confidence_below',
        'tile.tile_w',
        'tile.overlap_w',
        'tile.tiles_per_batch',
        'tile.amp',
        'tile.use_tqdm',
    }
)


def _default_cfg() -> dict[str, Any]:
    return {
        'paths': {
            'segy_files': [],
            'survey_id': '',
            'out_dir': './_fbpick_out',
        },
        'infer': {
            'ckpt_path': '',
            'device': 'auto',
            'allow_unsafe_override': False,
        },
        'dataset': {
            'use_header_cache': True,
            'waveform_mode': 'eager',
            'infer_endian': 'big',
        },
        'thresholds': {
            'confidence_min': 0.0,
            'trace_valid_min_fraction': 0.0,
            'qc_reject_confidence_below': 0.0,
        },
        'tile': {
            'tile_w': 6016,
            'overlap_w': 1024,
            'tiles_per_batch': 16,
            'amp': True,
            'use_tqdm': False,
        },
    }


def _validate_runtime_contract(typed: CoarseTrainConfig) -> None:
    if not isinstance(typed, CoarseTrainConfig):
        raise TypeError('typed must be CoarseTrainConfig')
    if str(typed.input.input_key) != 'input':
        raise ValueError('config.input.input_key must be "input" for coarse runtime')
    if str(typed.target.fb_index_key) != 'fb_idx_view':
        raise ValueError(
            'config.target.fb_index_key must be "fb_idx_view" to match coarse build_plan'
        )


def _build_model_from_ckpt(
    *,
    ckpt: dict[str, Any],
    typed: CoarseTrainConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], bool]:
    if str(ckpt.get('pipeline', '')) != 'fbpick_coarse':
        raise ValueError(
            f'checkpoint pipeline must be "fbpick_coarse", got {ckpt.get("pipeline")!r}'
        )

    model_sig = ckpt.get('model_sig')
    if not isinstance(model_sig, dict):
        raise TypeError('checkpoint model_sig must be dict')
    typed_model_sig = asdict(typed.model)
    if typed_model_sig != model_sig:
        raise ValueError('merged config model does not match checkpoint model_sig')

    output_ids = ckpt.get('output_ids')
    if not isinstance(output_ids, (list, tuple)):
        raise TypeError('checkpoint output_ids must be list[str] or tuple[str, ...]')
    if list(output_ids) != ['FB']:
        raise ValueError(f'checkpoint output_ids must be ["FB"], got {output_ids!r}')
    softmax_axis = ckpt.get('softmax_axis')
    if softmax_axis != 'time':
        raise ValueError(
            f'checkpoint softmax_axis must be "time" for coarse inference, got {softmax_axis!r}'
        )

    model_cfg = CoarseModelCfg(**model_sig)
    model = build_model(model_cfg)
    state_dict, used_ema = select_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()
    return model, dict(model_sig), used_ema


def _iter_ffid_windows(
    *,
    info: dict[str, Any],
    subset_traces: int,
) -> list[np.ndarray]:
    key_to_indices = info.get('ffid_key_to_indices')
    if not isinstance(key_to_indices, dict) or len(key_to_indices) == 0:
        raise ValueError(f'ffid_key_to_indices is missing or empty for {info.get("path")}')
    chno_values = np.asarray(info['chno_values'], dtype=np.int64)
    keys = info.get('ffid_unique_keys')
    if keys is None:
        keys = sorted(key_to_indices)

    windows: list[np.ndarray] = []
    for key in keys:
        idx = np.asarray(key_to_indices[int(key)], dtype=np.int64)
        if idx.size == 0:
            continue
        order = np.argsort(chno_values[idx], kind='mergesort')
        idx_sorted = idx[order]
        for start in range(0, int(idx_sorted.size), int(subset_traces)):
            windows.append(idx_sorted[start : start + int(subset_traces)])
    return windows


def _prepare_input_window(
    *,
    x_hw: np.ndarray,
    offsets: np.ndarray,
    dt_sec: float,
    subset_traces: int,
    standardizer: PerTraceStandardize,
    plan,
) -> tuple[torch.Tensor, np.ndarray]:
    h0 = int(x_hw.shape[0])
    w = int(x_hw.shape[1])
    if h0 <= 0 or w <= 0:
        raise ValueError(f'invalid gather window shape {(h0, w)}')
    if h0 > int(subset_traces):
        raise ValueError(f'gather window height {h0} exceeds subset_traces {int(subset_traces)}')
    if int(offsets.shape[0]) != h0:
        raise ValueError(f'offsets length {int(offsets.shape[0])} != H {h0}')

    x_pad = np.zeros((int(subset_traces), w), dtype=np.float32)
    x_pad[:h0] = np.asarray(x_hw, dtype=np.float32)
    x_std = standardizer(x_pad)
    if not isinstance(x_std, np.ndarray) or x_std.shape != x_pad.shape:
        raise TypeError('PerTraceStandardize must return numpy array with unchanged shape')
    x_std = x_std.astype(np.float32, copy=False)

    trace_valid = np.zeros((int(subset_traces),), dtype=np.bool_)
    trace_valid[:h0] = True
    offsets_pad = np.zeros((int(subset_traces),), dtype=np.float32)
    offsets_pad[:h0] = np.asarray(offsets, dtype=np.float32, copy=False)
    time_view = np.arange(w, dtype=np.float32) * np.float32(dt_sec)

    sample = {
        'x_view': x_std,
        'meta': {
            'trace_valid': trace_valid,
            'offsets_view': offsets_pad,
            'time_view': time_view,
        },
    }
    plan.run(sample, rng=None)
    x_in = sample.get('input')
    if not isinstance(x_in, torch.Tensor):
        raise TypeError('input-only coarse plan must produce torch.Tensor input')
    if tuple(x_in.shape) != (3, int(subset_traces), int(w)):
        raise ValueError(
            f'coarse input tensor must have shape (3,{int(subset_traces)},{int(w)}), got {tuple(x_in.shape)}'
        )
    return x_in, trace_valid


def run_infer_and_write(*, cfg: dict[str, Any], base_dir: Path) -> Path:
    expand_cfg_listfiles(cfg, keys=['paths.segy_files'])
    typed = load_coarse_train_config(cfg, base_dir=base_dir)
    _validate_runtime_contract(typed)

    paths_cfg = require_dict(cfg, 'paths')
    segy_files = require_list_str(paths_cfg, 'segy_files')
    if len(segy_files) == 0:
        raise ValueError('paths.segy_files must be non-empty')

    infer_cfg = require_dict(cfg, 'infer')
    device = resolve_device(optional_str(infer_cfg, 'device', 'auto'))
    ckpt_path = resolve_ckpt_path(cfg, base_dir=base_dir)
    ckpt = load_checkpoint(ckpt_path)
    model, model_sig, used_ema = _build_model_from_ckpt(
        ckpt=ckpt,
        typed=typed,
        device=device,
    )

    ds_cfg = require_dict(cfg, 'dataset')
    waveform_mode = optional_str(ds_cfg, 'waveform_mode', 'eager').lower()
    if waveform_mode not in ('eager', 'mmap'):
        raise ValueError('dataset.waveform_mode must be "eager" or "mmap"')
    infer_endian = optional_str(ds_cfg, 'infer_endian', 'big').lower()
    if infer_endian not in ('big', 'little'):
        raise ValueError('dataset.infer_endian must be "big" or "little"')
    use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)

    tile_cfg = build_tiled_w_cfg(require_dict(cfg, 'tile'))
    plan = build_input_only_plan_from_config(typed)
    subset_traces = int(typed.infer.subset_traces)
    transform_cfg = require_dict(cfg, 'transform')
    standardize_eps_raw = transform_cfg.get('standardize_eps')
    if standardize_eps_raw is None:
        eps_value = 1.0e-8
    else:
        if isinstance(standardize_eps_raw, bool) or not isinstance(standardize_eps_raw, (int, float)):
            raise TypeError('transform.standardize_eps must be float when provided')
        eps_value = float(standardize_eps_raw)
    standardizer = PerTraceStandardize(eps=float(eps_value))

    file_infos: list[dict[str, Any]] = []
    file_bases: list[int] = []
    n_total = 0
    n_samples_ref: int | None = None
    dt_us_ref: int | None = None
    time_axis: np.ndarray | None = None
    offsets_all: list[np.ndarray] = []
    resolved_segy_files = [str(Path(p).expanduser()) if Path(p).is_absolute() else str((base_dir / p).resolve()) for p in segy_files]
    for segy_path in resolved_segy_files:
        info = build_file_info(
            segy_path,
            ffid_byte=segyio.TraceField.FieldRecord,
            chno_byte=segyio.TraceField.TraceNumber,
            cmp_byte=segyio.TraceField.CDP,
            use_header_cache=bool(use_header_cache),
            header_cache_dir=None,
            include_centroids=False,
            waveform_mode=str(waveform_mode),
            segy_endian=str(infer_endian),
        )
        n_samples = int(info['n_samples'])
        dt_us = int(round(float(info['dt_sec']) * 1_000_000.0))
        if n_samples_ref is None:
            n_samples_ref = n_samples
            dt_us_ref = dt_us
            time_axis = np.arange(n_samples_ref, dtype=np.float32) * np.float32(float(info['dt_sec']))
        else:
            if n_samples != int(n_samples_ref):
                raise ValueError(
                    f'all SEG-Y files must share n_samples; got {n_samples} vs {int(n_samples_ref)} for {segy_path}'
                )
            if dt_us != int(dt_us_ref):
                raise ValueError(
                    f'all SEG-Y files must share dt; got {dt_us} vs {int(dt_us_ref)} microseconds for {segy_path}'
                )
        file_bases.append(int(n_total))
        n_total += int(info['n_traces'])
        offsets_all.append(np.asarray(info['offsets'], dtype=np.float32))
        file_infos.append(info)

    if n_total <= 0 or n_samples_ref is None or time_axis is None:
        raise RuntimeError('no traces were loaded from paths.segy_files')

    prob = np.zeros((int(n_total), int(n_samples_ref)), dtype=np.float32)
    pick_idx = np.full((int(n_total),), -1, dtype=np.int32)
    confidence = np.zeros((int(n_total),), dtype=np.float32)
    raw_trace_idx = np.arange(int(n_total), dtype=np.int64)
    offsets = np.concatenate(offsets_all, axis=0).astype(np.float32, copy=False)
    seen = np.zeros((int(n_total),), dtype=np.bool_)

    non_blocking = bool(device.type == 'cuda')
    try:
        for file_idx, info in enumerate(file_infos):
            file_base = int(file_bases[file_idx])
            windows = _iter_ffid_windows(info=info, subset_traces=int(subset_traces))
            for idx_win in windows:
                idx_np = np.asarray(idx_win, dtype=np.int64)
                if idx_np.size == 0:
                    continue
                x_hw = np.asarray(info['mmap'][idx_np], dtype=np.float32)
                x_in, trace_valid = _prepare_input_window(
                    x_hw=x_hw,
                    offsets=np.asarray(info['offsets'], dtype=np.float32)[idx_np],
                    dt_sec=float(info['dt_sec']),
                    subset_traces=int(subset_traces),
                    standardizer=standardizer,
                    plan=plan,
                )
                x_dev = x_in.unsqueeze(0).to(device=device, non_blocking=non_blocking)
                trace_valid_dev = torch.from_numpy(trace_valid[None, :]).to(device=device)
                batch_result = run_infer_batch(
                    model=model,
                    x_bchw=x_dev,
                    tiled_cfg=tile_cfg,
                    trace_valid=trace_valid_dev,
                )

                prob_win = batch_result.prob[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                pick_win = batch_result.pick_idx[0].detach().cpu().numpy().astype(np.int64, copy=False)
                conf_win = batch_result.confidence[0].detach().cpu().numpy().astype(np.float32, copy=False)
                for local_idx, raw_local_idx in enumerate(idx_np.tolist()):
                    global_idx = file_base + int(raw_local_idx)
                    if seen[global_idx]:
                        raise RuntimeError(f'duplicate prediction for raw trace index {global_idx}')
                    seen[global_idx] = True
                    prob[global_idx] = prob_win[local_idx]
                    pick_idx[global_idx] = np.int32(pick_win[local_idx])
                    confidence[global_idx] = conf_win[local_idx]
    finally:
        for info in file_infos:
            segy_obj = info.get('segy_obj')
            if segy_obj is not None:
                segy_obj.close()

    if not np.all(seen):
        missing = int(np.count_nonzero(~seen))
        raise RuntimeError(f'coarse inference did not cover all raw traces; missing={missing}')

    trace_valid = confidence >= float(typed.fbpick.thresholds.confidence_min)
    valid_fraction = float(np.mean(trace_valid.astype(np.float32)))
    if valid_fraction < float(typed.fbpick.thresholds.trace_valid_min_fraction):
        raise RuntimeError(
            'coarse trace_valid fraction below threshold: '
            f'{valid_fraction:.6f} < {float(typed.fbpick.thresholds.trace_valid_min_fraction):.6f}'
        )

    save_coarse_artifact(
        paths_cfg=typed.fbpick.paths,
        arrays={
            'prob': prob,
            'pick_idx': pick_idx,
            'confidence': confidence,
            'trace_valid': trace_valid,
            'raw_trace_idx': raw_trace_idx,
            'offsets': offsets,
            'time_axis': time_axis,
        },
        source_refs={
            'ckpt_path': str(ckpt_path),
            'config_path': str((base_dir / Path(cfg.get('__config_path__', ''))).resolve())
            if '__config_path__' in cfg
            else str(base_dir),
            'used_ema': 'true' if used_ema else 'false',
            'model_sig': str(model_sig),
            **{
                f'segy_file_{i:03d}': str(path)
                for i, path in enumerate(resolved_segy_files)
            },
        },
    )
    return Path(typed.fbpick.paths.out_dir) / typed.fbpick.paths.survey_id / 'coarse' / 'coarse_artifact.npz'


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, unknown = parser.parse_known_args(argv)

    infer_yaml_cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    infer_yaml_cfg['__config_path__'] = str(Path(args.config))
    merged_cfg = build_merged_cfg_with_ckpt_cfg(
        infer_yaml_cfg=infer_yaml_cfg,
        base_dir=base_dir,
        unknown_overrides=unknown,
        default_cfg=_default_cfg(),
        safe_paths=_SAFE_OVERRIDE_PATHS,
    )
    out_path = run_infer_and_write(cfg=merged_cfg, base_dir=base_dir)
    print(str(out_path))
