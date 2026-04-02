from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from seisai_utils.config import require_dict, require_list_str
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
from seisai_engine.pipelines.common.config_io import resolve_relpath
from seisai_engine.pipelines.fbpick.common.io import (
    load_coarse_artifact_from_paths,
    save_fine_artifact,
)

from .build_model import build_model
from .build_plan import build_input_only_plan_from_config
from .build_window_dataset import build_window_dataset, resolve_coarse_artifact_paths
from .config import FineInferConfig, FineModelCfg, load_fine_infer_config
from .infer import run_infer_batch

__all__ = ['main', 'run_infer_and_write']

DEFAULT_CONFIG_PATH = Path('examples/config_infer_fbpick_fine.yaml')
_SAFE_OVERRIDE_PATHS = frozenset(
    {
        'paths.segy_files',
        'paths.out_dir',
        'paths.survey_id',
        'infer.ckpt_path',
        'infer.device',
        'infer.allow_unsafe_override',
        'coarse_seed.artifact_npz_path',
        'coarse_seed.artifact_meta_path',
        'thresholds.confidence_min',
        'thresholds.trace_valid_min_fraction',
        'thresholds.qc_reject_confidence_below',
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
    }


def _validate_runtime_contract(typed: FineInferConfig) -> None:
    if not isinstance(typed, FineInferConfig):
        raise TypeError('typed must be FineInferConfig')
    if str(typed.input.input_key) != 'input':
        raise ValueError('config.input.input_key must be "input" for fine runtime')
    if str(typed.target.local_pick_idx_key) != 'local_pick_idx':
        raise ValueError(
            'config.target.local_pick_idx_key must be "local_pick_idx" to match fine build_plan'
        )
    if int(typed.model.in_chans) != 1 or int(typed.model.out_chans) != 1:
        raise ValueError('fine runtime requires model in_chans=1 and out_chans=1')


def _resolve_paths(base_dir: Path, paths: list[str]) -> list[str]:
    resolved: list[str] = []
    for idx, item in enumerate(paths):
        if not isinstance(item, str):
            raise TypeError(f'path[{idx}] must be str')
        resolved.append(resolve_relpath(base_dir, item))
    return resolved


def _to_numpy_vector(value: torch.Tensor | Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 2 and int(arr.shape[1]) == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f'{name} must have shape (B,) or (B,1), got {arr.shape}')
    return arr.astype(dtype, copy=False)


def _validate_dataset_runtime_sample(
    *,
    sample: dict[str, Any],
    amplitude_key: str,
) -> None:
    if not isinstance(sample, dict):
        raise TypeError('fine infer dataset sample must be dict')
    if amplitude_key not in sample:
        raise KeyError(f'fine infer dataset sample must contain {amplitude_key!r}')
    if 'input' not in sample:
        raise KeyError('fine infer dataset sample must contain "input"')
    if 'raw_sample_idx_local' not in sample:
        raise KeyError('fine infer dataset sample must contain "raw_sample_idx_local"')
    if 'meta' not in sample or not isinstance(sample['meta'], dict):
        raise KeyError('fine infer dataset sample must contain dict meta')
    if 'raw_sample_idx_local' not in sample['meta']:
        raise KeyError('fine infer dataset sample meta must contain "raw_sample_idx_local"')

    amplitude = sample[amplitude_key]
    x_input = sample['input']
    raw_sample_idx_local = sample['raw_sample_idx_local']
    meta_raw_sample_idx_local = sample['meta']['raw_sample_idx_local']

    if not isinstance(amplitude, torch.Tensor):
        amplitude = torch.as_tensor(amplitude)
    if not isinstance(x_input, torch.Tensor):
        raise TypeError('fine infer dataset sample["input"] must be torch.Tensor')
    if not isinstance(raw_sample_idx_local, torch.Tensor):
        raw_sample_idx_local = torch.as_tensor(raw_sample_idx_local)
    if not isinstance(meta_raw_sample_idx_local, torch.Tensor):
        meta_raw_sample_idx_local = torch.as_tensor(meta_raw_sample_idx_local)

    if tuple(x_input.shape[:2]) != (1, 1):
        raise ValueError(
            f'fine infer dataset sample["input"] must have shape (1,1,W), got {tuple(x_input.shape)}'
        )
    if amplitude.ndim != 2 or int(amplitude.shape[0]) != 1:
        raise ValueError(
            f'fine infer dataset sample[{amplitude_key!r}] must have shape (1,W), got {tuple(amplitude.shape)}'
        )
    if not torch.equal(x_input[0], amplitude):
        raise ValueError(
            f'fine infer dataset sample["input"][0] must match sample[{amplitude_key!r}] exactly'
        )
    if not torch.equal(raw_sample_idx_local.to(dtype=torch.int64), meta_raw_sample_idx_local.to(dtype=torch.int64)):
        raise ValueError(
            'fine infer dataset sample["raw_sample_idx_local"] must match '
            'meta["raw_sample_idx_local"] exactly'
        )


def _build_model_from_ckpt(
    *,
    ckpt: dict[str, Any],
    typed: FineInferConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], bool]:
    if str(ckpt.get('pipeline', '')) != 'fbpick_fine':
        raise ValueError(f'checkpoint pipeline must be "fbpick_fine", got {ckpt.get("pipeline")!r}')

    model_sig = ckpt.get('model_sig')
    if not isinstance(model_sig, dict):
        raise TypeError('checkpoint model_sig must be dict')
    typed_model_sig = asdict(typed.model)
    if typed_model_sig != model_sig:
        raise ValueError('merged config model does not match checkpoint model_sig')

    output_ids = ckpt.get('output_ids')
    if not isinstance(output_ids, (list, tuple)):
        raise TypeError('checkpoint output_ids must be list[str] or tuple[str, ...]')
    if list(output_ids) != ['FB_LOCAL_PROB']:
        raise ValueError(f'checkpoint output_ids must be ["FB_LOCAL_PROB"], got {output_ids!r}')
    if ckpt.get('softmax_axis') != 'time':
        raise ValueError(
            f'checkpoint softmax_axis must be "time" for fine inference, got {ckpt.get("softmax_axis")!r}'
        )
    if ckpt.get('input_semantics') != 'amplitude_only_1ch':
        raise ValueError(
            'checkpoint input_semantics must be "amplitude_only_1ch" for fine inference'
        )
    if ckpt.get('raw_pick_restore_key') != 'raw_sample_idx_local':
        raise ValueError(
            'checkpoint raw_pick_restore_key must be "raw_sample_idx_local" for fine inference'
        )
    if ckpt.get('invalid_index') != -1:
        raise ValueError('checkpoint invalid_index must be -1 for fine inference')

    model_cfg = FineModelCfg(**model_sig)
    model = build_model(model_cfg)
    state_dict, used_ema = select_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()
    return model, dict(model_sig), used_ema


def run_infer_and_write(*, cfg: dict[str, Any], base_dir: Path) -> Path:
    expand_cfg_listfiles(cfg, keys=['paths.segy_files'])
    typed = load_fine_infer_config(cfg, base_dir=base_dir)
    _validate_runtime_contract(typed)

    paths_cfg = require_dict(cfg, 'paths')
    segy_files = _resolve_paths(base_dir, require_list_str(paths_cfg, 'segy_files'))
    if len(segy_files) == 0:
        raise ValueError('paths.segy_files must be non-empty')

    device = resolve_device(str(typed.infer.device))
    ckpt_path = resolve_ckpt_path(cfg, base_dir=base_dir)
    ckpt = load_checkpoint(ckpt_path)
    model, model_sig, used_ema = _build_model_from_ckpt(
        ckpt=ckpt,
        typed=typed,
        device=device,
    )

    coarse_npz_path, coarse_meta_path = resolve_coarse_artifact_paths(typed.coarse_seed)
    _ = load_coarse_artifact_from_paths(
        npz_path=coarse_npz_path,
        meta_path=coarse_meta_path,
        survey_id=typed.fbpick.paths.survey_id,
    )

    plan = build_input_only_plan_from_config(typed)
    ds_infer_full = build_window_dataset(
        segy_files=list(segy_files),
        transform=None,
        plan=plan,
        input_cfg=typed.input,
        window_cfg=typed.window,
        coarse_seed_cfg=typed.coarse_seed,
    )
    _validate_dataset_runtime_sample(
        sample=ds_infer_full[0],
        amplitude_key=str(typed.input.amplitude_key),
    )

    loader = torch.utils.data.DataLoader(
        ds_infer_full,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'fine inference windows={len(ds_infer_full)} batch_size=1')

    local_prob_chunks: list[np.ndarray] = []
    local_pick_idx_chunks: list[np.ndarray] = []
    raw_pick_idx_chunks: list[np.ndarray] = []
    local_window_start_idx_chunks: list[np.ndarray] = []
    local_window_end_idx_chunks: list[np.ndarray] = []
    raw_trace_idx_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    seen_raw_trace_idx: set[int] = set()
    non_blocking = bool(device.type == 'cuda')

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, dict):
                raise TypeError('fine infer loader must yield dict batches')
            for key in ('input', 'raw_sample_idx_local', 'raw_trace_idx', 'local_window_start_idx', 'local_window_end_idx'):
                if key not in batch:
                    raise KeyError(f'fine infer batch must contain {key!r}')

            x_in = batch['input'].to(device=device, non_blocking=non_blocking)
            batch_result = run_infer_batch(
                model=model,
                x_bchw=x_in,
                raw_sample_idx_local=batch['raw_sample_idx_local'],
                trace_valid=batch.get('trace_valid'),
            )

            local_prob_np = batch_result.local_prob[:, 0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            local_pick_idx_np = _to_numpy_vector(
                batch_result.local_pick_idx,
                name='local_pick_idx',
                dtype=np.int32,
            )
            raw_pick_idx_np = _to_numpy_vector(
                batch_result.raw_pick_idx,
                name='raw_pick_idx',
                dtype=np.int32,
            )
            raw_trace_idx_np = _to_numpy_vector(
                batch['raw_trace_idx'],
                name='raw_trace_idx',
                dtype=np.int64,
            )
            local_window_start_idx_np = _to_numpy_vector(
                batch['local_window_start_idx'],
                name='local_window_start_idx',
                dtype=np.int64,
            )
            local_window_end_idx_np = _to_numpy_vector(
                batch['local_window_end_idx'],
                name='local_window_end_idx',
                dtype=np.int64,
            )
            confidence_np = _to_numpy_vector(
                batch_result.confidence,
                name='confidence',
                dtype=np.float32,
            )

            for raw_trace_idx in raw_trace_idx_np.tolist():
                raw_trace_idx_int = int(raw_trace_idx)
                if raw_trace_idx_int in seen_raw_trace_idx:
                    raise RuntimeError(f'duplicate fine prediction for raw_trace_idx={raw_trace_idx_int}')
                seen_raw_trace_idx.add(raw_trace_idx_int)

            local_prob_chunks.append(local_prob_np)
            local_pick_idx_chunks.append(local_pick_idx_np)
            raw_pick_idx_chunks.append(raw_pick_idx_np)
            local_window_start_idx_chunks.append(local_window_start_idx_np)
            local_window_end_idx_chunks.append(local_window_end_idx_np)
            raw_trace_idx_chunks.append(raw_trace_idx_np)
            confidence_chunks.append(confidence_np)

    if not local_prob_chunks:
        raise RuntimeError('fine inference produced no windows')

    local_prob = np.concatenate(local_prob_chunks, axis=0).astype(np.float32, copy=False)
    local_pick_idx = np.concatenate(local_pick_idx_chunks, axis=0).astype(np.int32, copy=False)
    raw_pick_idx = np.concatenate(raw_pick_idx_chunks, axis=0).astype(np.int32, copy=False)
    local_window_start_idx = np.concatenate(local_window_start_idx_chunks, axis=0).astype(np.int64, copy=False)
    local_window_end_idx = np.concatenate(local_window_end_idx_chunks, axis=0).astype(np.int64, copy=False)
    raw_trace_idx = np.concatenate(raw_trace_idx_chunks, axis=0).astype(np.int64, copy=False)
    confidence = np.concatenate(confidence_chunks, axis=0).astype(np.float32, copy=False)
    if int(local_prob.shape[0]) != int(len(ds_infer_full)):
        raise RuntimeError(
            f'fine inference window count mismatch: arrays={int(local_prob.shape[0])} dataset={len(ds_infer_full)}'
        )

    saved_paths = save_fine_artifact(
        paths_cfg=typed.fbpick.paths,
        arrays={
            'local_prob': local_prob,
            'local_pick_idx': local_pick_idx,
            'raw_pick_idx': raw_pick_idx,
            'local_window_start_idx': local_window_start_idx,
            'local_window_end_idx': local_window_end_idx,
            'raw_trace_idx': raw_trace_idx,
            'confidence': confidence,
        },
        source_refs={
            'ckpt_path': str(ckpt_path),
            'coarse_artifact_npz_path': str(coarse_npz_path),
            'coarse_artifact_meta_path': str(coarse_meta_path),
            'config_path': str((base_dir / Path(cfg.get('__config_path__', ''))).resolve())
            if '__config_path__' in cfg
            else str(base_dir),
            'used_ema': 'true' if used_ema else 'false',
            'model_sig': str(model_sig),
            **{f'segy_file_{i:03d}': str(path) for i, path in enumerate(segy_files)},
        },
    )
    return saved_paths.npz_path


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


if __name__ == '__main__':
    main()
