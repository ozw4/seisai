from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from seisai_utils.listfiles import expand_cfg_listfiles, get_cfg_listfile_meta

from seisai_engine.optim import build_optimizer
from seisai_engine.pipelines.common import (
    load_cfg_with_base_dir,
    maybe_load_init_weights,
    resolve_cfg_paths,
    resolve_device,
    resolve_out_dir,
)

from .build_dataset import (
    build_fbgate,
    build_labeled_infer_dataset,
    build_train_dataset,
)
from .build_model import build_model
from .build_plan import build_plan
from .config import CoarseTrainConfig, load_coarse_train_config
from .loss import build_criterion

__all__ = ['CoarseTrainBundle', 'build_train_bundle', 'load_train_bundle']


@dataclass
class CoarseTrainBundle:
    cfg: dict[str, Any]
    base_dir: Path
    typed: CoarseTrainConfig
    out_dir: Path
    model_sig: dict[str, Any]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: Any
    ds_train_full: Any
    ds_infer_full: Any
    device: torch.device


def _prepare_cfg(cfg: dict, *, base_dir: Path) -> tuple[dict, list[dict[str, object] | None] | None, list[dict[str, object] | None] | None]:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    paths = cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'paths must be dict'
        raise TypeError(msg)

    path_keys: list[str] = ['paths.segy_files', 'paths.fb_files']
    if 'infer_segy_files' in paths:
        path_keys.append('paths.infer_segy_files')
    if 'infer_fb_files' in paths:
        path_keys.append('paths.infer_fb_files')

    expand_cfg_listfiles(cfg, keys=path_keys)
    train_sampling_overrides = get_cfg_listfile_meta(cfg, key_path='paths.segy_files')
    if 'infer_segy_files' in paths:
        infer_sampling_overrides = get_cfg_listfile_meta(
            cfg,
            key_path='paths.infer_segy_files',
        )
    else:
        infer_sampling_overrides = train_sampling_overrides

    resolve_cfg_paths(cfg, base_dir, keys=path_keys)
    return cfg, train_sampling_overrides, infer_sampling_overrides


def _build_fbgate_from_cfg(cfg: dict, *, default_verbose: bool) -> FirstBreakGate:
    fbgate_cfg = cfg.get('fbgate')
    if fbgate_cfg is None:
        return build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=default_verbose)
    if not isinstance(fbgate_cfg, dict):
        msg = 'fbgate must be dict'
        raise TypeError(msg)
    apply_on = str(fbgate_cfg.get('apply_on', 'off'))
    min_pick_ratio = float(fbgate_cfg.get('min_pick_ratio', 0.0))
    verbose = bool(fbgate_cfg.get('verbose', default_verbose))
    return build_fbgate(
        apply_on=apply_on,
        min_pick_ratio=min_pick_ratio,
        verbose=verbose,
    )


def build_train_bundle(
    cfg: dict,
    *,
    base_dir: Path,
    device: torch.device,
) -> CoarseTrainBundle:
    cfg_prepared, train_sampling_overrides, infer_sampling_overrides = _prepare_cfg(
        cfg,
        base_dir=base_dir,
    )
    typed = load_coarse_train_config(cfg_prepared)
    if typed.paths.fb_files is None:
        msg = 'paths.fb_files is required for coarse training'
        raise ValueError(msg)
    if typed.paths.infer_segy_files is None or typed.paths.infer_fb_files is None:
        msg = 'coarse train infer dataset requires labels; provide infer files or fb_files'
        raise ValueError(msg)

    plan = build_plan(
        sigma_ms=typed.train.fb_sigma_ms,
        time_ref_sec=typed.norm_refs.time_ref_sec,
        offset_ref_m=typed.norm_refs.offset_ref_m,
    )
    fbgate = _build_fbgate_from_cfg(cfg_prepared, default_verbose=typed.dataset.verbose)

    ds_train_full = build_train_dataset(
        segy_files=list(typed.paths.segy_files),
        fb_files=list(typed.paths.fb_files),
        sampling_overrides=train_sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        subset_traces=typed.train.subset_traces,
        time_len=typed.transform.time_len,
        standardize_eps=typed.transform.standardize_eps,
        trace_decimate_prob=typed.train.trace_decimate_prob,
        trace_decimate_stride_range=typed.train.trace_decimate_stride_range,
        primary_keys=typed.dataset.primary_keys,
        secondary_key_fixed=typed.dataset.secondary_key_fixed,
        verbose=typed.dataset.verbose,
        progress=typed.dataset.progress,
        max_trials=typed.dataset.max_trials,
        use_header_cache=typed.dataset.use_header_cache,
        waveform_mode=typed.dataset.waveform_mode,
        segy_endian=typed.dataset.train_endian,
    )
    ds_infer_full = build_labeled_infer_dataset(
        segy_files=list(typed.paths.infer_segy_files),
        fb_files=list(typed.paths.infer_fb_files),
        sampling_overrides=infer_sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        subset_traces=typed.infer.subset_traces,
        time_len=typed.transform.time_len,
        standardize_eps=typed.transform.standardize_eps,
        primary_keys=typed.dataset.primary_keys,
        secondary_key_fixed=typed.dataset.secondary_key_fixed,
        verbose=typed.dataset.verbose,
        progress=typed.dataset.progress,
        max_trials=typed.dataset.max_trials,
        use_header_cache=typed.dataset.use_header_cache,
        waveform_mode=typed.dataset.waveform_mode,
        segy_endian=typed.dataset.train_endian,
    )

    model_sig = dict(typed.model_sig)
    model = build_model(model_sig)
    maybe_load_init_weights(
        cfg=cfg_prepared,
        base_dir=base_dir,
        model=model,
        model_sig=model_sig,
    )
    model.to(device)

    optimizer = build_optimizer(
        cfg_prepared,
        model,
        lr=typed.train.lr,
        weight_decay=typed.train.weight_decay,
    )
    criterion = build_criterion()

    return CoarseTrainBundle(
        cfg=cfg_prepared,
        base_dir=base_dir,
        typed=typed,
        out_dir=resolve_out_dir(cfg_prepared, base_dir),
        model_sig=model_sig,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ds_train_full=ds_train_full,
        ds_infer_full=ds_infer_full,
        device=device,
    )


def load_train_bundle(
    config_path: str | Path,
    *,
    device: torch.device | None = None,
) -> CoarseTrainBundle:
    cfg, base_dir = load_cfg_with_base_dir(Path(config_path))
    if device is None:
        device = resolve_device(None)
    return build_train_bundle(cfg, base_dir=base_dir, device=device)
