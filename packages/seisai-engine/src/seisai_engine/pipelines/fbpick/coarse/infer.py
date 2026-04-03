from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from seisai_dataset import InputOnlyPlan, collate_pad_w_right
from seisai_engine.infer.runner import TiledWConfig, iter_infer_loader_tiled_w
from seisai_engine.pipelines.fbpick.common import (
    build_lineage_payload,
    save_coarse_npz,
)

from .build_dataset import build_raw_infer_dataset
from .build_plan import build_plan
from .config import load_coarse_infer_config

__all__ = ['run_coarse_infer']


def _softmax_last_axis(logits_hw: np.ndarray) -> np.ndarray:
    shifted = logits_hw - logits_hw.max(axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    denom = exp.sum(axis=-1, keepdims=True)
    return exp / denom


def run_coarse_infer(
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    source_model_id: str | None = None,
    iter_id: int | None = None,
    repo_root: Path = Path('/workspace'),
) -> Path:
    typed = load_coarse_infer_config(cfg)
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
        subset_traces=typed.infer.subset_traces,
        overlap_h=typed.infer.overlap_h,
        time_len=typed.transform.time_len,
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
        collate_fn=collate_pad_w_right,
    )

    try:
        info = dataset.file_infos[0]
        n_traces = int(info['n_traces'])
        n_samples_orig = int(info['n_samples'])
        logits_sum = np.zeros((n_traces, n_samples_orig), dtype=np.float32)
        counts = np.zeros((n_traces,), dtype=np.int32)

        tiled_cfg = TiledWConfig(
            tile_w=typed.infer.tile_w,
            overlap_w=typed.infer.overlap_w,
            tiles_per_batch=typed.infer.tiles_per_batch,
            amp=typed.infer.amp,
            use_tqdm=typed.infer.use_tqdm,
        )

        for logits_b1hw, metas in iter_infer_loader_tiled_w(
            model,
            loader,
            device=device,
            cfg=tiled_cfg,
            output_to_cpu=True,
        ):
            logits_bhw = logits_b1hw[:, 0, :, :].detach().cpu().numpy()
            for batch_idx, meta in enumerate(metas):
                raw_idx_global = np.asarray(meta['raw_idx_global'], dtype=np.int64)
                trace_valid = np.asarray(meta['trace_valid'], dtype=np.bool_)
                if raw_idx_global.shape != trace_valid.shape:
                    msg = 'raw_idx_global and trace_valid must have the same shape'
                    raise ValueError(msg)

                for row_idx, (raw_idx, is_valid) in enumerate(
                    zip(raw_idx_global, trace_valid, strict=True)
                ):
                    if not bool(is_valid):
                        continue
                    raw_trace_idx = int(raw_idx)
                    if raw_trace_idx < 0 or raw_trace_idx >= n_traces:
                        msg = f'raw trace index out of range: {raw_trace_idx}'
                        raise ValueError(msg)
                    row_logits = np.asarray(logits_bhw[batch_idx, row_idx], dtype=np.float32)
                    if int(row_logits.shape[0]) < n_samples_orig:
                        msg = 'inference logits width is shorter than n_samples_orig'
                        raise ValueError(msg)
                    logits_sum[raw_trace_idx] += row_logits[:n_samples_orig]
                    counts[raw_trace_idx] += 1

        if np.any(counts <= 0):
            msg = 'some raw traces were not covered by inference windows'
            raise RuntimeError(msg)

        avg_logits = logits_sum / counts[:, None].astype(np.float32)
        prob = _softmax_last_axis(avg_logits)
        coarse_pick_i = avg_logits.argmax(axis=-1).astype(np.int32)
        coarse_pmax = prob.max(axis=-1).astype(np.float32)
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
