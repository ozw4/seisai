from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from seisai_engine.pipelines.fbpick.common import (
    build_lineage_payload,
    load_coarse_npz,
    save_robust_npz,
)

from .confidence import compute_confidence_terms
from .config import load_physics_lite_config, physics_lite_config_to_dict
from .feasible import compute_feasible_band
from .merge import apply_keep_reject_fill
from .pick_table import normalize_coarse_pick_table
from .trend import build_trend_result

__all__ = ['build_robust_payload_from_coarse', 'derive_robust_npz_path', 'run_physics_lite']


def derive_robust_npz_path(coarse_npz_path: str | Path) -> Path:
    path = Path(coarse_npz_path).expanduser().resolve()
    if not path.name.endswith('.coarse.npz'):
        msg = f'expected *.coarse.npz input, got {path.name}'
        raise ValueError(msg)
    stem = path.name[: -len('.coarse.npz')]
    return path.with_name(f'{stem}.robust.npz')


def build_robust_payload_from_coarse(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    cfg: dict[str, Any] | None,
    source_model_id: str | None = None,
    iter_id: int | str | None = '',
    repo_root: Path | None = None,
) -> dict[str, np.ndarray]:
    typed_cfg = load_physics_lite_config(cfg)
    canonical_cfg = physics_lite_config_to_dict(typed_cfg)

    table = normalize_coarse_pick_table(coarse_npz)
    feasible = compute_feasible_band(table, typed_cfg.feasible_band)
    trend = build_trend_result(table, feasible, typed_cfg)
    confidence = compute_confidence_terms(table, feasible, trend, typed_cfg)
    merged = apply_keep_reject_fill(table, feasible, trend, confidence, typed_cfg)

    return {
        'dt_sec': np.asarray(table.dt_scalar_sec, dtype=np.float32),
        'n_samples_orig': np.asarray(table.n_samples_orig, dtype=np.int32),
        'n_traces': np.asarray(table.n_traces, dtype=np.int32),
        'ffid_values': np.asarray(table.ffid, dtype=np.int32),
        'chno_values': np.asarray(table.chno, dtype=np.int32),
        'offsets_m': np.asarray(table.offset_m, dtype=np.float32),
        'trace_indices': np.asarray(table.trace_id, dtype=np.int64),
        'robust_pick_i': np.asarray(merged.robust_pick_i, dtype=np.int32),
        'robust_pick_t_sec': np.asarray(merged.robust_pick_t_sec, dtype=np.float32),
        'robust_conf': np.asarray(merged.robust_conf, dtype=np.float32),
        'robust_source': np.asarray(merged.robust_source, dtype=np.uint8),
        'used_theoretical_mask': np.asarray(
            merged.used_theoretical_mask,
            dtype=np.bool_,
        ),
        'reason_mask': np.asarray(merged.reason_mask, dtype=np.uint8),
        'conf_prob1': np.asarray(confidence.conf_prob1, dtype=np.float32),
        'conf_trend1': np.asarray(confidence.conf_trend1, dtype=np.float32),
        'conf_rs1': np.asarray(confidence.conf_rs1, dtype=np.float32),
        'lineage': build_lineage_payload(
            canonical_cfg,
            repo_root=repo_root,
            source_model_id=source_model_id,
            iter_id=iter_id,
        ),
    }


def run_physics_lite(
    coarse_npz_path: str | Path,
    *,
    cfg: dict[str, Any] | None,
    out_path: str | Path | None = None,
    source_model_id: str | None = None,
    iter_id: int | str | None = '',
    repo_root: Path | None = None,
) -> Path:
    coarse_path = Path(coarse_npz_path).expanduser().resolve()
    payload = build_robust_payload_from_coarse(
        load_coarse_npz(coarse_path),
        cfg=cfg,
        source_model_id=source_model_id,
        iter_id=iter_id,
        repo_root=repo_root,
    )
    target_path = derive_robust_npz_path(coarse_path) if out_path is None else Path(out_path)
    return save_robust_npz(target_path, **payload)
