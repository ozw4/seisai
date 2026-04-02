from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from seisai_engine.pipelines.fbpick.common.io import resolve_artifact_paths, save_global_qc_artifact

from .config import GlobalQcConfig

__all__ = [
    'GlobalQcExportResult',
    'export_global_qc_result',
]


@dataclass(frozen=True)
class GlobalQcExportResult:
    artifact_npz_path: Path
    artifact_meta_path: Path
    csv_path: Path | None


def _validate_vector_lengths(
    *,
    raw_trace_idx: np.ndarray,
    pick_global: np.ndarray,
    confidence_global: np.ndarray,
    reject_flag: np.ndarray,
    qc_status: np.ndarray,
) -> int:
    n_traces = int(raw_trace_idx.shape[0])
    for name, value in (
        ('pick_global', pick_global),
        ('confidence_global', confidence_global),
        ('reject_flag', reject_flag),
        ('qc_status', qc_status),
    ):
        if value.shape != (n_traces,):
            msg = f'{name} must have shape ({n_traces},), got {value.shape}'
            raise ValueError(msg)
    return n_traces


def _resolve_csv_path(cfg: GlobalQcConfig) -> Path:
    if cfg.export.csv_path is not None:
        return Path(cfg.export.csv_path).expanduser().resolve()
    return resolve_artifact_paths(cfg.fbpick.paths, stage='global_qc').stage_dir / 'global_qc_results.csv'


def export_global_qc_result(
    cfg: GlobalQcConfig,
    *,
    pick_global: np.ndarray,
    confidence_global: np.ndarray,
    reject_flag: np.ndarray,
    qc_status: np.ndarray,
    raw_trace_idx: np.ndarray,
    source_refs: Mapping[str, str] | None = None,
) -> GlobalQcExportResult:
    if not isinstance(cfg, GlobalQcConfig):
        msg = 'cfg must be GlobalQcConfig'
        raise TypeError(msg)

    raw_trace_idx_np = np.asarray(raw_trace_idx, dtype=np.int64)
    pick_global_np = np.asarray(pick_global, dtype=np.int32)
    confidence_global_np = np.asarray(confidence_global, dtype=np.float32)
    reject_flag_np = np.asarray(reject_flag, dtype=bool)
    qc_status_np = np.asarray(qc_status, dtype=np.int8)
    _validate_vector_lengths(
        raw_trace_idx=raw_trace_idx_np,
        pick_global=pick_global_np,
        confidence_global=confidence_global_np,
        reject_flag=reject_flag_np,
        qc_status=qc_status_np,
    )

    saved = save_global_qc_artifact(
        paths_cfg=cfg.fbpick.paths,
        arrays={
            'pick_global': pick_global_np,
            'confidence_global': confidence_global_np,
            'reject_flag': reject_flag_np,
            'qc_status': qc_status_np,
            'raw_trace_idx': raw_trace_idx_np,
        },
        source_refs=source_refs,
    )

    csv_path: Path | None = None
    if cfg.export.write_csv:
        csv_path = _resolve_csv_path(cfg)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['raw_trace_idx', 'pick_global', 'confidence_global', 'reject_flag', 'qc_status']
            )
            for row_idx in range(int(raw_trace_idx_np.shape[0])):
                writer.writerow(
                    [
                        int(raw_trace_idx_np[row_idx]),
                        int(pick_global_np[row_idx]),
                        float(confidence_global_np[row_idx]),
                        bool(reject_flag_np[row_idx]),
                        int(qc_status_np[row_idx]),
                    ]
                )

    return GlobalQcExportResult(
        artifact_npz_path=saved.npz_path,
        artifact_meta_path=saved.meta_path,
        csv_path=csv_path,
    )
