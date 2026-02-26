from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.npz_io import npz_1d


@dataclass(frozen=True)
class Stage2Paths:
    infer_npz: Path
    out_segy: Path
    sidecar_npz: Path
    pick_csr_npz: Path | None


@dataclass(frozen=True)
class Stage1Seed:
    pick_final: np.ndarray
    scores_weight: dict[str, np.ndarray]
    trend_center_i_local: np.ndarray
    local_trend_ok: np.ndarray


def resolve_stage2_paths(
    segy_path: Path,
    *,
    cfg,
    infer_npz_path_for_segy_fn: Callable[..., Path],
    out_segy_path_for_in_fn: Callable[..., Path],
    out_sidecar_npz_path_for_out_fn: Callable[..., Path],
    out_pick_csr_npz_path_for_out_fn: Callable[..., Path],
) -> Stage2Paths:
    infer_npz = infer_npz_path_for_segy_fn(segy_path, cfg=cfg)
    if not infer_npz.exists():
        msg = f'infer npz not found for segy: {segy_path}  expected={infer_npz}'
        raise FileNotFoundError(msg)

    out_segy = out_segy_path_for_in_fn(segy_path, cfg=cfg)
    out_segy.parent.mkdir(parents=True, exist_ok=True)

    sidecar_npz = out_sidecar_npz_path_for_out_fn(out_segy, cfg=cfg)
    pick_csr_npz: Path | None = None
    if bool(cfg.emit_training_artifacts):
        pick_csr_npz = out_pick_csr_npz_path_for_out_fn(out_segy, cfg=cfg)

    return Stage2Paths(
        infer_npz=infer_npz,
        out_segy=out_segy,
        sidecar_npz=sidecar_npz,
        pick_csr_npz=pick_csr_npz,
    )


def load_stage1_seed_from_infer_npz(
    *,
    infer_npz: Path,
    n_traces: int,
    dt_sec_in: float,
    cfg,
    load_stage1_local_trend_center_i_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> Stage1Seed:
    with np.load(infer_npz, allow_pickle=False) as z:
        pick_final = npz_1d(
            z,
            cfg.pick_key,
            context='infer npz',
            n=int(n_traces),
            dtype=np.int64,
        )

        scores_weight: dict[str, np.ndarray] = {}
        for k in cfg.score_keys_for_weight:
            scores_weight[k] = npz_1d(
                z,
                k,
                context='infer npz',
                n=int(n_traces),
                dtype=np.float32,
            )

        trend_center_i_local, local_trend_ok = load_stage1_local_trend_center_i_fn(
            z=z,
            n_traces=n_traces,
            dt_sec_in=dt_sec_in,
            cfg=cfg,
        )

    return Stage1Seed(
        pick_final=pick_final,
        scores_weight=scores_weight,
        trend_center_i_local=trend_center_i_local,
        local_trend_ok=local_trend_ok,
    )


__all__ = [
    'Stage1Seed',
    'Stage2Paths',
    'load_stage1_seed_from_infer_npz',
    'resolve_stage2_paths',
]
