from __future__ import annotations

from pathlib import Path

import numpy as np
from common.npz_io import npz_1d, npz_scalar_float
from stage2.io import Stage2Seed


def load_stage4_seed_from_pred_npz(
    *,
    pred_npz: Path,
    n_traces: int,
    dt_sec_in: float,
    cfg,
) -> Stage2Seed:
    scores_weight: dict[str, np.ndarray] = {}
    with np.load(pred_npz, allow_pickle=False) as z:
        pick_final = npz_1d(
            z,
            'pick_final',
            context='pred npz',
            n=int(n_traces),
            dtype=np.int64,
        )
        pmax_psn = npz_1d(
            z,
            'pmax_psn',
            context='pred npz',
            n=int(n_traces),
            dtype=np.float32,
        )
        cmax_rs = npz_1d(
            z,
            'cmax_rs',
            context='pred npz',
            n=int(n_traces),
            dtype=np.float32,
        )
        dt_npz = npz_scalar_float(z, 'dt_sec', context='pred npz')

        stage4_score_map = {
            'conf_prob1': pmax_psn,
            'conf_rs1': cmax_rs,
        }
        for k in cfg.score_keys_for_weight:
            if k in stage4_score_map:
                scores_weight[k] = stage4_score_map[k].astype(np.float32, copy=False)
                continue
            if k in z.files:
                scores_weight[k] = npz_1d(
                    z,
                    k,
                    context='pred npz',
                    n=int(n_traces),
                    dtype=np.float32,
                )
                continue
            msg = f'pred npz missing score key={k!r} for cfg.score_keys_for_weight'
            raise KeyError(msg)

    dt_in = float(dt_sec_in)
    if (not np.isfinite(dt_npz)) or dt_npz <= 0.0:
        msg = f'invalid dt_sec in pred npz: {dt_npz}'
        raise ValueError(msg)
    if (not np.isfinite(dt_in)) or dt_in <= 0.0:
        msg = f'invalid dt_sec_in from segy: {dt_in}'
        raise ValueError(msg)

    tol = 1e-7 * max(abs(dt_in), 1.0)
    if abs(dt_npz - dt_in) > tol:
        msg = (
            f'dt mismatch between segy and pred npz: segy={dt_in:.9g} sec, '
            f'npz={dt_npz:.9g} sec (tol={tol:.3g}). '
            f'Run stage4 with the same input segy.'
        )
        raise ValueError(msg)

    center_i_local = np.full(int(n_traces), np.nan, dtype=np.float32)
    local_ok = np.zeros(int(n_traces), dtype=bool)
    return Stage2Seed(
        pick_final=pick_final,
        scores_weight=scores_weight,
        trend_center_i_local=center_i_local,
        local_trend_ok=local_ok,
    )


__all__ = ['load_stage4_seed_from_pred_npz']
