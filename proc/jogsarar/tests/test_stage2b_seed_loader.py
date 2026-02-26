from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import stage2_make_psn512_windows as st2
from stage2b.io import load_stage4_seed_from_pred_npz


def test_load_stage4_seed_from_pred_npz_maps_expected_arrays(tmp_path) -> None:
    n_traces = 4
    pred_npz = tmp_path / 'dummy.psn_pred.npz'
    np.savez_compressed(
        pred_npz,
        pick_final=np.asarray([1, 2, 3, 4], dtype=np.int32),
        pmax_psn=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        cmax_rs=np.asarray([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        dt_sec=np.float32(0.002),
    )

    cfg = replace(st2.DEFAULT_STAGE2_CFG, score_keys_for_weight=('conf_prob1', 'conf_rs1'))
    seed = load_stage4_seed_from_pred_npz(
        pred_npz=pred_npz,
        n_traces=n_traces,
        dt_sec_in=0.002,
        cfg=cfg,
    )

    assert seed.pick_final.dtype == np.int64
    assert seed.pick_final.shape == (n_traces,)
    assert seed.scores_weight['conf_prob1'].dtype == np.float32
    assert seed.scores_weight['conf_rs1'].dtype == np.float32
    assert seed.scores_weight['conf_prob1'].shape == (n_traces,)
    assert seed.scores_weight['conf_rs1'].shape == (n_traces,)
    assert seed.trend_center_i_local.dtype == np.float32
    assert seed.local_trend_ok.dtype == bool
    assert np.all(np.isnan(seed.trend_center_i_local))
    assert not np.any(seed.local_trend_ok)


def test_load_stage4_seed_from_pred_npz_raises_on_dt_mismatch(tmp_path) -> None:
    pred_npz = tmp_path / 'dummy.psn_pred.npz'
    np.savez_compressed(
        pred_npz,
        pick_final=np.asarray([1, 2], dtype=np.int32),
        pmax_psn=np.asarray([0.1, 0.2], dtype=np.float32),
        cmax_rs=np.asarray([0.8, 0.7], dtype=np.float32),
        dt_sec=np.float32(0.004),
    )

    with pytest.raises(ValueError):
        load_stage4_seed_from_pred_npz(
            pred_npz=pred_npz,
            n_traces=2,
            dt_sec_in=0.002,
            cfg=st2.DEFAULT_STAGE2_CFG,
        )
