from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import segyio


def _load_stage2_module():
    repo_root = Path(__file__).resolve().parents[1]
    stage2_path = repo_root / 'proc/jogsarar/stage2_make_psn512_windows.py'

    module_dir = str(stage2_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(
        'jogsarar_stage2_make_psn512_windows_test',
        stage2_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage2_weight_keys_exclude_conf_trend1() -> None:
    stage2 = _load_stage2_module()

    assert tuple(stage2.SCORE_KEYS_FOR_WEIGHT) == ('conf_prob1', 'conf_rs1')
    assert 'conf_trend1' not in tuple(stage2.SCORE_KEYS_FOR_WEIGHT)


def test_stage2_runs_without_stage1_conf_trend_keys(tmp_path: Path) -> None:
    stage2 = _load_stage2_module()

    segy_path = Path('test_data/ridgecrest_das/20200623002546.sgy').resolve()
    assert segy_path.exists()

    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
        n_traces = int(src.tracecount)
        n_samples = int(src.samples.size)

    infer_root = tmp_path / 'infer'
    out_root = tmp_path / 'out'
    infer_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # Stage1新仕様: conf_trend*/trend_* は未保存。Stage2 が最小キーで処理できることを確認。
    pick_final = np.full(n_traces, 2, dtype=np.int64)
    conf_prob1 = np.full(n_traces, 0.8, dtype=np.float32)
    conf_rs1 = np.full(n_traces, 0.9, dtype=np.float32)

    infer_npz = infer_root / f'{segy_path.stem}.prob.npz'
    np.savez_compressed(
        infer_npz,
        pick_final=pick_final,
        conf_prob1=conf_prob1,
        conf_rs1=conf_rs1,
    )

    stage2.IN_SEGY_ROOT = segy_path.parent
    stage2.IN_INFER_ROOT = infer_root
    stage2.OUT_SEGY_ROOT = out_root
    stage2.THRESH_MODE = 'per_segy'

    stage2.process_one_segy(segy_path, global_thresholds=None)

    out_segy = out_root / f'{segy_path.stem}.win512.sgy'
    sidecar = out_segy.with_suffix('.sidecar.npz')

    assert out_segy.exists()
    assert sidecar.exists()

    with np.load(sidecar, allow_pickle=False) as z:
        assert 'semi_low_mask' in z.files
        assert 'semi_covered' in z.files
        assert 'semi_support_count' in z.files
        assert 'semi_v_trend' in z.files
        assert 'global_missing_filled_mask' in z.files
        assert 'trend_center_i_final' in z.files
        assert 'conf_trend1' in z.files
        assert 'th_conf_trend1' in z.files

        assert np.asarray(z['semi_low_mask']).shape == (n_traces,)
        assert np.asarray(z['conf_trend1']).shape == (n_traces,)
