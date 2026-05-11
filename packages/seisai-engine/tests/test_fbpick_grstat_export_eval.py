from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
GRSTAT_MODULE_PATH = (
    REPO_ROOT
    / 'packages/seisai-engine/src/seisai_engine/pipelines/fbpick/export/grstat.py'
)


def _load_grstat_module():
    spec = importlib.util.spec_from_file_location(
        '_fbpick_export_grstat_test', GRSTAT_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load spec for {GRSTAT_MODULE_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_grstat(path: Path) -> None:
    path.write_text(
        '\n'.join(
            [
                '*****< grstat first-break time dump : ver.dec96 >***********************',
                '* rec.no.=    10',
                'fb            1    5   20   40-9999   80  100',
                '* rec.no.=    20',
                'fb            1    5   10   30   50   70   90',
                '*',
            ]
        )
        + '\n',
        encoding='utf-8',
    )


def test_load_grstat_matrix_preserves_rec_numbers_and_converts_to_samples(
    tmp_path: Path,
) -> None:
    mod = _load_grstat_module()
    path = tmp_path / 'ref.crd'
    _write_grstat(path)

    parsed = mod.load_grstat_matrix(path, dt_multiplier=2.0)

    assert parsed.record_numbers.tolist() == [10, 20]
    assert parsed.samples.tolist() == [
        [10, 20, 0, 40, 50],
        [5, 15, 25, 35, 45],
    ]


def test_evaluate_grstat_matrix_aligns_by_ffid_and_reports_errors(
    tmp_path: Path,
) -> None:
    mod = _load_grstat_module()
    path = tmp_path / 'ref.crd'
    _write_grstat(path)
    parsed = mod.load_grstat_matrix(path, dt_multiplier=2.0)
    pred = np.asarray(
        [
            [11, 20, 0, 38, 50],
            [5, 10, 25, 40, 45],
        ],
        dtype=np.int32,
    )

    summary, rows = mod.evaluate_grstat_matrix(
        prediction_samples=pred,
        prediction_ffids=[10, 20],
        reference=parsed,
        dt_multiplier=2.0,
    )

    assert summary['alignment_mode'] == 'record_number'
    assert summary['n_reference_valid'] == 9
    assert summary['n_eval'] == 9
    assert summary['R2'] == 7 / 9
    assert summary['mae_samples_max'] == 5
    assert len(rows) == 9
    assert rows[0]['ffid'] == 10
    assert rows[0]['chno'] == 1
    assert rows[0]['error_samples'] == 1
