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





def _write_new_grstat(path: Path) -> None:
    path.write_text(
        '\n'.join(
            [
                '** GRSTAT ver.dec96a : first-break time dump ***********************************',
                '********************************************************************************',
                'fb:         10       1       5    20.000    40.000 -9999.000    80.000   100.000',
                'fb:         20       1       5    10.000    30.000    50.000    70.000    90.000',
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





def test_load_grstat_matrix_reads_new_recno_channel_range_format(tmp_path: Path) -> None:
    mod = _load_grstat_module()
    path = tmp_path / 'ref_new.crd'
    _write_new_grstat(path)

    parsed = mod.load_grstat_matrix(path, dt_multiplier=2.0)

    assert parsed.record_numbers.tolist() == [10, 20]
    assert parsed.samples.tolist() == [
        [10, 20, 0, 40, 50],
        [5, 15, 25, 35, 45],
    ]
    assert float(parsed.raw_values[0, 0]) == 20.0


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


def test_evaluate_export_npz_against_grstat_reuses_prediction_artifact(
    tmp_path: Path,
) -> None:
    mod = _load_grstat_module()
    ref_path = tmp_path / 'ref_new.crd'
    _write_new_grstat(ref_path)

    export_npz = tmp_path / 'prediction_export.npz'
    np.savez_compressed(
        export_npz,
        fb_mat_samples=np.asarray(
            [
                [10, 20, 0, 40, 50],
                [5, 14, 25, 35, 45],
            ],
            dtype=np.int32,
        ),
        gather_range_ffids=np.asarray([10, 20], dtype=np.int32),
        dt_multiplier=np.asarray(2.0, dtype=np.float64),
        summary_json=np.asarray('{"out_crd":"pred.crd"}'),
    )

    summary_json = tmp_path / 'eval.json'
    summary_csv = tmp_path / 'eval.csv'
    per_trace_csv = tmp_path / 'eval_rows.csv'
    summary = mod.evaluate_export_npz_against_grstat(
        export_npz_path=export_npz,
        reference_grstat_path=ref_path,
        eval_summary_json_path=summary_json,
        eval_summary_csv_path=summary_csv,
        eval_per_trace_csv_path=per_trace_csv,
    )

    assert summary['prediction_npz'] == str(export_npz.resolve())
    assert summary['prediction_crd'] == 'pred.crd'
    assert summary['n_eval'] == 9
    assert summary['mae_samples_max'] == 1
    assert summary_json.is_file()
    assert summary_csv.is_file()
    assert per_trace_csv.is_file()
