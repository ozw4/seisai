from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
IO_GRSTAT_MODULE_PATH = (
    REPO_ROOT / 'packages/seisai-pick/src/seisai_pick/pickio/io_grstat.py'
)


def _load_io_grstat_module():
    spec = importlib.util.spec_from_file_location(
        '_seisai_pick_io_grstat_test', IO_GRSTAT_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load spec for {IO_GRSTAT_MODULE_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_numpy2fbcrd_writes_new_recno_channel_range_format(tmp_path: Path) -> None:
    mod = _load_io_grstat_module()
    out = tmp_path / 'new.crd'
    fb = np.asarray(
        [
            [46, 41, 36, 33, 28],
            [0, 15, 21, 28, 33],
        ],
        dtype=np.int32,
    )

    written = mod.numpy2fbcrd(
        dt=2.0,
        fbnum=fb,
        gather_range=[10, 20],
        output_name=str(out),
        output_format='recno_channel_range',
        values_per_line=3,
    )

    text = out.read_text(encoding='utf-8')
    assert '* rec.no.=' not in text
    assert 'fb:         10       1       3    92.000    82.000    72.000' in text
    assert 'fb:         20       1       3 -9999.000    30.000    42.000' in text

    parsed = mod.load_fb_irasformat(str(out), dt=2.0, verbose=False).reshape(2, 5)
    assert parsed.tolist() == fb.tolist()
    assert written.tolist() == [
        [92, 82, 72, 66, 56],
        [-9999, 30, 42, 56, 66],
    ]


def test_load_fb_irasformat_reads_new_reference_format(tmp_path: Path) -> None:
    mod = _load_io_grstat_module()
    path = tmp_path / 'reference.crd'
    path.write_text(
        '\n'.join(
            [
                '** GRSTAT ver.dec96a : first-break time dump ***********************************',
                '********************************************************************************',
                'fb:          1       1       3    92.000    82.000 -9999.000',
                'fb:          1       4       5    66.000    56.000',
                'fb:          2       1       5    10.000    30.000    42.000    56.000    66.000',
                '*',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    parsed = mod.load_fb_irasformat(str(path), dt=2.0, verbose=False).reshape(2, 5)

    assert parsed.tolist() == [
        [46, 41, 0, 33, 28],
        [5, 15, 21, 28, 33],
    ]


def test_numpy2fbcrd_can_still_write_legacy_format(tmp_path: Path) -> None:
    mod = _load_io_grstat_module()
    out = tmp_path / 'legacy.crd'
    fb = np.asarray([[10, 20, 0]], dtype=np.int32)

    mod.numpy2fbcrd(
        dt=2.0,
        fbnum=fb,
        gather_range=[7],
        output_name=str(out),
        output_format='legacy',
    )

    text = out.read_text(encoding='utf-8')
    assert '* rec.no.=    7' in text
    assert 'fb            1    3   20   40-9999' in text
    parsed = mod.load_fb_irasformat(str(out), dt=2.0, verbose=False).reshape(1, 3)
    assert parsed.tolist() == [[10, 20, 0]]


def test_load_grstat_matrix_preserves_record_numbers(tmp_path: Path) -> None:
    mod = _load_io_grstat_module()
    path = tmp_path / 'legacy_non_contiguous_recno.crd'
    path.write_text(
        '\n'.join(
            [
                '*****< grstat first-break time dump : ver.dec96 >***********************',
                '* rec.no.=   10',
                'fb            1    3   20   40-9999',
                '* rec.no.=   20',
                'fb            1    3   10   30   50',
                '*',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    parsed = mod.load_grstat_matrix(path, dt_multiplier=2.0)

    assert parsed.record_numbers.tolist() == [10, 20]
    assert parsed.samples.tolist() == [[10, 20, 0], [5, 15, 25]]
    assert parsed.raw_values.tolist() == [[20.0, 40.0, -9999.0], [10.0, 30.0, 50.0]]


def test_load_grstat_matrix_reads_new_format(tmp_path: Path) -> None:
    mod = _load_io_grstat_module()
    path = tmp_path / 'new.crd'
    path.write_text(
        '\n'.join(
            [
                '** GRSTAT ver.dec96a : first-break time dump ***********************************',
                'fb:         101       1       2    92.000    82.000',
                'fb:         101       3       3 -9999.000',
                'fb:         103       1       3    10.000    30.000    42.000',
                '*',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    parsed = mod.load_grstat_matrix(path, dt_multiplier=2.0)

    assert parsed.record_numbers.tolist() == [101, 103]
    assert parsed.samples.tolist() == [[46, 41, 0], [5, 15, 21]]


def test_convert_grstat_format_cli_preserves_rec_numbers(tmp_path: Path) -> None:
    mod = _load_io_grstat_module()
    script_path = REPO_ROOT / 'cli/convert_grstat_format.py'
    spec = importlib.util.spec_from_file_location('_convert_grstat_format_test', script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load spec for {script_path}')
    cli_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli_mod
    spec.loader.exec_module(cli_mod)

    legacy = tmp_path / 'legacy.crd'
    converted = tmp_path / 'converted.crd'
    legacy.write_text(
        '\n'.join(
            [
                '*****< grstat first-break time dump : ver.dec96 >***********************',
                '* rec.no.=   10',
                'fb            1    3   20   40-9999',
                '* rec.no.=   20',
                'fb            1    3   10   30   50',
                '*',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    cli_mod.main(
        [
            '--input-crd',
            str(legacy),
            '--output-crd',
            str(converted),
            '--dt-ms',
            '2.0',
            '--values-per-line',
            '2',
        ]
    )

    text = converted.read_text(encoding='utf-8')
    assert '* rec.no.=' not in text
    assert 'fb:         10       1       2    20.000    40.000' in text
    assert 'fb:         20       1       2    10.000    30.000' in text
    parsed = mod.load_grstat_matrix(converted, dt_multiplier=2.0)
    assert parsed.record_numbers.tolist() == [10, 20]
    assert parsed.samples.tolist() == [[10, 20, 0], [5, 15, 25]]


def test_load_grstat_matrix_preserves_record_numbers_for_legacy(tmp_path: Path) -> None:
    mod = _load_io_grstat_module()
    path = tmp_path / 'legacy_ref.crd'
    path.write_text(
        '\n'.join(
            [
                '*****< grstat first-break time dump : ver.dec96 >***********************',
                '* rec.no.=   101',
                'fb            1    3   20   40-9999',
                '* rec.no.=   205',
                'fb            1    3   10   30   50',
                '*',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    parsed = mod.load_grstat_matrix(path, dt_multiplier=2.0)

    assert parsed.record_numbers.tolist() == [101, 205]
    assert parsed.samples.tolist() == [[10, 20, 0], [5, 15, 25]]
    assert parsed.raw_values.tolist() == [[20.0, 40.0, -9999.0], [10.0, 30.0, 50.0]]


def test_load_grstat_matrix_allows_variable_channels_when_requested(tmp_path: Path) -> None:
    mod = _load_io_grstat_module()
    path = tmp_path / 'ragged.crd'
    path.write_text(
        '\n'.join(
            [
                'fb:          1       1       2    20.000    40.000',
                'fb:          2       1       3    10.000    30.000    50.000',
                '*',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    parsed = mod.load_grstat_matrix(
        path,
        dt_multiplier=2.0,
        strict_channel_count=False,
    )

    assert parsed.record_numbers.tolist() == [1, 2]
    assert parsed.samples.tolist() == [[10, 20, 0], [5, 15, 25]]

    try:
        mod.load_grstat_matrix(path, dt_multiplier=2.0, strict_channel_count=True)
    except ValueError as exc:
        assert 'channel count differs' in str(exc)
    else:
        raise AssertionError('strict_channel_count=True should reject ragged records')
