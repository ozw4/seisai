from __future__ import annotations

from pathlib import Path

from seisai_engine.tracking.data_id import build_data_manifest, calc_data_id


def _write_tmp_file(path: Path, content: str) -> None:
    path.write_text(content, encoding='utf-8')


def test_build_data_manifest_extracts_files(tmp_path: Path) -> None:
    f1 = tmp_path / 'a.txt'
    f2 = tmp_path / 'b.txt'
    _write_tmp_file(f1, 'a')
    _write_tmp_file(f2, 'b')

    cfg = {
        'paths': {
            'segy_files': [str(f1), str(f2)],
            'notes': [str(f1)],
            'out_dir': str(tmp_path / 'out'),
        }
    }
    manifest = build_data_manifest(cfg)
    files = manifest['files']
    assert len(files) == 2
    assert all(entry['key'] == 'segy_files' for entry in files)


def test_data_id_stable_across_order(tmp_path: Path) -> None:
    f1 = tmp_path / 'a.txt'
    f2 = tmp_path / 'b.txt'
    _write_tmp_file(f1, 'a')
    _write_tmp_file(f2, 'b')

    cfg1 = {'paths': {'segy_files': [str(f1), str(f2)]}}
    cfg2 = {'paths': {'segy_files': [str(f2), str(f1)]}}

    id1 = calc_data_id(build_data_manifest(cfg1))
    id2 = calc_data_id(build_data_manifest(cfg2))

    assert id1 == id2
