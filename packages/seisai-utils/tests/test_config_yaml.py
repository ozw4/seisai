from __future__ import annotations

from pathlib import Path

import pytest

from seisai_utils.config_yaml import load_yaml


def test_load_yaml_resolves_target_list_values_relative_to_yaml_dir(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / 'cfg'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / 'config.yaml'
    cfg_path.write_text(
        '\n'.join(
            [
                'paths:',
                '  segy_files:',
                '    - data/train.sgy',
                '  phase_pick_files:',
                '    - picks/train.npz',
                '  infer_segy_files:',
                '    - data/infer.sgy',
                '  infer_phase_pick_files:',
                '    - picks/infer.npz',
                '  input_segy_files:',
                '    - pair/input.sgy',
                '  target_segy_files:',
                '    - pair/target.sgy',
                '  infer_input_segy_files:',
                '    - pair/infer_input.sgy',
                '  infer_target_segy_files:',
                '    - pair/infer_target.sgy',
                '  out_dir: ./out',
                'other:',
                '  value: keep-me',
            ]
        ),
        encoding='utf-8',
    )

    cfg = load_yaml(cfg_path)

    assert cfg['paths']['segy_files'] == [str((cfg_dir / 'data/train.sgy').resolve())]
    assert cfg['paths']['phase_pick_files'] == [
        str((cfg_dir / 'picks/train.npz').resolve())
    ]
    assert cfg['paths']['infer_segy_files'] == [
        str((cfg_dir / 'data/infer.sgy').resolve())
    ]
    assert cfg['paths']['infer_phase_pick_files'] == [
        str((cfg_dir / 'picks/infer.npz').resolve())
    ]
    assert cfg['paths']['input_segy_files'] == [
        str((cfg_dir / 'pair/input.sgy').resolve())
    ]
    assert cfg['paths']['target_segy_files'] == [
        str((cfg_dir / 'pair/target.sgy').resolve())
    ]
    assert cfg['paths']['infer_input_segy_files'] == [
        str((cfg_dir / 'pair/infer_input.sgy').resolve())
    ]
    assert cfg['paths']['infer_target_segy_files'] == [
        str((cfg_dir / 'pair/infer_target.sgy').resolve())
    ]
    assert cfg['paths']['out_dir'] == './out'
    assert cfg['other']['value'] == 'keep-me'


def test_load_yaml_resolves_target_str_value_and_keeps_absolute_value(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / 'cfg'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    absolute = str((tmp_path / 'already-absolute.sgy').resolve())
    cfg_path = cfg_dir / 'config.yaml'
    cfg_path.write_text(
        '\n'.join(
            [
                'paths:',
                '  segy_files: lists/train.txt',
                f'  infer_segy_files: "{absolute}"',
            ]
        ),
        encoding='utf-8',
    )

    cfg = load_yaml(cfg_path)

    assert cfg['paths']['segy_files'] == str((cfg_dir / 'lists/train.txt').resolve())
    assert cfg['paths']['infer_segy_files'] == absolute


def test_load_yaml_invalid_target_type_raises_type_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text(
        '\n'.join(
            [
                'paths:',
                '  segy_files: 1',
            ]
        ),
        encoding='utf-8',
    )

    with pytest.raises(TypeError, match='config.paths.segy_files must be str or list'):
        load_yaml(cfg_path)
