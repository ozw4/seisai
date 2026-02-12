from __future__ import annotations

from pathlib import Path

from seisai_engine.pipelines.common.config_io import load_config
from seisai_engine.pipelines.common.listfiles import expand_cfg_listfiles


def _write_file(path: Path, text: str = 'x') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def test_load_config_resolves_relative_paths_from_yaml_dir(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_dir = tmp_path / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = cfg_dir / 'config.yaml'
    cfg_path.write_text(
        '\n'.join(
            [
                'paths:',
                '  segy_files:',
                '    - data/train.sgy',
                '  phase_pick_files:',
                '    - data/train.npz',
                '  infer_segy_files:',
                '    - data/infer.sgy',
                '  infer_phase_pick_files:',
                '    - data/infer.npz',
                '  out_dir: ./out',
            ]
        ),
        encoding='utf-8',
    )

    other_cwd = tmp_path / 'elsewhere'
    other_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_cwd)

    cfg = load_config(cfg_path)

    assert cfg['paths']['segy_files'] == [str((cfg_dir / 'data/train.sgy').resolve())]
    assert cfg['paths']['phase_pick_files'] == [
        str((cfg_dir / 'data/train.npz').resolve())
    ]
    assert cfg['paths']['infer_segy_files'] == [
        str((cfg_dir / 'data/infer.sgy').resolve())
    ]
    assert cfg['paths']['infer_phase_pick_files'] == [
        str((cfg_dir / 'data/infer.npz').resolve())
    ]
    assert cfg['paths']['out_dir'] == './out'


def test_load_config_keeps_str_listfile_then_expand_works_from_other_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_dir = tmp_path / 'configs'
    data_file = cfg_dir / 'data' / 'gather.sgy'
    listfile = cfg_dir / 'lists' / 'paths.txt'
    _write_file(data_file)
    listfile.parent.mkdir(parents=True, exist_ok=True)
    listfile.write_text('../data/gather.sgy\n', encoding='utf-8')

    cfg_path = cfg_dir / 'config.yaml'
    cfg_path.write_text(
        '\n'.join(
            [
                'paths:',
                '  segy_files: lists/paths.txt',
            ]
        ),
        encoding='utf-8',
    )

    other_cwd = tmp_path / 'elsewhere'
    other_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_cwd)

    cfg = load_config(cfg_path)
    assert cfg['paths']['segy_files'] == str(listfile.resolve())

    expand_cfg_listfiles(cfg, keys=['paths.segy_files'])
    assert cfg['paths']['segy_files'] == [str(data_file.resolve())]
