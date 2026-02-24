from __future__ import annotations

from pathlib import Path

import pytest

from seisai_engine.pipelines.common.listfiles import (
    expand_cfg_listfiles,
    get_cfg_listfile_meta,
    load_path_listfile,
    load_path_listfile_with_meta,
)


def _write_file(path: Path, text: str = 'x') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def test_load_path_listfile_ignores_empty_and_comment_lines(tmp_path: Path) -> None:
    f1 = tmp_path / 'a.sgy'
    f2 = tmp_path / 'b.sgy'
    _write_file(f1)
    _write_file(f2)

    listfile = tmp_path / 'list.txt'
    listfile.write_text(
        f'\n# comment\n  # indented\n{f1}\n\n{f2}\n', encoding='utf-8'
    )

    paths = load_path_listfile(listfile)
    assert paths == [str(f1.resolve()), str(f2.resolve())]


def test_load_path_listfile_resolves_relative_to_listfile_dir(
    tmp_path: Path,
) -> None:
    list_dir = tmp_path / 'lists'
    data_dir = list_dir / 'data'
    f1 = data_dir / 'a.sgy'
    _write_file(f1)

    listfile = list_dir / 'paths.txt'
    listfile.write_text('data/a.sgy\n', encoding='utf-8')

    paths = load_path_listfile(listfile)
    assert paths == [str(f1.resolve())]


def test_load_path_listfile_expands_env_and_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / 'env'
    env_file = env_dir / 'env.sgy'
    _write_file(env_file)

    home_dir = tmp_path / 'home'
    home_file = home_dir / 'home.sgy'
    _write_file(home_file)

    monkeypatch.setenv('LISTFILES_DIR', str(env_dir))
    monkeypatch.setenv('HOME', str(home_dir))

    listfile = tmp_path / 'paths.txt'
    listfile.write_text('$LISTFILES_DIR/env.sgy\n~/home.sgy\n', encoding='utf-8')

    paths = load_path_listfile(listfile)
    assert paths == [str(env_file.resolve()), str(home_file.resolve())]


def test_load_path_listfile_empty_raises_value_error(tmp_path: Path) -> None:
    listfile = tmp_path / 'empty.txt'
    listfile.write_text('# comment only\n\n', encoding='utf-8')

    with pytest.raises(ValueError):
        load_path_listfile(listfile)


def test_load_path_listfile_missing_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_path_listfile(tmp_path / 'missing.txt')


def test_load_path_listfile_missing_entry_raises_file_not_found(
    tmp_path: Path,
) -> None:
    listfile = tmp_path / 'paths.txt'
    listfile.write_text('missing.sgy\n', encoding='utf-8')

    with pytest.raises(FileNotFoundError):
        load_path_listfile(listfile)


def test_load_path_listfile_directory_entry_raises_value_error(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / 'data_dir'
    data_dir.mkdir(parents=True, exist_ok=True)

    listfile = tmp_path / 'paths.txt'
    listfile.write_text(f'{data_dir}\n', encoding='utf-8')

    with pytest.raises(ValueError):
        load_path_listfile(listfile)


def test_expand_cfg_listfiles_dot_path_inplace(tmp_path: Path) -> None:
    data_file = tmp_path / 'data.sgy'
    _write_file(data_file)

    listfile = tmp_path / 'paths.txt'
    listfile.write_text(f'{data_file}\n', encoding='utf-8')

    cfg = {'paths': {'segy_files': str(listfile)}}
    out = expand_cfg_listfiles(cfg, keys=['paths.segy_files'])

    assert out is cfg
    assert cfg['paths']['segy_files'] == [str(data_file.resolve())]


def test_load_path_listfile_with_tab_json_metadata(tmp_path: Path) -> None:
    f1 = tmp_path / 'a.sgy'
    f2 = tmp_path / 'b.sgy'
    _write_file(f1)
    _write_file(f2)

    listfile = tmp_path / 'paths.txt'
    listfile.write_text(
        '\n'.join(
            [
                (
                    f'{f1}\t'
                    '{"primary_keys":["ffid"],"primary_ranges":{"ffid":[[1,100]]}}'
                ),
                str(f2),
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    paths, metas = load_path_listfile_with_meta(listfile)
    assert paths == [str(f1.resolve()), str(f2.resolve())]
    assert metas == [
        {'primary_keys': ['ffid'], 'primary_ranges': {'ffid': [[1, 100]]}},
        None,
    ]


def test_load_path_listfile_with_tab_invalid_json_raises(tmp_path: Path) -> None:
    f1 = tmp_path / 'a.sgy'
    _write_file(f1)
    listfile = tmp_path / 'paths.txt'
    listfile.write_text(f'{f1}\t{{invalid json}}\n', encoding='utf-8')

    with pytest.raises(ValueError, match='invalid metadata json'):
        load_path_listfile_with_meta(listfile)


def test_expand_cfg_listfiles_stores_metadata_for_key(tmp_path: Path) -> None:
    f1 = tmp_path / 'a.sgy'
    f2 = tmp_path / 'b.sgy'
    _write_file(f1)
    _write_file(f2)
    listfile = tmp_path / 'paths.txt'
    listfile.write_text(
        '\n'.join(
            [
                f'{f1}\t{{"secondary_key_fixed":{{"ffid":true}}}}',
                str(f2),
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    cfg = {'paths': {'segy_files': str(listfile)}}
    expand_cfg_listfiles(cfg, keys=['paths.segy_files'])

    assert cfg['paths']['segy_files'] == [str(f1.resolve()), str(f2.resolve())]
    assert get_cfg_listfile_meta(cfg, key_path='paths.segy_files') == [
        {'secondary_key_fixed': {'ffid': True}},
        None,
    ]
