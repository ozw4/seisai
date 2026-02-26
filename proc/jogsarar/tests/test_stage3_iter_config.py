from __future__ import annotations

from pathlib import Path

import yaml

from common.stage3_iter_config import build_iter_stage3_config, write_stage3_listfiles


def _read_lines(path: Path) -> list[str]:
    with path.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def test_write_stage3_listfiles_keeps_only_pairs(tmp_path: Path) -> None:
    stage2_out = tmp_path / 'stage2'
    keep_win = stage2_out / 'lineA' / 'shot01.win512.sgy'
    keep_win.parent.mkdir(parents=True, exist_ok=True)
    keep_win.touch()
    keep_pick = keep_win.with_suffix('.phase_pick.csr.npz')
    keep_pick.touch()

    skip_win = stage2_out / 'lineB' / 'shot02.win512.sgy'
    skip_win.parent.mkdir(parents=True, exist_ok=True)
    skip_win.touch()

    out_dir = tmp_path / 'stage3'
    segy_list, phase_list = write_stage3_listfiles(stage2_out=stage2_out, out_dir=out_dir)

    assert _read_lines(segy_list) == [str(keep_win.resolve())]
    assert _read_lines(phase_list) == [str(keep_pick.resolve())]


def test_build_iter_stage3_config_rewrites_paths_and_exp_name(tmp_path: Path) -> None:
    base_cfg = tmp_path / 'base.yaml'
    base_cfg.write_text(
        yaml.safe_dump(
            {
                'paths': {
                    'segy_files': 'a.txt',
                    'phase_pick_files': 'b.txt',
                    'infer_segy_files': 'c.txt',
                    'infer_phase_pick_files': 'd.txt',
                    'out_dir': './runs/base',
                },
                'tracking': {'exp_name': 'expA'},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    out_dir = tmp_path / 'iter03' / 'stage3'
    segy_list = out_dir / 'segy_list.txt'
    phase_list = out_dir / 'phase_pick_list.txt'
    out_dir.mkdir(parents=True, exist_ok=True)
    segy_list.write_text('/tmp/a.win512.sgy\n', encoding='utf-8')
    phase_list.write_text('/tmp/a.phase_pick.csr.npz\n', encoding='utf-8')

    out_cfg = build_iter_stage3_config(
        base_config=base_cfg,
        out_dir=out_dir,
        segy_list=segy_list,
        phase_pick_list=phase_list,
        iter_id=3,
    )

    with out_cfg.open('r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)

    assert loaded['paths']['segy_files'] == str(segy_list.resolve())
    assert loaded['paths']['phase_pick_files'] == str(phase_list.resolve())
    assert loaded['paths']['infer_segy_files'] == str(segy_list.resolve())
    assert loaded['paths']['infer_phase_pick_files'] == str(phase_list.resolve())
    assert loaded['paths']['out_dir'] == str((out_dir / 'psn_train').resolve())
    assert loaded['tracking']['exp_name'] == 'expA_iter03'


def test_build_iter_stage3_config_avoids_double_suffix(tmp_path: Path) -> None:
    base_cfg = tmp_path / 'base.yaml'
    base_cfg.write_text(
        yaml.safe_dump(
            {
                'paths': {
                    'segy_files': 'a.txt',
                    'phase_pick_files': 'b.txt',
                    'infer_segy_files': 'c.txt',
                    'infer_phase_pick_files': 'd.txt',
                    'out_dir': './runs/base',
                },
                'tracking': {'exp_name': 'expA_iter03'},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    out_dir = tmp_path / 'iter03' / 'stage3'
    out_dir.mkdir(parents=True, exist_ok=True)
    segy_list = out_dir / 'segy_list.txt'
    phase_list = out_dir / 'phase_pick_list.txt'
    segy_list.write_text('/tmp/a.win512.sgy\n', encoding='utf-8')
    phase_list.write_text('/tmp/a.phase_pick.csr.npz\n', encoding='utf-8')

    out_cfg = build_iter_stage3_config(
        base_config=base_cfg,
        out_dir=out_dir,
        segy_list=segy_list,
        phase_pick_list=phase_list,
        iter_id=3,
    )

    with out_cfg.open('r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)

    assert loaded['tracking']['exp_name'] == 'expA_iter03'
