from __future__ import annotations

import re
from pathlib import Path

import yaml

from common.iter_layout import iter_tag
from common.paths import stage2_phase_pick_csr_npz_path


def _write_listfile(path: Path, *, values: list[Path]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for p in values:
            f.write(str(p.resolve()))
            f.write('\n')


def write_stage3_listfiles(*, stage2_out: Path, out_dir: Path) -> tuple[Path, Path]:
    stage2_root = Path(stage2_out).expanduser().resolve()
    if not stage2_root.is_dir():
        msg = f'stage2_out must be an existing directory: {stage2_root}'
        raise NotADirectoryError(msg)

    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    win_files = sorted(stage2_root.rglob('*.win512.sgy'))
    segy_files: list[Path] = []
    phase_pick_files: list[Path] = []
    for win_path in win_files:
        phase_pick_path = stage2_phase_pick_csr_npz_path(win_path)
        if not phase_pick_path.is_file():
            continue
        segy_files.append(win_path)
        phase_pick_files.append(phase_pick_path)

    segy_list = target_dir / 'segy_list.txt'
    phase_pick_list = target_dir / 'phase_pick_list.txt'
    _write_listfile(segy_list, values=segy_files)
    _write_listfile(phase_pick_list, values=phase_pick_files)
    return segy_list, phase_pick_list


def build_iter_stage3_config(
    *,
    base_config: Path,
    out_dir: Path,
    segy_list: Path,
    phase_pick_list: Path,
    iter_id: int | None,
) -> Path:
    cfg_path = Path(base_config).expanduser().resolve()
    if not cfg_path.is_file():
        msg = f'base_config not found: {cfg_path}'
        raise FileNotFoundError(msg)

    with cfg_path.open('r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        msg = f'config top-level must be dict, got {type(loaded).__name__}'
        raise TypeError(msg)

    paths = loaded.get('paths')
    if not isinstance(paths, dict):
        msg = 'config.paths must be dict'
        raise TypeError(msg)

    target_dir = Path(out_dir).expanduser().resolve()
    if target_dir.exists() and (not target_dir.is_dir()):
        msg = f'out_dir must be a directory path: {target_dir}'
        raise NotADirectoryError(msg)
    target_dir.mkdir(parents=True, exist_ok=True)

    paths['segy_files'] = str(Path(segy_list).expanduser().resolve())
    paths['phase_pick_files'] = str(Path(phase_pick_list).expanduser().resolve())
    paths['infer_segy_files'] = str(Path(segy_list).expanduser().resolve())
    paths['infer_phase_pick_files'] = str(Path(phase_pick_list).expanduser().resolve())
    paths['out_dir'] = str((target_dir / 'psn_train').resolve())

    if iter_id is not None:
        tracking = loaded.get('tracking')
        if isinstance(tracking, dict):
            exp_name = tracking.get('exp_name')
            if isinstance(exp_name, str) and exp_name:
                suffix = f'_{iter_tag(iter_id)}'
                has_iter_suffix = re.search(r'_iter\d+$', exp_name) is not None
                if (not exp_name.endswith(suffix)) and (not has_iter_suffix):
                    tracking['exp_name'] = f'{exp_name}{suffix}'

    out_cfg = target_dir / 'config_stage3_iter.yaml'
    with out_cfg.open('w', encoding='utf-8') as f:
        yaml.safe_dump(loaded, f, sort_keys=False)
    return out_cfg


__all__ = [
    'build_iter_stage3_config',
    'write_stage3_listfiles',
]
