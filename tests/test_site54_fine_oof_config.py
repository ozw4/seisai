from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_make_fine_fold_configs() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / 'proc'
        / 'fbpick'
        / 'site54'
        / 'oof'
        / 'scripts'
        / 'make_fine_fold_configs.py'
    )
    spec = importlib.util.spec_from_file_location(
        'site54_make_fine_fold_configs',
        script,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fine_train_config_fixed_last_uses_last_checkpoint_policy(
    tmp_path: Path,
) -> None:
    module = _load_make_fine_fold_configs()
    paths = {
        'train_sgy': tmp_path / 'train_sgy.txt',
        'train_fb': tmp_path / 'train_fb.txt',
        'train_robust': tmp_path / 'train_robust.txt',
        'inner_valid_sgy': tmp_path / 'inner_valid_sgy.txt',
        'inner_valid_fb': tmp_path / 'inner_valid_fb.txt',
        'inner_valid_robust': tmp_path / 'inner_valid_robust.txt',
        'heldout_sgy': tmp_path / 'heldout_sgy.txt',
        'heldout_fb': tmp_path / 'heldout_fb.txt',
        'heldout_robust': tmp_path / 'heldout_robust.txt',
    }

    cfg = module.fine_train_config(
        base_cfg={'paths': {}, 'ckpt': {}},
        paths=paths,
        out_dir=tmp_path / 'run' / 'fold00' / '06_fine_train',
        policy='fixed_last',
    )

    assert cfg['paths']['infer_segy_files'] == str(paths['train_sgy'])
    assert cfg['paths']['infer_fb_files'] == str(paths['train_fb'])
    assert cfg['paths']['infer_robust_npz_files'] == str(paths['train_robust'])
    assert cfg['ckpt'] == {
        'save_best_only': False,
        'metric': 'last',
        'mode': 'max',
    }


def test_fine_infer_config_omits_fb_files_for_raw_only_runtime(
    tmp_path: Path,
) -> None:
    module = _load_make_fine_fold_configs()
    paths = {
        'heldout_sgy': tmp_path / 'heldout_sgy.txt',
        'heldout_fb': tmp_path / 'heldout_fb.txt',
        'heldout_robust': tmp_path / 'heldout_robust.txt',
        'heldout_coarse': tmp_path / 'heldout_coarse.txt',
    }

    cfg = module.fine_infer_config(
        base_cfg={'paths': {}, 'infer': {}},
        paths=paths,
        out_dir=tmp_path / 'run' / 'fold00' / '07_fine_infer',
        ckpt_path=tmp_path / 'run' / 'fold00' / '06_fine_train' / 'ckpt' / 'best.pt',
    )

    assert 'fb_files' not in cfg['paths']
    assert cfg['paths']['segy_files'] == str(paths['heldout_sgy'])
    assert cfg['paths']['robust_npz_files'] == str(paths['heldout_robust'])
    assert cfg['paths']['coarse_npz_files'] == str(paths['heldout_coarse'])
