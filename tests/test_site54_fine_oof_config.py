from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml


def _load_make_fine_fold_configs() -> ModuleType:
    return _load_oof_script(
        'make_fine_fold_configs.py',
        'site54_make_fine_fold_configs',
    )


def _load_make_coarse_fold_configs() -> ModuleType:
    return _load_oof_script(
        'make_coarse_fold_configs.py',
        'site54_make_coarse_fold_configs',
    )


def _load_make_physics_fold_configs() -> ModuleType:
    return _load_oof_script(
        'make_physics_fold_configs.py',
        'site54_make_physics_fold_configs',
    )


def _load_oof_script(script_name: str, module_name: str) -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / 'proc'
        / 'fbpick'
        / 'site54'
        / 'oof'
        / 'scripts'
        / script_name
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
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


def test_make_fine_defaults_write_under_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_make_fine_fold_configs()
    cv_root = tmp_path / 'oof'
    run_id = 'run-x'
    run_root = tmp_path / 'custom_runs' / run_id
    fold_root = cv_root / 'fold_lists' / 'folds'
    oof_list_dir = run_root / 'aggregate' / '05_collect_oof_lists'
    data_dir = tmp_path / 'data'
    template_dir = cv_root / 'config_templates'
    base_train = template_dir / 'fine_train.yaml'
    base_infer = template_dir / 'fine_infer.yaml'

    sgys = [str(data_dir / f'survey{i}.sgy') for i in range(6)]
    fbs = [str(data_dir / f'survey{i}.fb.npy') for i in range(6)]
    robust = [str(data_dir / f'survey{i}.robust.npz') for i in range(6)]
    coarse = [str(data_dir / f'survey{i}.coarse.npz') for i in range(6)]

    for i in range(6):
        fold_dir = fold_root / f'fold{i:02d}'
        fold_dir.mkdir(parents=True)
        (fold_dir / 'heldout_sgy.txt').write_text(sgys[i] + '\n', encoding='utf-8')
        (fold_dir / 'heldout_fb.txt').write_text(fbs[i] + '\n', encoding='utf-8')

    oof_list_dir.mkdir(parents=True)
    (oof_list_dir / 'oof_train_sgy_all.txt').write_text(
        '\n'.join(sgys) + '\n',
        encoding='utf-8',
    )
    (oof_list_dir / 'oof_train_fb_all.txt').write_text(
        '\n'.join(fbs) + '\n',
        encoding='utf-8',
    )
    (oof_list_dir / 'oof_train_robust_all.txt').write_text(
        '\n'.join(robust) + '\n',
        encoding='utf-8',
    )
    (oof_list_dir / 'oof_train_coarse_all.txt').write_text(
        '\n'.join(coarse) + '\n',
        encoding='utf-8',
    )
    template_dir.mkdir(parents=True)
    base_train.write_text(
        'paths: {}\ntrain: {}\ninfer: {}\nvis: {}\n',
        encoding='utf-8',
    )
    base_infer.write_text('paths: {}\ninfer: {}\n', encoding='utf-8')

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'make_fine_fold_configs.py',
            '--repo-root',
            str(tmp_path),
            '--cv-root',
            str(cv_root),
            '--run-id',
            run_id,
            '--run-root',
            str(run_root),
            '--fine-inner-valid-size',
            '1',
        ],
    )

    assert module.main() == 0
    assert (run_root / 'configs' / 'fold00' / '06_fine_train.yaml').is_file()
    train_cfg = yaml.safe_load(
        (run_root / 'configs' / 'fold00' / '06_fine_train.yaml').read_text(
            encoding='utf-8',
        ),
    )
    for key, value in train_cfg['paths'].items():
        assert 'heldout_' not in str(value), (key, value)
    assert (
        run_root
        / 'aggregate'
        / '05_collect_oof_lists'
        / 'fine_fold_lists'
        / 'fold00'
        / 'heldout_sgy.txt'
    ).is_file()
    assert not (cv_root / 'configs').exists()
    assert not (cv_root / 'lists').exists()
    assert not (cv_root / 'runs' / run_id).exists()


def test_make_coarse_defaults_write_configs_under_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_make_coarse_fold_configs()
    cv_root = tmp_path / 'oof'
    run_id = 'run-x'
    run_root = cv_root / 'runs' / run_id
    (cv_root / 'fold_lists' / 'folds').mkdir(parents=True)

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'make_coarse_fold_configs.py',
            '--cv-root',
            str(cv_root),
            '--run-id',
            run_id,
        ],
    )

    module.main()

    assert (run_root / 'configs' / 'fold00' / '01_coarse_train.yaml').is_file()
    assert (run_root / 'configs' / 'fold00' / '01_coarse_train_smoke.yaml').is_file()
    assert (run_root / 'configs' / 'fold00' / '02_coarse_infer.yaml').is_file()
    assert not (cv_root / 'configs').exists()
    assert not (run_root / 'configs' / run_id).exists()


def test_make_physics_defaults_write_configs_under_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_make_physics_fold_configs()
    cv_root = tmp_path / 'oof'
    run_id = 'run-x'
    run_root = cv_root / 'runs' / run_id
    fold_root = cv_root / 'fold_lists' / 'folds'

    for i in range(6):
        fold_dir = fold_root / f'fold{i:02d}'
        fold_dir.mkdir(parents=True)
        (fold_dir / 'heldout_sgy.txt').write_text(
            '/data/survey.sgy\n',
            encoding='utf-8',
        )
        (fold_dir / 'heldout_fb.txt').write_text(
            '/data/survey.fb.npy\n',
            encoding='utf-8',
        )

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'make_physics_fold_configs.py',
            '--cv-root',
            str(cv_root),
            '--run-id',
            run_id,
        ],
    )

    module.main()

    assert (run_root / 'configs' / 'fold00' / '03_physics.yaml').is_file()
    assert (run_root / 'configs' / 'fold00' / '04_physics_qc.yaml').is_file()
    assert not (cv_root / 'configs').exists()
    assert not (run_root / 'configs' / run_id).exists()
