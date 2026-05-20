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


def _load_run_site54_oof_cv() -> ModuleType:
    return _load_oof_script(
        'run_site54_oof_cv.py',
        'site54_run_site54_oof_cv',
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


def _assert_partial_physics_fallback(cfg: dict) -> None:
    runtime = cfg['physical_runtime']
    assert runtime['trend_result_mode'] == 'lazy'
    assert runtime['fallback_existing_trend_mode'] == 'partial'
    partial = runtime['partial_trend_fallback']
    assert partial['enabled'] is True
    assert partial['max_fraction'] == 0.05
    assert partial['max_traces'] == 50000
    assert partial['cluster_consecutive_indices'] is True
    assert partial['use_global_fallback'] is True
    assert partial['fallback_if_too_many'] == 'full'
    assert partial['local_window_from_trend_config'] is True
    assert partial['emit_progress'] is True
    assert runtime['geometry_invalid_fallback'] == 'neighbor_or_coarse_in_band'
    assert runtime['group_invalid_fallback'] == 'neighbor_or_coarse_in_band'
    assert runtime['fine_window_constraint'][
        'allow_robust_fallback_as_fine_center'
    ] is False
    assert runtime['fine_window_constraint'][
        'allow_feasible_clip_as_fine_center'
    ] is False
    assert runtime['neighbor_physical_fit_reuse']['enabled'] is True
    assert runtime['fallback_policy']['enabled'] is True
    assert runtime['fallback_policy']['order'] == [
        'self_physical_fit',
        'neighbor_physical_fit_reuse',
        'coarse_in_band',
        'reject',
    ]


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
    cfg = module.fine_infer_config(
        base_cfg={'paths': {}, 'infer': {}, 'viewer': {'first_panel_only': True}},
        segy_file='/data/heldout.sgy',
        robust_npz_file='/data/heldout.robust.npz',
        coarse_npz_file='/data/heldout.coarse.npz',
        out_dir=tmp_path / 'run' / 'fold00' / '07_fine_infer',
        ckpt_path=tmp_path / 'run' / 'fold00' / '06_fine_train' / 'ckpt' / 'best.pt',
    )

    assert 'fb_files' not in cfg['paths']
    assert cfg['paths']['segy_files'] == ['/data/heldout.sgy']
    assert 'viewer_fb_files' not in cfg['paths']
    assert cfg['paths']['robust_npz_files'] == ['/data/heldout.robust.npz']
    assert cfg['paths']['coarse_npz_files'] == ['/data/heldout.coarse.npz']
    assert cfg['viewer']['first_panel_only'] is True


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

    sgys = [str(data_dir / f'survey{i}.sgy') for i in range(7)]
    fbs = [str(data_dir / f'survey{i}.fb.npy') for i in range(7)]
    robust = [str(data_dir / f'survey{i}.robust.npz') for i in range(7)]
    coarse = [str(data_dir / f'survey{i}.coarse.npz') for i in range(7)]

    for i in range(6):
        fold_dir = fold_root / f'fold{i:02d}'
        fold_dir.mkdir(parents=True)
        heldout_indices = [0, 1] if i == 0 else [i + 1]
        (fold_dir / 'heldout_sgy.txt').write_text(
            '\n'.join(sgys[idx] for idx in heldout_indices) + '\n',
            encoding='utf-8',
        )
        (fold_dir / 'heldout_fb.txt').write_text(
            '\n'.join(fbs[idx] for idx in heldout_indices) + '\n',
            encoding='utf-8',
        )

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
    base_infer.write_text(
        'paths: {}\ninfer: {}\nviewer:\n  gather_selection: even\n',
        encoding='utf-8',
    )

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
    assert (run_root / 'configs' / 'fold00' / '07_fine_infer.yaml').is_file()
    assert (run_root / 'configs' / 'fold00' / '07_fine_infer_001.yaml').is_file()
    train_cfg = yaml.safe_load(
        (run_root / 'configs' / 'fold00' / '06_fine_train.yaml').read_text(
            encoding='utf-8',
        ),
    )
    infer_cfg = yaml.safe_load(
        (run_root / 'configs' / 'fold00' / '07_fine_infer.yaml').read_text(
            encoding='utf-8',
        ),
    )
    for key, value in train_cfg['paths'].items():
        assert 'heldout_' not in str(value), (key, value)
    assert infer_cfg['paths']['segy_files'] == [sgys[0]]
    assert infer_cfg['paths']['viewer_fb_files'] == [fbs[0]]
    assert infer_cfg['paths']['robust_npz_files'] == [robust[0]]
    assert infer_cfg['paths']['coarse_npz_files'] == [coarse[0]]
    assert infer_cfg['viewer']['gather_selection'] == 'even'
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
    train_text = (run_root / 'configs' / 'fold00' / '01_coarse_train.yaml').read_text(
        encoding='utf-8',
    )
    train_cfg = yaml.safe_load(train_text)
    assert train_cfg['fbgate']['train']['apply_on'] == 'off'
    assert train_cfg['fbgate']['train']['min_pick_ratio'] == 0.01
    assert train_cfg['fbgate']['infer']['apply_on'] == 'off'
    assert train_cfg['fbgate']['infer']['min_pick_ratio'] == 0.01
    assert 'min_pick_ratio: 0.3' not in train_text


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
    physics_cfg = yaml.safe_load(
        (run_root / 'configs' / 'fold00' / '03_physics.yaml').read_text(
            encoding='utf-8',
        ),
    )
    physics_qc_cfg = yaml.safe_load(
        (run_root / 'configs' / 'fold00' / '04_physics_qc.yaml').read_text(
            encoding='utf-8',
        ),
    )
    _assert_partial_physics_fallback(physics_cfg)
    _assert_partial_physics_fallback(physics_qc_cfg)
    assert physics_qc_cfg['vis']['first_panel_only'] is True
    assert physics_qc_cfg['vis']['auto_figsize'] is True
    assert physics_qc_cfg['vis']['max_display_traces'] == 1200
    assert physics_qc_cfg['vis']['gather_selection'] == 'even'
    assert physics_qc_cfg['vis']['overlays']['gt_pick'] is True
    assert physics_qc_cfg['vis']['overlays']['robust_pick'] is False


def _write_runner_fold_lists(cv_root: Path) -> None:
    fold_root = cv_root / 'fold_lists' / 'folds'
    heldout_index = 0
    for i in range(6):
        fold = f'fold{i:02d}'
        fold_dir = fold_root / fold
        fold_dir.mkdir(parents=True)
        train_sgy = [f'/data/{fold}/train{j}.sgy' for j in range(2)]
        train_fb = [f'/data/{fold}/train{j}.fb.npy' for j in range(2)]
        valid_sgy = [f'/data/{fold}/valid{j}.sgy' for j in range(1)]
        valid_fb = [f'/data/{fold}/valid{j}.fb.npy' for j in range(1)]
        heldout_sgy = [
            f'/data/{fold}/heldout{heldout_index + j}.sgy' for j in range(9)
        ]
        heldout_fb = [
            f'/data/{fold}/heldout{heldout_index + j}.fb.npy' for j in range(9)
        ]
        heldout_index += 9
        for name, values in {
            'train_sgy.txt': train_sgy,
            'train_fb.txt': train_fb,
            'inner_valid_sgy.txt': valid_sgy,
            'inner_valid_fb.txt': valid_fb,
            'heldout_sgy.txt': heldout_sgy,
            'heldout_fb.txt': heldout_fb,
        }.items():
            (fold_dir / name).write_text(
                '\n'.join(values) + '\n',
                encoding='utf-8',
            )


def test_run_site54_oof_cv_dry_run_uses_run_scoped_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_run_site54_oof_cv()
    cv_root = tmp_path / 'oof'
    run_id = 'rerun_probe'

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'run_site54_oof_cv.py',
            '--repo-root',
            str(tmp_path),
            '--cv-root',
            str(cv_root),
            '--run-id',
            run_id,
            '--stage',
            'all',
            '--dry-run',
        ],
    )

    assert module.main() == 0
    out = capsys.readouterr().out
    assert f'RUN_ROOT={cv_root / "runs" / run_id}' in out
    assert f'CONFIG_ROOT={cv_root / "runs" / run_id / "configs"}' in out
    assert str(cv_root / 'runs' / run_id / 'configs' / 'fold00') in out
    assert str(
        cv_root
        / 'runs'
        / run_id
        / 'aggregate'
        / '05_collect_oof_lists'
        / 'fine_fold_lists'
    ) in out
    assert str(cv_root / 'configs') not in out
    assert str(cv_root / 'lists') not in out
    assert str(cv_root / 'logs') not in out
    assert not (cv_root / 'runs' / run_id / 'manifest.yaml').exists()


def test_run_site54_oof_cv_prepare_configs_writes_manifest_and_configs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_run_site54_oof_cv()
    cv_root = tmp_path / 'oof'
    run_id = 'rerun_probe'
    run_root = cv_root / 'runs' / run_id
    _write_runner_fold_lists(cv_root)

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'run_site54_oof_cv.py',
            '--repo-root',
            str(tmp_path),
            '--cv-root',
            str(cv_root),
            '--run-id',
            run_id,
            '--stage',
            'prepare_configs',
        ],
    )

    assert module.main() == 0
    assert (run_root / 'manifest.yaml').is_file()
    assert (run_root / 'configs' / 'fold00' / '01_coarse_train.yaml').is_file()
    assert (run_root / 'configs' / 'fold00' / '02_coarse_infer.yaml').is_file()
    assert (run_root / 'configs' / 'fold00' / '03_physics.yaml').is_file()
    assert (run_root / 'configs' / 'fold00' / '04_physics_qc.yaml').is_file()
    manifest = yaml.safe_load(
        (run_root / 'manifest.yaml').read_text(encoding='utf-8'),
    )
    assert manifest['run_root'] == str(run_root)
    assert manifest['config_root'] == str(run_root / 'configs')
    assert manifest['physics'] == {
        'fallback_existing_trend_mode': 'partial',
        'partial_trend_fallback': {
            'max_fraction': 0.05,
            'max_traces': 50000,
            'fallback_if_too_many': 'robust',
        },
    }
    physics_cfg = yaml.safe_load(
        (run_root / 'configs' / 'fold00' / '03_physics.yaml').read_text(
            encoding='utf-8',
        ),
    )
    physics_qc_cfg = yaml.safe_load(
        (run_root / 'configs' / 'fold00' / '04_physics_qc.yaml').read_text(
            encoding='utf-8',
        ),
    )
    _assert_partial_physics_fallback(physics_cfg)
    _assert_partial_physics_fallback(physics_qc_cfg)
