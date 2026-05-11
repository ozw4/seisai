from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / 'cli' / 'run_arakawa_fbpick_physical_export.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('_arakawa_runner_cli_test', CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load spec for {CLI_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_arakawa_runner_uses_fixed_templates_and_runs_three_stages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, '_repo_root', lambda: tmp_path)

    segy = tmp_path / 'Arakawa2026' / 'line.sgy'
    segy.parent.mkdir()
    segy.touch()

    template_dir = tmp_path / 'proc' / 'arakawa' / 'configs'
    template_dir.mkdir(parents=True)
    (template_dir / 'coarse_one.yaml').write_text(
        yaml.safe_dump(
            {
                'paths': {
                    'segy_files': ['/old/input.sgy'],
                    'out_dir': '/old/coarse',
                },
                'infer': {'ckpt_path': '/fixed/best.pt'},
                'model': {'pre_stages': 3, 'backbone': 'resnet18'},
                'dataset': {'coord_unit_scale_to_m': 1.0},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    (template_dir / 'physics_one.yaml').write_text(
        yaml.safe_dump(
            {
                'paths': {
                    'coarse_npz_path': '/old/coarse.npz',
                    'out_path': '/old/robust.npz',
                },
                'physical_trend': {'enabled': True},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    cfg_path = tmp_path / 'runner.yaml'
    cfg_path.write_text(
        yaml.safe_dump({'paths': {'segy': str(segy)}}),
        encoding='utf-8',
    )

    calls: dict[str, object] = {}

    def _resolve_relpath(base_dir, value):
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = Path(base_dir) / p
        return str(p.resolve())

    def _fake_run_coarse(path):
        calls['coarse_cfg_path'] = str(path)
        cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        calls['coarse_cfg'] = cfg
        out = Path(cfg['paths']['out_dir']) / 'Arakawa2026__line.coarse.npz'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out

    def _fake_run_physics(path):
        calls['physics_cfg_path'] = str(path)
        cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        calls['physics_cfg'] = cfg
        out = Path(cfg['paths']['out_path'])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out

    def _fake_export(**kwargs):
        calls['export_kwargs'] = kwargs
        Path(kwargs['out_crd_path']).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs['out_crd_path']).touch()
        Path(kwargs['out_npz_path']).touch()
        return {'out_crd': str(kwargs['out_crd_path'])}

    runtime = SimpleNamespace(
        load_cfg_with_base_dir=lambda path: (
            yaml.safe_load(Path(path).read_text(encoding='utf-8')),
            Path(path).parent,
        ),
        resolve_relpath=_resolve_relpath,
        build_fbpick_tag=lambda path: 'Arakawa2026__line',
        run_coarse_infer=_fake_run_coarse,
        run_physics=_fake_run_physics,
        export_robust_pick_to_grstat=_fake_export,
    )
    monkeypatch.setattr(module, '_load_runtime', lambda: runtime)

    result = module.run_pipeline(cfg_path)

    work_dir = tmp_path / 'proc' / 'arakawa'
    expected_coarse = work_dir / 'coarse' / 'Arakawa2026__line.coarse.npz'
    expected_robust = work_dir / 'robust' / 'Arakawa2026__line.robust.npz'
    expected_crd = (
        work_dir
        / 'grstat'
        / 'Arakawa2026__line.physical_center.snap_peak.ltcor2.crd'
    )

    assert result == expected_crd
    assert calls['coarse_cfg']['paths']['segy_files'] == [str(segy)]
    assert calls['coarse_cfg']['paths']['out_dir'] == str(work_dir / 'coarse')
    assert calls['coarse_cfg']['infer']['ckpt_path'] == '/fixed/best.pt'
    assert calls['coarse_cfg']['model']['pre_stages'] == 3
    assert calls['physics_cfg']['paths']['coarse_npz_path'] == str(expected_coarse)
    assert calls['physics_cfg']['paths']['out_path'] == str(expected_robust)
    assert calls['export_kwargs']['robust_npz_path'] == expected_robust
    assert calls['export_kwargs']['out_crd_path'] == expected_crd
    assert calls['export_kwargs']['pick_key'] == 'physical_center_i'
    assert calls['export_kwargs']['max_shift_samples'] == 2
