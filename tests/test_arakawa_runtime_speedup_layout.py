from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / 'proc' / 'arakawa' / 'experiments' / 'runtime_speedup' / 'configs'
LEGACY_CONFIG_DIR = REPO_ROOT / 'proc' / 'arakawa' / 'configs' / 'runtime_speedup'


def _load_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(CONFIG_DIR.glob('*.yaml')):
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert isinstance(data, dict)
        configs[path.stem] = data
    return configs


def test_runtime_speedup_source_configs_live_only_under_experiments() -> None:
    configs = _load_configs()

    assert set(configs) == {
        'A0D_downsample_only',
        'A0_full',
        'A1_diagnostics_only',
        'A2_anchor_selection_dry_run',
        'A3_anchor_stride5_nearest_anchor',
        'A4_anchor_stride5_t0_shift',
        'A5_anchor_stride5_t0_shift_adaptive_refit',
        'A6_A5_obs_downsample256',
        'B201_side_on_gap_on_control',
        'B202_side_on_gap_off',
        'B203_side_off_gap_on',
        'B204_side_off_gap_off',
    }
    assert not list(LEGACY_CONFIG_DIR.glob('*.yaml'))


def test_runtime_speedup_work_dirs_resolve_under_outputs() -> None:
    configs = _load_configs()

    for run_name, cfg in configs.items():
        paths = cfg['paths']
        work_dir = (CONFIG_DIR / paths['work_dir']).resolve()
        assert work_dir == (
            REPO_ROOT / 'proc' / 'arakawa' / 'outputs' / 'runtime_runs' / run_name
        )


def test_a0d_downsample_only_uses_observation_sampling_without_anchor_reuse() -> None:
    cfg = _load_configs()['A0D_downsample_only']
    runtime = cfg['physical_runtime']

    assert runtime['fit_policy'] == 'full'
    assert runtime['anchor_reuse']['enabled'] is False
    assert runtime['observation_sampling']['enabled'] is True
    assert runtime['observation_sampling']['method'] == 'offset_bin'
    assert runtime['observation_sampling']['max_obs_per_fit'] == 256
    assert 'out_path' not in cfg['paths']
