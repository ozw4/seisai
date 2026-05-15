from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ARAKAWA_DIR = REPO_ROOT / "proc" / "arakawa"
CONFIG_DIR = ARAKAWA_DIR / "configs"
RUNTIME_CONFIG_DIR = ARAKAWA_DIR / "experiments" / "runtime_speedup" / "configs"


def _load_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), path
    return data


def test_required_arakawa_user_files_exist() -> None:
    required_paths = [
        ARAKAWA_DIR / "README.md",
        ARAKAWA_DIR / "README_RUNTIME_SPEEDUP.md",
        ARAKAWA_DIR / "MIGRATION.md",
        CONFIG_DIR / "run_coarse_physics_export_minimal.yaml",
        CONFIG_DIR / "run_coarse_physics_export.yaml",
        CONFIG_DIR / "templates" / "coarse.yaml",
        CONFIG_DIR / "templates" / "fine.yaml",
        CONFIG_DIR / "templates" / "physics.yaml",
        CONFIG_DIR / "templates" / "physics_qc_no_fb.yaml",
        RUNTIME_CONFIG_DIR / "A0_full.yaml",
        RUNTIME_CONFIG_DIR / "A0D_downsample_only.yaml",
        RUNTIME_CONFIG_DIR / "A1_diagnostics_only.yaml",
        RUNTIME_CONFIG_DIR / "A3_anchor_stride5_nearest_anchor.yaml",
        RUNTIME_CONFIG_DIR / "A5_anchor_stride5_t0_shift_adaptive_refit.yaml",
        RUNTIME_CONFIG_DIR / "A6_A5_obs_downsample256.yaml",
    ]

    missing = [path for path in required_paths if not path.is_file()]
    assert missing == []


def test_arakawa_yaml_files_parse_successfully() -> None:
    yaml_paths = [
        *sorted(CONFIG_DIR.rglob("*.yaml")),
        *sorted(RUNTIME_CONFIG_DIR.glob("*.yaml")),
    ]
    assert yaml_paths

    for path in yaml_paths:
        yaml.safe_load(path.read_text(encoding="utf-8"))


def test_minimal_user_config_has_only_allowed_high_level_sections() -> None:
    cfg = _load_mapping(CONFIG_DIR / "run_coarse_physics_export_minimal.yaml")

    assert set(cfg) <= {"paths", "visualization"}
    assert set(cfg["paths"]) == {"sgy_file"}
    assert set(cfg["visualization"]) == {"enabled"}


def test_runtime_speedup_configs_live_in_expected_directory() -> None:
    config_names = {path.name for path in RUNTIME_CONFIG_DIR.glob("*.yaml")}

    assert config_names == {
        "A0_full.yaml",
        "A0D_downsample_only.yaml",
        "A1_diagnostics_only.yaml",
        "A2_anchor_selection_dry_run.yaml",
        "A3_anchor_stride5_nearest_anchor.yaml",
        "A4_anchor_stride5_t0_shift.yaml",
        "A5_anchor_stride5_t0_shift_adaptive_refit.yaml",
        "A6_A5_obs_downsample256.yaml",
        "B201_side_on_gap_on_control.yaml",
        "B202_side_on_gap_off.yaml",
        "B203_side_off_gap_on.yaml",
        "B204_side_off_gap_off.yaml",
        "C301_side_gap_context_precompute_control.yaml",
    }
    assert not (CONFIG_DIR / "runtime_speedup").exists()


def test_generated_output_paths_are_ignored_by_git() -> None:
    try:
        repo_result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("git is not available")

    if repo_result.returncode != 0:
        pytest.skip("git ignore rules are only testable inside a git checkout")

    paths = [
        "proc/arakawa/outputs/generated_configs/example.yaml",
        "proc/arakawa/outputs/runtime_runs/A0_full/example.npz",
        "proc/arakawa/outputs/grstat/example.crd",
        "proc/arakawa/generated_configs/example.yaml",
        "proc/arakawa/runtime_runs/A0_full/example.npz",
        "proc/arakawa/grstat/example.crd",
    ]

    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, path
