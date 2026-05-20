from __future__ import annotations

# ruff: noqa: INP001,S603
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_ROOT = REPO_ROOT / "proc" / "fbpick" / "site54" / "oof"
SCRIPTS_ROOT = OOF_ROOT / "scripts"


def run_python(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def copy_oof_inputs(tmp_path: Path) -> Path:
    dst = tmp_path / "oof"
    dst.mkdir()
    for name in ("scripts", "fold_lists", "config_templates"):
        shutil.copytree(
            OOF_ROOT / name,
            dst / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return dst


def test_fold_list_validation_reports_expected_site54_coverage() -> None:
    result = run_python(
        SCRIPTS_ROOT / "check_fold_lists.py",
        "--fold-list-root",
        OOF_ROOT / "fold_lists",
    )

    assert result.returncode == 0, result.stderr
    assert "OK: total heldout regions = 54; duplicate heldout SGY = 0" in result.stdout


def test_generator_defaults_write_run_scoped_configs(tmp_path: Path) -> None:
    cv_root = copy_oof_inputs(tmp_path)
    run_id = "rerun_probe"
    run_root = cv_root / "runs" / run_id

    coarse = run_python(
        cv_root / "scripts" / "make_coarse_fold_configs.py",
        "--cv-root",
        cv_root,
        "--run-id",
        run_id,
    )
    assert coarse.returncode == 0, coarse.stderr

    physics = run_python(
        cv_root / "scripts" / "make_physics_fold_configs.py",
        "--cv-root",
        cv_root,
        "--run-id",
        run_id,
    )
    assert physics.returncode == 0, physics.stderr

    assert (run_root / "configs" / "fold00" / "01_coarse_train.yaml").is_file()
    assert (run_root / "configs" / "fold00" / "03_physics.yaml").is_file()
    assert not (cv_root / "configs").exists()
    physics_cfg = yaml.safe_load(
        (run_root / "configs" / "fold00" / "03_physics.yaml").read_text(
            encoding="utf-8",
        ),
    )
    physics_qc_cfg = yaml.safe_load(
        (run_root / "configs" / "fold00" / "04_physics_qc.yaml").read_text(
            encoding="utf-8",
        ),
    )
    for cfg in (physics_cfg, physics_qc_cfg):
        runtime = cfg["physical_runtime"]
        assert runtime["fallback_existing_trend_mode"] == "partial"
        assert runtime["partial_trend_fallback"]["enabled"] is True
        assert runtime["partial_trend_fallback"]["max_fraction"] == 0.05
        assert runtime["partial_trend_fallback"]["max_traces"] == 50000
    assert physics_qc_cfg["vis"]["first_panel_only"] is True


def test_runner_dry_run_uses_run_scoped_config_paths(tmp_path: Path) -> None:
    cv_root = copy_oof_inputs(tmp_path)
    run_id = "rerun_probe"
    run_root = cv_root / "runs" / run_id
    config_root = run_root / "configs"

    result = run_python(
        cv_root / "scripts" / "run_site54_oof_cv.py",
        "--repo-root",
        tmp_path,
        "--cv-root",
        cv_root,
        "--run-id",
        run_id,
        "--run-root",
        run_root,
        "--config-root",
        config_root,
        "--stage",
        "all",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert f"RUN_ROOT={run_root}" in result.stdout
    assert f"CONFIG_ROOT={config_root}" in result.stdout
    assert str(config_root / "fold00" / "01_coarse_train.yaml") in result.stdout
    assert str(cv_root / "configs") not in result.stdout
    assert not (run_root / "manifest.yaml").exists()


def test_fine_base_configs_exist() -> None:
    assert (OOF_ROOT / "config_templates" / "fine_train.yaml").is_file()
    fine_infer_path = OOF_ROOT / "config_templates" / "fine_infer.yaml"
    assert fine_infer_path.is_file()
    fine_infer_cfg = yaml.safe_load(fine_infer_path.read_text(encoding="utf-8"))
    assert fine_infer_cfg["viewer"]["first_panel_only"] is True


def test_clean_rerun_procedure_has_no_legacy_root_paths() -> None:
    readme = (OOF_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Clean Rerun Procedure")
    end = readme.index("## Legacy paths")
    procedure = readme[start:end]

    assert "/oof/configs" not in procedure
    assert "/oof/lists/fine" not in procedure
    assert "/oof/logs" not in procedure
