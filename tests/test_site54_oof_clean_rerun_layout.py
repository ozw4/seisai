from __future__ import annotations

# ruff: noqa: INP001,S603
import shutil
import subprocess
import sys
from pathlib import Path

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
    assert (OOF_ROOT / "config_templates" / "fine_infer.yaml").is_file()


def test_clean_rerun_procedure_has_no_legacy_root_paths() -> None:
    readme = (OOF_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Clean Rerun Procedure")
    end = readme.index("## Legacy paths")
    procedure = readme[start:end]

    assert "/oof/configs" not in procedure
    assert "/oof/lists/fine" not in procedure
    assert "/oof/logs" not in procedure
