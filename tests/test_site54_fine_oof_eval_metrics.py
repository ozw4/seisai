from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "proc"
    / "fbpick"
    / "site54"
    / "oof"
    / "scripts"
    / "evaluate_fine_oof.py"
)


def _write_case(
    *,
    run_root: Path,
    fold_list_root: Path,
    fold: str,
    data_name: str,
    teacher: list[float],
    final_pick: list[float],
    robust_pick: list[float],
    coarse_pick: list[float],
    reject_mask: list[bool],
    final_pick_f: list[float] | None = None,
) -> None:
    fold_dir = fold_list_root / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    fb_path = run_root / "fixtures" / f"{fold}_{data_name}.fb.npy"
    fb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(fb_path, np.asarray(teacher, dtype=np.float32))

    segy = f"/data/{fold}/{data_name}.sgy"
    with (fold_dir / "heldout_sgy.txt").open("a", encoding="utf-8") as f:
        f.write(f"{segy}\n")
    with (fold_dir / "heldout_fb.txt").open("a", encoding="utf-8") as f:
        f.write(f"{fb_path}\n")

    n_traces = len(teacher)
    out_dir = run_root / fold / "07_fine_infer"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_npz = out_dir / f"{fold}__{data_name}.fbpick_final.npz"
    final_pick_arr = np.asarray(final_pick, dtype=np.float32)
    final_pick_i = np.where(np.isfinite(final_pick_arr), final_pick_arr, -1).astype(
        np.int32,
    )
    final_pick_f_arr = (
        np.asarray(final_pick_f, dtype=np.float32)
        if final_pick_f is not None
        else final_pick_arr
    )
    np.savez(
        final_npz,
        dt_sec=np.asarray(0.004, dtype=np.float32),
        n_samples_orig=np.asarray(1200, dtype=np.int32),
        n_traces=np.asarray(n_traces, dtype=np.int32),
        final_pick_i=final_pick_i,
        final_pick_f=final_pick_f_arr,
        final_conf=np.ones(n_traces, dtype=np.float32),
        robust_pick_i=np.asarray(robust_pick, dtype=np.float32),
        robust_conf=np.ones(n_traces, dtype=np.float32),
        coarse_pick_i=np.asarray(coarse_pick, dtype=np.float32),
        coarse_pmax=np.ones(n_traces, dtype=np.float32),
        high_conf_mask=np.ones(n_traces, dtype=np.bool_),
        reject_mask=np.asarray(reject_mask, dtype=np.bool_),
        reason_mask=np.zeros(n_traces, dtype=np.uint8),
        window_start_i=(final_pick_i - 128).astype(np.int32),
        window_end_i=(final_pick_i + 127).astype(np.int32),
        fine_pick_local_i=np.full(n_traces, 128, dtype=np.int32),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fine_oof_eval_uses_teacher_denominator_and_weighted_summary(
    tmp_path: Path,
) -> None:
    cv_root = tmp_path / "oof"
    run_id = "eval_probe"
    run_root = cv_root / "runs" / run_id
    fold_list_root = (
        run_root / "aggregate" / "05_collect_oof_lists" / "fine_fold_lists"
    )
    out_dir = run_root / "aggregate" / "08_eval"

    _write_case(
        run_root=run_root,
        fold_list_root=fold_list_root,
        fold="fold00",
        data_name="a",
        teacher=[10, 20, 30, 40],
        final_pick=[10, 21, np.nan, 100],
        robust_pick=[10, np.nan, 31, 100],
        coarse_pick=[10, 21, np.nan, 100],
        reject_mask=[False, False, False, True],
    )
    _write_case(
        run_root=run_root,
        fold_list_root=fold_list_root,
        fold="fold00",
        data_name="b",
        teacher=[50],
        final_pick=[50],
        robust_pick=[50],
        coarse_pick=[50],
        reject_mask=[False],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cv-root",
            str(cv_root),
            "--run-id",
            run_id,
            "--fold-list-root",
            str(fold_list_root),
            "--pred-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    per_data = _read_csv(out_dir / "per_data.csv")
    final_a = next(
        row
        for row in per_data
        if row["stage"] == "final" and row["data_name"] == "fold00__a"
    )
    assert final_a["n_teacher"] == "4"
    assert final_a["n_pred_valid"] == "3"
    assert final_a["n_pred_accepted"] == "2"
    assert float(final_a["coverage"]) == 0.5
    assert float(final_a["within_1_samples"]) == 0.5
    assert float(final_a["accepted_mae_samples"]) == 0.5
    assert math.isclose(float(final_a["accepted_rmse_samples"]), math.sqrt(0.5))

    n_teacher_by_stage = {
        row["stage"]: row["n_teacher"]
        for row in per_data
        if row["data_name"] == "fold00__a"
    }
    assert n_teacher_by_stage == {"coarse": "4", "final": "4", "robust": "4"}

    summary = _read_csv(out_dir / "summary.csv")
    final_summary = next(row for row in summary if row["stage"] == "final")
    assert final_summary["n_teacher"] == "5"
    assert final_summary["n_pred_accepted"] == "3"
    assert float(final_summary["within_1_samples"]) == 0.6

    per_fold = _read_csv(out_dir / "per_fold.csv")
    final_fold = next(row for row in per_fold if row["stage"] == "final")
    assert final_fold["n_teacher"] == "5"
    assert float(final_fold["coverage"]) == 0.6

    for name in ("per_data.csv", "per_fold.csv", "summary.csv"):
        rows = _read_csv(out_dir / name)
        assert rows
        assert not any("_ms" in column for column in rows[0])
        assert "n_eval" not in rows[0]


def test_fine_oof_eval_final_stage_uses_final_pick_i(tmp_path: Path) -> None:
    cv_root = tmp_path / "oof"
    run_id = "eval_final_pick_i"
    run_root = cv_root / "runs" / run_id
    fold_list_root = (
        run_root / "aggregate" / "05_collect_oof_lists" / "fine_fold_lists"
    )
    out_dir = run_root / "aggregate" / "08_eval"

    _write_case(
        run_root=run_root,
        fold_list_root=fold_list_root,
        fold="fold00",
        data_name="a",
        teacher=[110, 190, 310],
        final_pick=[100, 200, 300],
        final_pick_f=[1000, 1000, 1000],
        robust_pick=[100, 200, 300],
        coarse_pick=[100, 200, 300],
        reject_mask=[False, False, False],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cv-root",
            str(cv_root),
            "--run-id",
            run_id,
            "--fold-list-root",
            str(fold_list_root),
            "--pred-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    summary = _read_csv(out_dir / "summary.csv")
    final_summary = next(row for row in summary if row["stage"] == "final")
    assert float(final_summary["accepted_mae_samples"]) == 10.0

    top_errors = _read_csv(out_dir / "top_errors_final.csv")
    assert top_errors
    assert "final_pick_i" in top_errors[0]
    assert "final_pick_f" not in top_errors[0]
    assert {float(row["abs_error_samples"]) for row in top_errors} == {10.0}

    per_data = _read_csv(out_dir / "per_data.csv")
    assert not any(row["stage"] == "final_high_conf" for row in per_data)

    meta = json.loads((out_dir / "eval_meta.json").read_text(encoding="utf-8"))
    assert meta["final_pick_key"] == "final_pick_i"
    assert meta["include_high_conf_final"] is False


def test_fine_oof_eval_final_stage_requires_final_pick_i(tmp_path: Path) -> None:
    cv_root = tmp_path / "oof"
    run_id = "eval_missing_final_pick_i"
    run_root = cv_root / "runs" / run_id
    fold_list_root = (
        run_root / "aggregate" / "05_collect_oof_lists" / "fine_fold_lists"
    )
    out_dir = run_root / "aggregate" / "08_eval"

    _write_case(
        run_root=run_root,
        fold_list_root=fold_list_root,
        fold="fold00",
        data_name="a",
        teacher=[110],
        final_pick=[100],
        robust_pick=[100],
        coarse_pick=[100],
        reject_mask=[False],
    )

    final_npz = run_root / "fold00" / "07_fine_infer" / "fold00__a.fbpick_final.npz"
    with np.load(final_npz, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files if key != "final_pick_i"}
    np.savez(final_npz, **payload)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cv-root",
            str(cv_root),
            "--run-id",
            run_id,
            "--fold-list-root",
            str(fold_list_root),
            "--pred-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "final stage evaluation requires final_pick_i" in result.stderr
