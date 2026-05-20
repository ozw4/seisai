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
    n_samples_orig: int = 1200,
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
        n_samples_orig=np.asarray(n_samples_orig, dtype=np.int32),
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


def test_fine_oof_eval_uses_teacher_only_mask_and_clips_out_of_record_picks(
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
        teacher=[0, 10, 20, 1199, 1200, np.nan],
        final_pick=[500, -5, 25, 1300, 100, 100],
        robust_pick=[500, -5, 25, 1300, 100, 100],
        coarse_pick=[500, -5, 25, 1300, 100, 100],
        reject_mask=[False, False, True, False, False, False],
        n_samples_orig=1200,
    )

    for stale_name in (
        "per_fold.csv",
        "summary.csv",
        "top_errors_final.csv",
        "top_errors_robust.csv",
        "top_errors_coarse.csv",
    ):
        (out_dir / stale_name).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / stale_name).write_text("stale\n", encoding="utf-8")

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
        if row["stage"] == "final" and row["data_key"] == "fold00__a"
    )
    assert final_a["n_teacher"] == "3"
    assert final_a["n_pred_finite"] == "3"
    assert final_a["n_pick_clipped_low"] == "1"
    assert final_a["n_pick_clipped_high"] == "1"
    assert final_a["n_pick_clipped"] == "2"
    assert float(final_a["coverage"]) == 1.0
    assert float(final_a["mae_samples"]) == 5.0
    assert math.isclose(float(final_a["rmse_samples"]), math.sqrt(125.0 / 3.0))
    assert float(final_a["max_abs_samples"]) == 10.0
    assert math.isclose(float(final_a["mae_ms"]), 20.0, abs_tol=1e-5)

    n_teacher_by_stage = {
        row["stage"]: row["n_teacher"]
        for row in per_data
        if row["data_key"] == "fold00__a"
    }
    assert n_teacher_by_stage == {"coarse": "3", "final": "3", "robust": "3"}

    summary = _read_csv(out_dir / "summary_by_stage.csv")
    final_summary = next(row for row in summary if row["stage"] == "final")
    assert final_summary["n_files"] == "1"
    assert final_summary["n_teacher"] == "3"
    assert final_summary["n_pred_finite"] == "3"
    assert final_summary["n_pick_clipped"] == "2"
    assert float(final_summary["mae_samples"]) == 5.0

    for name in (
        "per_fold.csv",
        "summary.csv",
        "top_errors_final.csv",
        "top_errors_robust.csv",
        "top_errors_coarse.csv",
    ):
        assert not (out_dir / name).exists()

    assert not any(column.startswith("accepted_") for column in per_data[0])
    assert "n_pred_accepted" not in per_data[0]
    assert "high_conf_used" not in per_data[0]

    meta = json.loads((out_dir / "eval_meta.json").read_text(encoding="utf-8"))
    assert meta["teacher_mask"] == "finite & fb_i > 0 & fb_i < n_samples_orig"
    assert meta["pick_mask"] == "finite only"
    assert meta["out_of_record_pick_policy"] == "clip_to_record_edge_before_error"
    assert meta["stage_pick_keys"] == {
        "final": "final_pick_i",
        "robust": "robust_pick_i",
        "coarse": "coarse_pick_i",
    }
    assert meta["reject_mask_used"] is False
    assert meta["high_conf_used"] is False
    assert meta["top_errors_written"] is False


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

    summary = _read_csv(out_dir / "summary_by_stage.csv")
    final_summary = next(row for row in summary if row["stage"] == "final")
    assert float(final_summary["mae_samples"]) == 10.0

    per_data = _read_csv(out_dir / "per_data.csv")
    assert not any(row["stage"] == "final_high_conf" for row in per_data)
    assert not (out_dir / "top_errors_final.csv").exists()

    meta = json.loads((out_dir / "eval_meta.json").read_text(encoding="utf-8"))
    assert meta["stage_pick_keys"]["final"] == "final_pick_i"
    assert meta["high_conf_used"] is False


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
