#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np

STAGE_PICK_KEYS = {
    "final": "final_pick_i",
    "robust": "robust_pick_i",
    "coarse": "coarse_pick_i",
}

METRIC_COLUMNS = [
    "run_id",
    "fold",
    "data_key",
    "stage",
    "segy_path",
    "fb_path",
    "pred_path",
    "dt_sec",
    "n_samples_orig",
    "n_traces",
    "n_teacher",
    "n_pred_finite",
    "n_pick_clipped_low",
    "n_pick_clipped_high",
    "n_pick_clipped",
    "coverage",
    "mae_samples",
    "rmse_samples",
    "median_abs_samples",
    "p90_abs_samples",
    "p95_abs_samples",
    "p99_abs_samples",
    "max_abs_samples",
    "mae_ms",
    "rmse_ms",
    "median_abs_ms",
    "p90_abs_ms",
    "p95_abs_ms",
    "p99_abs_ms",
    "max_abs_ms",
]


def read_list(path: Path) -> list[str]:
    return [
        x.strip()
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]


def build_fbpick_tag(segy_path: str | Path) -> str:
    p = Path(segy_path)
    return f"{p.parent.name}__{p.stem}" if p.parent.name else p.stem


def build_final_npz_name(segy_path: str | Path) -> str:
    return build_fbpick_tag(segy_path) + ".fbpick_final.npz"


def load_fb_i(path: str | Path) -> np.ndarray:
    p = Path(path)
    loaded = np.load(p, allow_pickle=False)
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "fb_i" not in loaded.files:
                raise KeyError(f"{p} is npz but does not contain key fb_i")
            arr = np.asarray(loaded["fb_i"])
        else:
            arr = np.asarray(loaded)
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()

    if arr.ndim != 1:
        raise ValueError(f"fb_i must be 1D: {p} shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"fb_i must be numeric: {p} dtype={arr.dtype}")
    return arr.astype(np.float64, copy=False)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and columns is None:
        path.write_text("", encoding="utf-8")
        return

    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stage_specs(stages: Iterable[str]) -> dict[str, str]:
    out = {}
    for raw_stage in stages:
        stage = raw_stage.strip()
        if not stage:
            continue
        if stage not in STAGE_PICK_KEYS:
            raise ValueError(f"unknown stage={stage}; valid={sorted(STAGE_PICK_KEYS)}")
        out[stage] = STAGE_PICK_KEYS[stage]
    return out


def new_accumulator() -> dict[str, object]:
    return {
        "n_files": 0,
        "n_traces": 0,
        "n_teacher": 0,
        "n_pred_finite": 0,
        "n_pick_clipped_low": 0,
        "n_pick_clipped_high": 0,
        "err_sum": 0.0,
        "sq_err_sum": 0.0,
        "sq_err_ms_sum": 0.0,
        "abs_errors": [],
        "abs_errors_ms": [],
    }


def update_accumulator(
    acc: dict[str, object],
    *,
    n_traces: int,
    n_teacher: int,
    n_pred_finite: int,
    n_pick_clipped_low: int,
    n_pick_clipped_high: int,
    err: np.ndarray,
    abs_err: np.ndarray,
    dt_sec: float,
) -> None:
    acc["n_files"] = int(acc["n_files"]) + 1
    acc["n_traces"] = int(acc["n_traces"]) + int(n_traces)
    acc["n_teacher"] = int(acc["n_teacher"]) + int(n_teacher)
    acc["n_pred_finite"] = int(acc["n_pred_finite"]) + int(n_pred_finite)
    acc["n_pick_clipped_low"] = int(acc["n_pick_clipped_low"]) + int(
        n_pick_clipped_low
    )
    acc["n_pick_clipped_high"] = int(acc["n_pick_clipped_high"]) + int(
        n_pick_clipped_high
    )
    if err.size:
        acc["err_sum"] = float(acc["err_sum"]) + float(np.sum(err, dtype=np.float64))
        acc["sq_err_sum"] = float(acc["sq_err_sum"]) + float(
            np.sum(np.square(err), dtype=np.float64)
        )
        ms_scale = float(dt_sec) * 1000.0
        acc["sq_err_ms_sum"] = float(acc["sq_err_ms_sum"]) + float(
            np.sum(np.square(err * ms_scale), dtype=np.float64)
        )
        abs_errors = cast("list[np.ndarray]", acc["abs_errors"])
        abs_errors.append(abs_err.astype(np.float32, copy=False))
        abs_errors_ms = cast("list[np.ndarray]", acc["abs_errors_ms"])
        abs_errors_ms.append((abs_err * ms_scale).astype(np.float32, copy=False))


def metric_values(
    *,
    n_traces: int,
    n_teacher: int,
    n_pred_finite: int,
    n_pick_clipped_low: int,
    n_pick_clipped_high: int,
    err: np.ndarray,
    abs_err: np.ndarray,
    dt_sec: float,
) -> dict[str, object]:
    out: dict[str, object] = {
        "n_traces": int(n_traces),
        "n_teacher": int(n_teacher),
        "n_pred_finite": int(n_pred_finite),
        "n_pick_clipped_low": int(n_pick_clipped_low),
        "n_pick_clipped_high": int(n_pick_clipped_high),
        "n_pick_clipped": int(n_pick_clipped_low + n_pick_clipped_high),
        "coverage": "" if n_teacher == 0 else float(n_pred_finite / n_teacher),
    }
    out.update(error_metric_values(err=err, abs_err=abs_err, dt_sec=dt_sec))
    return out


def error_metric_values(
    *,
    err: np.ndarray,
    abs_err: np.ndarray,
    dt_sec: float,
) -> dict[str, object]:
    if err.size == 0:
        return {
            "mae_samples": "",
            "rmse_samples": "",
            "median_abs_samples": "",
            "p90_abs_samples": "",
            "p95_abs_samples": "",
            "p99_abs_samples": "",
            "max_abs_samples": "",
            "mae_ms": "",
            "rmse_ms": "",
            "median_abs_ms": "",
            "p90_abs_ms": "",
            "p95_abs_ms": "",
            "p99_abs_ms": "",
            "max_abs_ms": "",
        }

    sample_metrics = {
        "mae_samples": float(np.mean(abs_err)),
        "rmse_samples": float(np.sqrt(np.mean(np.square(err)))),
        "median_abs_samples": float(np.percentile(abs_err, 50)),
        "p90_abs_samples": float(np.percentile(abs_err, 90)),
        "p95_abs_samples": float(np.percentile(abs_err, 95)),
        "p99_abs_samples": float(np.percentile(abs_err, 99)),
        "max_abs_samples": float(np.max(abs_err)),
    }
    ms_scale = float(dt_sec) * 1000.0
    return {
        **sample_metrics,
        "mae_ms": sample_metrics["mae_samples"] * ms_scale,
        "rmse_ms": sample_metrics["rmse_samples"] * ms_scale,
        "median_abs_ms": sample_metrics["median_abs_samples"] * ms_scale,
        "p90_abs_ms": sample_metrics["p90_abs_samples"] * ms_scale,
        "p95_abs_ms": sample_metrics["p95_abs_samples"] * ms_scale,
        "p99_abs_ms": sample_metrics["p99_abs_samples"] * ms_scale,
        "max_abs_ms": sample_metrics["max_abs_samples"] * ms_scale,
    }


def evaluate_stage_metrics(
    *,
    fb_i: np.ndarray,
    pick_i: np.ndarray,
    teacher_mask: np.ndarray,
    n_samples_orig: int,
    n_traces: int,
    dt_sec: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    pick_finite_mask = np.isfinite(pick_i)
    eval_mask = teacher_mask & pick_finite_mask
    pick_eval_i = np.clip(
        pick_i.astype(np.float64, copy=False),
        0.0,
        float(n_samples_orig - 1),
    )
    err = pick_eval_i[eval_mask] - fb_i[eval_mask]
    abs_err = np.abs(err)
    n_pick_clipped_low = int(np.count_nonzero(eval_mask & (pick_i < 0.0)))
    n_pick_clipped_high = int(
        np.count_nonzero(eval_mask & (pick_i > float(n_samples_orig - 1)))
    )
    values = metric_values(
        n_traces=n_traces,
        n_teacher=int(np.count_nonzero(teacher_mask)),
        n_pred_finite=int(np.count_nonzero(eval_mask)),
        n_pick_clipped_low=n_pick_clipped_low,
        n_pick_clipped_high=n_pick_clipped_high,
        err=err,
        abs_err=abs_err,
        dt_sec=dt_sec,
    )
    return values, err, abs_err


def accumulator_row(acc: dict[str, object]) -> dict[str, object]:
    n_teacher = int(acc["n_teacher"])
    n_pred_finite = int(acc["n_pred_finite"])
    out: dict[str, object] = {
        "n_files": int(acc["n_files"]),
        "n_traces": int(acc["n_traces"]),
        "n_teacher": n_teacher,
        "n_pred_finite": n_pred_finite,
        "n_pick_clipped_low": int(acc["n_pick_clipped_low"]),
        "n_pick_clipped_high": int(acc["n_pick_clipped_high"]),
        "n_pick_clipped": int(acc["n_pick_clipped_low"])
        + int(acc["n_pick_clipped_high"]),
        "coverage": "" if n_teacher == 0 else float(n_pred_finite / n_teacher),
    }

    if n_pred_finite > 0:
        abs_error_parts = cast("list[np.ndarray]", acc["abs_errors"])
        abs_all = np.concatenate(abs_error_parts, axis=0).astype(
            np.float64,
            copy=False,
        )
        abs_error_ms_parts = cast("list[np.ndarray]", acc["abs_errors_ms"])
        abs_ms_all = np.concatenate(abs_error_ms_parts, axis=0).astype(
            np.float64,
            copy=False,
        )
        out.update(
            {
                "mae_samples": float(np.mean(abs_all)),
                "rmse_samples": float(
                    np.sqrt(float(acc["sq_err_sum"]) / n_pred_finite)
                ),
                "median_abs_samples": float(np.percentile(abs_all, 50)),
                "p90_abs_samples": float(np.percentile(abs_all, 90)),
                "p95_abs_samples": float(np.percentile(abs_all, 95)),
                "p99_abs_samples": float(np.percentile(abs_all, 99)),
                "max_abs_samples": float(np.max(abs_all)),
                "mae_ms": float(np.mean(abs_ms_all)),
                "rmse_ms": float(
                    np.sqrt(float(acc["sq_err_ms_sum"]) / n_pred_finite)
                ),
                "median_abs_ms": float(np.percentile(abs_ms_all, 50)),
                "p90_abs_ms": float(np.percentile(abs_ms_all, 90)),
                "p95_abs_ms": float(np.percentile(abs_ms_all, 95)),
                "p99_abs_ms": float(np.percentile(abs_ms_all, 99)),
                "max_abs_ms": float(np.max(abs_ms_all)),
            }
        )
    else:
        out.update(
            error_metric_values(
                err=np.asarray([]),
                abs_err=np.asarray([]),
                dt_sec=0.0,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv-root",
        default="/workspace/proc/fbpick/site54/oof",
        help="site54 OOF CV root",
    )
    parser.add_argument(
        "--run-id",
        default="baseline_physical_center",
        help="run id used to derive default fine lists, predictions, and eval output",
    )
    parser.add_argument(
        "--fold-list-root",
        default=None,
        help="foldXX/heldout_sgy.txt and heldout_fb.txt root",
    )
    parser.add_argument(
        "--pred-root",
        default=None,
        help="run root containing foldXX/07_fine_infer outputs",
    )
    parser.add_argument(
        "--pred-stage-subdir",
        default="07_fine_infer",
        help="prediction stage directory under each fold in --pred-root",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output report directory",
    )
    parser.add_argument(
        "--stages",
        default="final,robust,coarse",
        help="comma-separated stages: final,robust,coarse",
    )
    args = parser.parse_args()

    cv_root = Path(args.cv_root)
    run_root = cv_root / "runs" / args.run_id
    fold_list_root = (
        Path(args.fold_list_root)
        if args.fold_list_root
        else run_root / "aggregate" / "05_collect_oof_lists" / "fine_fold_lists"
    )
    pred_root = Path(args.pred_root) if args.pred_root else run_root
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else run_root / "aggregate" / "08_eval"
    )

    specs = stage_specs(args.stages.split(","))

    per_data_rows: list[dict[str, object]] = []
    global_accs: dict[str, dict[str, object]] = {
        stage: new_accumulator() for stage in specs
    }

    fold_dirs = sorted(p for p in fold_list_root.glob("fold??") if p.is_dir())
    if not fold_dirs:
        raise RuntimeError(f"no fold dirs found: {fold_list_root}/fold??")

    n_files = 0

    for fold_dir in fold_dirs:
        fold = fold_dir.name
        sgys = read_list(fold_dir / "heldout_sgy.txt")
        fbs = read_list(fold_dir / "heldout_fb.txt")
        if len(sgys) != len(fbs):
            raise RuntimeError(f"{fold}: heldout_sgy and heldout_fb length mismatch")

        for segy, fb_path in zip(sgys, fbs, strict=True):
            tag = build_fbpick_tag(segy)
            final_path = (
                pred_root
                / fold
                / args.pred_stage_subdir
                / build_final_npz_name(segy)
            )
            if not final_path.is_file():
                raise FileNotFoundError(f"missing final npz: {final_path}")

            with np.load(final_path, allow_pickle=False) as z:
                payload = {k: z[k] for k in z.files}

            fb_i = load_fb_i(fb_path)
            n_traces = int(np.asarray(payload["n_traces"]).item())
            n_samples_orig = int(np.asarray(payload["n_samples_orig"]).item())
            dt_sec = float(np.asarray(payload["dt_sec"]).item())

            if fb_i.shape != (n_traces,):
                raise ValueError(
                    f"{tag}: fb shape {fb_i.shape} != n_traces {n_traces}"
                )

            teacher_mask = (
                np.isfinite(fb_i) & (fb_i > 0.0) & (fb_i < float(n_samples_orig))
            )
            n_teacher = int(np.count_nonzero(teacher_mask))

            if n_samples_orig <= 0:
                raise ValueError(f"{tag}: n_samples_orig must be positive")

            for stage, pick_key in specs.items():
                if pick_key not in payload:
                    if stage == "final" and pick_key == "final_pick_i":
                        raise KeyError("final stage evaluation requires final_pick_i")
                    raise KeyError(f"{final_path} missing key {pick_key}")

                pick = np.asarray(payload[pick_key], dtype=np.float64)
                if pick.shape != (n_traces,):
                    raise ValueError(
                        f"{tag}: {pick_key} shape {pick.shape} != ({n_traces},)"
                    )

                metrics, err, abs_err = evaluate_stage_metrics(
                    fb_i=fb_i,
                    pick_i=pick,
                    teacher_mask=teacher_mask,
                    n_samples_orig=n_samples_orig,
                    n_traces=n_traces,
                    dt_sec=dt_sec,
                )

                row = {
                    "run_id": args.run_id,
                    "fold": fold,
                    "data_key": tag,
                    "stage": stage,
                    "segy_path": segy,
                    "fb_path": fb_path,
                    "pred_path": str(final_path),
                    "dt_sec": dt_sec,
                    "n_samples_orig": n_samples_orig,
                    **metrics,
                }
                per_data_rows.append(row)
                update_accumulator(
                    global_accs[stage],
                    n_traces=n_traces,
                    n_teacher=int(metrics["n_teacher"]),
                    n_pred_finite=int(metrics["n_pred_finite"]),
                    n_pick_clipped_low=int(metrics["n_pick_clipped_low"]),
                    n_pick_clipped_high=int(metrics["n_pick_clipped_high"]),
                    err=err,
                    abs_err=abs_err,
                    dt_sec=dt_sec,
                )

            n_files += 1
            print(f"[eval] {fold} {tag} done n_teacher={n_teacher}")

    summary_rows = [
        {
            "run_id": args.run_id,
            "stage": stage,
            **accumulator_row(global_accs[stage]),
        }
        for stage in specs
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_data.csv", per_data_rows, columns=METRIC_COLUMNS)
    write_csv(
        out_dir / "summary_by_stage.csv",
        summary_rows,
        columns=["run_id", "stage", "n_files", *METRIC_COLUMNS[9:]],
    )
    for stale_name in (
        "per_fold.csv",
        "summary.csv",
        "top_errors_final.csv",
        "top_errors_robust.csv",
        "top_errors_coarse.csv",
    ):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    meta = {
        "run_id": args.run_id,
        "run_root": str(run_root),
        "fold_list_root": str(fold_list_root),
        "pred_root": str(pred_root),
        "pred_stage_subdir": str(args.pred_stage_subdir),
        "out_dir": str(out_dir),
        "n_files": n_files,
        "teacher_mask": "finite & fb_i > 0 & fb_i < n_samples_orig",
        "pick_mask": "finite only",
        "out_of_record_pick_policy": "clip_to_record_edge_before_error",
        "clip_low": 0,
        "clip_high": "n_samples_orig - 1",
        "eval_mask": "teacher_mask & finite(pick_i)",
        "stage_pick_keys": STAGE_PICK_KEYS,
        "reject_mask_used": False,
        "high_conf_used": False,
        "top_errors_written": False,
        "outputs": {
            "per_data_csv": str(out_dir / "per_data.csv"),
            "summary_by_stage_csv": str(out_dir / "summary_by_stage.csv"),
            "eval_meta_json": str(out_dir / "eval_meta.json"),
        },
    }
    (out_dir / "eval_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("wrote:")
    print(" ", out_dir / "per_data.csv")
    print(" ", out_dir / "summary_by_stage.csv")
    print(" ", out_dir / "eval_meta.json")
    print()
    print("summary:")
    for row in summary_rows:
        print(
            row["stage"],
            "n_teacher=", row.get("n_teacher"),
            "n_pred_finite=", row.get("n_pred_finite"),
            "coverage=", row.get("coverage"),
            "mae_samples=", row.get("mae_samples"),
            "p90_abs_samples=", row.get("p90_abs_samples"),
            "n_pick_clipped=", row.get("n_pick_clipped"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
