#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, cast

import numpy as np

DEFAULT_THRESHOLDS_SAMPLES = (1.0, 2.0, 4.0, 8.0)
REJECT_MASK_STAGES = {"final", "robust"}
StageMetricResult = tuple[
    dict[str, object],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    dict[float, int],
]
FileStageCacheValue = tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]


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


def empty_metric_values(
    *, thresholds_samples: tuple[float, ...], n_traces: int, n_teacher: int
) -> dict[str, object]:
    out: dict[str, object] = {
        "n_traces": int(n_traces),
        "n_teacher": int(n_teacher),
        "n_pred_valid": 0,
        "n_pred_accepted": 0,
        "coverage": "" if n_teacher == 0 else 0.0,
        "accepted_mae_samples": "",
        "accepted_bias_samples": "",
        "accepted_rmse_samples": "",
        "accepted_median_abs_samples": "",
        "accepted_p90_abs_samples": "",
        "accepted_p95_abs_samples": "",
        "accepted_p99_abs_samples": "",
        "accepted_max_abs_samples": "",
        "mean_conf": "",
        "median_conf": "",
    }
    for th in thresholds_samples:
        out[f"within_{th:g}_samples"] = "" if n_teacher == 0 else 0.0
    return out


def metric_prefix_values(
    *,
    err_samples: np.ndarray,
    abs_samples: np.ndarray,
    conf: np.ndarray | None,
    n_traces: int,
    n_teacher: int,
    n_pred_valid: int,
    n_pred_accepted: int,
    hit_counts: dict[float, int],
    thresholds_samples: tuple[float, ...],
) -> dict[str, object]:
    if n_pred_accepted == 0:
        out = empty_metric_values(
            thresholds_samples=thresholds_samples,
            n_traces=n_traces,
            n_teacher=n_teacher,
        )
        out["n_pred_valid"] = int(n_pred_valid)
        return out

    out = {
        "n_traces": int(n_traces),
        "n_teacher": int(n_teacher),
        "n_pred_valid": int(n_pred_valid),
        "n_pred_accepted": int(n_pred_accepted),
        "coverage": "" if n_teacher == 0 else float(n_pred_accepted / n_teacher),
        "accepted_mae_samples": float(np.mean(abs_samples)),
        "accepted_bias_samples": float(np.mean(err_samples)),
        "accepted_rmse_samples": float(np.sqrt(np.mean(np.square(err_samples)))),
        "accepted_median_abs_samples": float(np.percentile(abs_samples, 50)),
        "accepted_p90_abs_samples": float(np.percentile(abs_samples, 90)),
        "accepted_p95_abs_samples": float(np.percentile(abs_samples, 95)),
        "accepted_p99_abs_samples": float(np.percentile(abs_samples, 99)),
        "accepted_max_abs_samples": float(np.max(abs_samples)),
    }

    if conf is not None and conf.size:
        conf_finite = conf[np.isfinite(conf)]
        out["mean_conf"] = float(np.mean(conf_finite)) if conf_finite.size else ""
        out["median_conf"] = (
            float(np.percentile(conf_finite, 50)) if conf_finite.size else ""
        )
    else:
        out["mean_conf"] = ""
        out["median_conf"] = ""

    for th in thresholds_samples:
        out[f"within_{th:g}_samples"] = (
            "" if n_teacher == 0 else float(hit_counts[float(th)] / n_teacher)
        )

    return out


def evaluate_stage_metrics(
    *,
    fb_i: np.ndarray,
    pick: np.ndarray,
    teacher_mask: np.ndarray,
    valid_pick: np.ndarray,
    accepted_mask: np.ndarray,
    conf_arr: np.ndarray | None,
    n_traces: int,
    thresholds_samples: tuple[float, ...],
) -> StageMetricResult:
    teacher_accepted = teacher_mask & accepted_mask
    n_teacher = int(np.count_nonzero(teacher_mask))
    n_pred_accepted = int(np.count_nonzero(teacher_accepted))
    err = pick[teacher_accepted] - fb_i[teacher_accepted]
    abs_err = np.abs(err)
    hit_counts = {
        float(th): int(
            np.count_nonzero(teacher_accepted & (np.abs(pick - fb_i) <= float(th)))
        )
        for th in thresholds_samples
    }
    conf = conf_arr[teacher_accepted] if conf_arr is not None else None
    values = metric_prefix_values(
        err_samples=err,
        abs_samples=abs_err,
        conf=conf,
        n_traces=n_traces,
        n_teacher=n_teacher,
        n_pred_valid=int(np.count_nonzero(teacher_mask & valid_pick)),
        n_pred_accepted=n_pred_accepted,
        hit_counts=hit_counts,
        thresholds_samples=thresholds_samples,
    )
    return values, err, abs_err, conf, np.flatnonzero(teacher_accepted), hit_counts


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: list[str] = []
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


def parse_csv_numbers(raw: str, *, dtype: type = float) -> tuple:
    out = []
    for raw_item in raw.split(","):
        item = raw_item.strip()
        if not item:
            continue
        out.append(dtype(item))
    return tuple(out)


def stage_specs(stages: Iterable[str]) -> dict[str, tuple[str, str | None]]:
    mapping = {
        "final": ("final_pick_i", "final_conf"),
        "robust": ("robust_pick_i", "robust_conf"),
        "coarse": ("coarse_pick_i", "coarse_pmax"),
        "center": ("center_raw_i", None),
    }
    out = {}
    for raw_stage in stages:
        stage = raw_stage.strip()
        if not stage:
            continue
        if stage not in mapping:
            raise ValueError(f"unknown stage={stage}; valid={sorted(mapping)}")
        out[stage] = mapping[stage]
    return out


def new_accumulator(thresholds_samples: tuple[float, ...]) -> dict[str, object]:
    return {
        "n_files": 0,
        "n_traces": 0,
        "n_teacher": 0,
        "n_pred_valid": 0,
        "n_pred_accepted": 0,
        "err_sum": 0.0,
        "sq_err_sum": 0.0,
        "abs_errors": [],
        "conf_sum": 0.0,
        "conf_count": 0,
        "hit_counts": {float(th): 0 for th in thresholds_samples},
    }


def update_accumulator(
    acc: dict[str, object],
    *,
    n_traces: int,
    n_teacher: int,
    n_pred_valid: int,
    n_pred_accepted: int,
    err: np.ndarray,
    abs_err: np.ndarray,
    conf: np.ndarray | None,
    hit_counts: dict[float, int],
) -> None:
    acc["n_files"] = int(acc["n_files"]) + 1
    acc["n_traces"] = int(acc["n_traces"]) + int(n_traces)
    acc["n_teacher"] = int(acc["n_teacher"]) + int(n_teacher)
    acc["n_pred_valid"] = int(acc["n_pred_valid"]) + int(n_pred_valid)
    acc["n_pred_accepted"] = int(acc["n_pred_accepted"]) + int(n_pred_accepted)
    if err.size:
        acc["err_sum"] = float(acc["err_sum"]) + float(np.sum(err, dtype=np.float64))
        acc["sq_err_sum"] = float(acc["sq_err_sum"]) + float(
            np.sum(np.square(err), dtype=np.float64)
        )
        abs_errors = cast(list[np.ndarray], acc["abs_errors"])
        abs_errors.append(abs_err.astype(np.float32, copy=False))
    if conf is not None and conf.size:
        conf_finite = conf[np.isfinite(conf)]
        acc["conf_sum"] = float(acc["conf_sum"]) + float(
            np.sum(conf_finite, dtype=np.float64)
        )
        acc["conf_count"] = int(acc["conf_count"]) + int(conf_finite.size)
    acc_hits = cast(dict[float, int], acc["hit_counts"])
    for th, value in hit_counts.items():
        acc_hits[float(th)] = int(acc_hits[float(th)]) + int(value)


def accumulator_row(
    acc: dict[str, object],
    *,
    thresholds_samples: tuple[float, ...],
) -> dict[str, object]:
    n_teacher = int(acc["n_teacher"])
    n_pred_accepted = int(acc["n_pred_accepted"])
    out: dict[str, object] = {
        "n_files": int(acc["n_files"]),
        "n_traces": int(acc["n_traces"]),
        "n_teacher": n_teacher,
        "n_pred_valid": int(acc["n_pred_valid"]),
        "n_pred_accepted": n_pred_accepted,
        "coverage": "" if n_teacher == 0 else float(n_pred_accepted / n_teacher),
    }

    if n_pred_accepted > 0:
        abs_error_parts = cast(list[np.ndarray], acc["abs_errors"])
        abs_all = np.concatenate(abs_error_parts, axis=0).astype(
            np.float64,
            copy=False,
        )
        out.update(
            {
                "accepted_mae_samples": float(np.mean(abs_all)),
                "accepted_bias_samples": float(acc["err_sum"]) / n_pred_accepted,
                "accepted_rmse_samples": float(
                    np.sqrt(float(acc["sq_err_sum"]) / n_pred_accepted)
                ),
                "accepted_median_abs_samples": float(np.percentile(abs_all, 50)),
                "accepted_p90_abs_samples": float(np.percentile(abs_all, 90)),
                "accepted_p95_abs_samples": float(np.percentile(abs_all, 95)),
                "accepted_p99_abs_samples": float(np.percentile(abs_all, 99)),
                "accepted_max_abs_samples": float(np.max(abs_all)),
            }
        )
    else:
        out.update(
            {
                "accepted_mae_samples": "",
                "accepted_bias_samples": "",
                "accepted_rmse_samples": "",
                "accepted_median_abs_samples": "",
                "accepted_p90_abs_samples": "",
                "accepted_p95_abs_samples": "",
                "accepted_p99_abs_samples": "",
                "accepted_max_abs_samples": "",
            }
        )

    conf_count = int(acc["conf_count"])
    out["mean_conf"] = (
        float(acc["conf_sum"]) / conf_count if conf_count > 0 else ""
    )

    acc_hits = cast(dict[float, int], acc["hit_counts"])
    for th in thresholds_samples:
        out[f"within_{th:g}_samples"] = (
            "" if n_teacher == 0 else float(int(acc_hits[float(th)]) / n_teacher)
        )
    return out


def require_payload_vector(
    payload: dict[str, np.ndarray],
    key: str,
    *,
    n_traces: int,
    tag: str,
) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"{tag}: missing key {key}")
    arr = np.asarray(payload[key])
    if arr.shape != (n_traces,):
        raise ValueError(f"{tag}: {key} shape {arr.shape} != ({n_traces},)")
    return arr


def scalar_payload_value(
    payload: dict[str, np.ndarray],
    key: str,
    trace_idx: int,
) -> object:
    if key not in payload:
        return ""
    return np.asarray(payload[key])[trace_idx].item()


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
        help="comma-separated stages: final,robust,coarse,center",
    )
    parser.add_argument(
        "--thresholds-samples",
        default=",".join(f"{x:g}" for x in DEFAULT_THRESHOLDS_SAMPLES),
        help="coverage thresholds in samples",
    )
    parser.add_argument(
        "--topk-per-file",
        type=int,
        default=20,
        help="number of worst final-pick traces to keep per file",
    )
    parser.add_argument(
        "--include-high-conf-final",
        action="store_true",
        help="also evaluate final picks only where high_conf_mask is true",
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
    thresholds_samples = parse_csv_numbers(args.thresholds_samples, dtype=float)

    specs = stage_specs(args.stages.split(","))
    acc_stages = list(specs)
    if args.include_high_conf_final:
        acc_stages.append("final_high_conf")

    per_data_rows: list[dict[str, object]] = []
    top_error_rows: list[dict[str, object]] = []
    fold_accs: dict[tuple[str, str], dict[str, object]] = {}
    global_accs: dict[str, dict[str, object]] = {
        stage: new_accumulator(thresholds_samples) for stage in acc_stages
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
                np.isfinite(fb_i) & (fb_i >= 0.0) & (fb_i < float(n_samples_orig))
            )
            n_teacher = int(np.count_nonzero(teacher_mask))

            file_stage_cache: dict[str, FileStageCacheValue] = {}

            for stage, (pick_key, conf_key) in specs.items():
                if pick_key not in payload:
                    if stage == "final" and pick_key == "final_pick_i":
                        raise KeyError("final stage evaluation requires final_pick_i")
                    raise KeyError(f"{final_path} missing key {pick_key}")

                pick = np.asarray(payload[pick_key], dtype=np.float64)
                if pick.shape != (n_traces,):
                    raise ValueError(
                        f"{tag}: {pick_key} shape {pick.shape} != ({n_traces},)"
                    )

                valid_pick = (
                    np.isfinite(pick) & (pick >= 0.0) & (pick < float(n_samples_orig))
                )
                reject_mask = np.zeros(n_traces, dtype=np.bool_)
                if stage in REJECT_MASK_STAGES:
                    if "reject_mask" not in payload:
                        raise KeyError(f"{final_path} missing reject_mask")
                    reject_mask = np.asarray(payload["reject_mask"], dtype=np.bool_)
                    if reject_mask.shape != (n_traces,):
                        raise ValueError(
                            f"{tag}: reject_mask shape {reject_mask.shape} "
                            f"!= ({n_traces},)"
                        )
                accepted_mask = valid_pick & (~reject_mask)

                conf_arr = None
                if conf_key is not None and conf_key in payload:
                    candidate = np.asarray(payload[conf_key], dtype=np.float64)
                    if candidate.shape == (n_traces,):
                        conf_arr = candidate

                metrics, err, abs_err, conf, valid_indices, hit_counts = (
                    evaluate_stage_metrics(
                        fb_i=fb_i,
                        pick=pick,
                        teacher_mask=teacher_mask,
                        valid_pick=valid_pick,
                        accepted_mask=accepted_mask,
                        conf_arr=conf_arr,
                        n_traces=n_traces,
                        thresholds_samples=thresholds_samples,
                    )
                )

                row = {
                    "run_id": args.run_id,
                    "fold": fold,
                    "data_name": tag,
                    "stage": stage,
                    "segy": segy,
                    "fb": fb_path,
                    "final_npz": str(final_path),
                    "dt_sec": dt_sec,
                    "n_samples_orig": n_samples_orig,
                    **metrics,
                }
                per_data_rows.append(row)
                fold_acc = fold_accs.setdefault(
                    (fold, stage), new_accumulator(thresholds_samples)
                )
                for acc in (fold_acc, global_accs[stage]):
                    update_accumulator(
                        acc,
                        n_traces=n_traces,
                        n_teacher=int(metrics["n_teacher"]),
                        n_pred_valid=int(metrics["n_pred_valid"]),
                        n_pred_accepted=int(metrics["n_pred_accepted"]),
                        err=err,
                        abs_err=abs_err,
                        conf=conf,
                        hit_counts=hit_counts,
                    )

                file_stage_cache[stage] = (err, abs_err, conf, valid_indices)

            if args.include_high_conf_final and "final" in file_stage_cache:
                if "high_conf_mask" not in payload:
                    raise KeyError(f"{final_path} missing high_conf_mask")
                if "final_pick_i" not in payload:
                    raise KeyError("final_high_conf evaluation requires final_pick_i")
                pick = np.asarray(payload["final_pick_i"], dtype=np.float64)
                conf_arr = np.asarray(
                    payload.get("final_conf", np.full(n_traces, np.nan)),
                    dtype=np.float64,
                )
                high_conf = np.asarray(payload["high_conf_mask"], dtype=np.bool_)
                valid_pick = (
                    np.isfinite(pick) & (pick >= 0.0) & (pick < float(n_samples_orig))
                )
                metrics, err, abs_err, conf, _, hit_counts = evaluate_stage_metrics(
                    fb_i=fb_i,
                    pick=pick,
                    teacher_mask=teacher_mask,
                    valid_pick=valid_pick,
                    accepted_mask=valid_pick & high_conf,
                    conf_arr=conf_arr if conf_arr.shape == (n_traces,) else None,
                    n_traces=n_traces,
                    thresholds_samples=thresholds_samples,
                )

                per_data_rows.append(
                    {
                        "run_id": args.run_id,
                        "fold": fold,
                        "data_name": tag,
                        "stage": "final_high_conf",
                        "segy": segy,
                        "fb": fb_path,
                        "final_npz": str(final_path),
                        "dt_sec": dt_sec,
                        "n_samples_orig": n_samples_orig,
                        **metrics,
                    }
                )
                fold_acc = fold_accs.setdefault(
                    (fold, "final_high_conf"), new_accumulator(thresholds_samples)
                )
                for acc in (fold_acc, global_accs["final_high_conf"]):
                    update_accumulator(
                        acc,
                        n_traces=n_traces,
                        n_teacher=int(metrics["n_teacher"]),
                        n_pred_valid=int(metrics["n_pred_valid"]),
                        n_pred_accepted=int(metrics["n_pred_accepted"]),
                        err=err,
                        abs_err=abs_err,
                        conf=conf,
                        hit_counts=hit_counts,
                    )

            # worst trace list for final stage only
            if "final" in file_stage_cache and args.topk_per_file > 0:
                err, abs_err, conf, valid_indices = file_stage_cache["final"]
                k = min(int(args.topk_per_file), int(abs_err.size))
                if k > 0:
                    if k < abs_err.size:
                        sel = np.argpartition(abs_err, -k)[-k:]
                        sel = sel[np.argsort(abs_err[sel])[::-1]]
                    else:
                        sel = np.argsort(abs_err)[::-1]

                    final_pick_i = require_payload_vector(
                        payload,
                        "final_pick_i",
                        n_traces=n_traces,
                        tag=tag,
                    )
                    for j in sel:
                        trace_idx = int(valid_indices[j])
                        top_error_rows.append(
                            {
                                "run_id": args.run_id,
                                "fold": fold,
                                "data_key": tag,
                                "stage": "final",
                                "segy_file": segy,
                                "fb_file": fb_path,
                                "final_npz": str(final_path),
                                "trace_index": trace_idx,
                                "ffid": scalar_payload_value(
                                    payload,
                                    "ffid_values",
                                    trace_idx,
                                ),
                                "chno": scalar_payload_value(
                                    payload,
                                    "chno_values",
                                    trace_idx,
                                ),
                                "fb_i": float(fb_i[trace_idx]),
                                "final_pick_i": int(final_pick_i[trace_idx]),
                                "error_samples": float(err[j]),
                                "abs_error_samples": float(abs_err[j]),
                                "error_ms": float(err[j] * dt_sec * 1000.0),
                                "abs_error_ms": float(abs_err[j] * dt_sec * 1000.0),
                                "coarse_pick_i": scalar_payload_value(
                                    payload,
                                    "coarse_pick_i",
                                    trace_idx,
                                ),
                                "robust_pick_i": scalar_payload_value(
                                    payload,
                                    "robust_pick_i",
                                    trace_idx,
                                ),
                                "window_start_i": scalar_payload_value(
                                    payload,
                                    "window_start_i",
                                    trace_idx,
                                ),
                                "window_end_i": scalar_payload_value(
                                    payload,
                                    "window_end_i",
                                    trace_idx,
                                ),
                                "fine_pick_local_i": scalar_payload_value(
                                    payload,
                                    "fine_pick_local_i",
                                    trace_idx,
                                ),
                                "final_conf": (
                                    float(payload["final_conf"][trace_idx])
                                    if "final_conf" in payload
                                    else ""
                                ),
                                "reject_mask": (
                                    bool(payload["reject_mask"][trace_idx])
                                    if "reject_mask" in payload
                                    else ""
                                ),
                                "reason_mask": (
                                    int(payload["reason_mask"][trace_idx])
                                    if "reason_mask" in payload
                                    else ""
                                ),
                            }
                        )

            n_files += 1
            print(f"[eval] {fold} {tag} done n_teacher={n_teacher}")

    per_fold_rows = [
        {
            "run_id": args.run_id,
            "fold": fold,
            "stage": stage,
            **accumulator_row(acc, thresholds_samples=thresholds_samples),
        }
        for (fold, stage), acc in sorted(fold_accs.items())
    ]
    summary_rows = [
        {
            "run_id": args.run_id,
            "stage": stage,
            **accumulator_row(
                global_accs[stage],
                thresholds_samples=thresholds_samples,
            ),
        }
        for stage in acc_stages
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_data.csv", per_data_rows)
    write_csv(out_dir / "per_fold.csv", per_fold_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "top_errors_final.csv", top_error_rows)

    meta = {
        "run_id": args.run_id,
        "run_root": str(run_root),
        "fold_list_root": str(fold_list_root),
        "pred_root": str(pred_root),
        "pred_stage_subdir": str(args.pred_stage_subdir),
        "out_dir": str(out_dir),
        "n_files": n_files,
        "stages": acc_stages,
        "thresholds_samples": list(thresholds_samples),
        "final_pick_key": "final_pick_i",
        "include_high_conf_final": bool(args.include_high_conf_final),
        "outputs": {
            "per_data_csv": str(out_dir / "per_data.csv"),
            "per_fold_csv": str(out_dir / "per_fold.csv"),
            "summary_csv": str(out_dir / "summary.csv"),
            "top_errors_final_csv": str(out_dir / "top_errors_final.csv"),
        },
    }
    (out_dir / "eval_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("wrote:")
    print(" ", out_dir / "per_data.csv")
    print(" ", out_dir / "per_fold.csv")
    print(" ", out_dir / "summary.csv")
    print(" ", out_dir / "top_errors_final.csv")
    print(" ", out_dir / "eval_meta.json")
    print()
    print("summary:")
    for row in summary_rows:
        print(
            row["stage"],
            "n_teacher=", row.get("n_teacher"),
            "n_pred_accepted=", row.get("n_pred_accepted"),
            "coverage=", row.get("coverage"),
            "accepted_mae_samples=", row.get("accepted_mae_samples"),
            "accepted_p90_abs_samples=", row.get("accepted_p90_abs_samples"),
            "within_4_samples=", row.get("within_4_samples"),
            "within_8_samples=", row.get("within_8_samples"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
