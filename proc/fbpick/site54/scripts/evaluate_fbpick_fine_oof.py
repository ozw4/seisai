#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


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


def finite_float(value: object) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def metric_prefix_values(
    *,
    err_samples: np.ndarray,
    abs_samples: np.ndarray,
    dt_sec: float,
    conf: np.ndarray | None,
    n_traces: int,
    n_valid_fb: int,
    n_eval: int,
    n_invalid_fb: int,
    thresholds_samples: tuple[float, ...],
    thresholds_ms: tuple[float, ...],
) -> dict[str, object]:
    if n_eval == 0:
        out: dict[str, object] = {
            "n_traces": n_traces,
            "n_valid_fb": n_valid_fb,
            "n_eval": 0,
            "n_invalid_fb": n_invalid_fb,
            "mae_samples": "",
            "bias_samples": "",
            "median_abs_samples": "",
            "p90_abs_samples": "",
            "p95_abs_samples": "",
            "p99_abs_samples": "",
            "max_abs_samples": "",
            "mae_ms": "",
            "bias_ms": "",
            "median_abs_ms": "",
            "p90_abs_ms": "",
            "p95_abs_ms": "",
            "p99_abs_ms": "",
            "max_abs_ms": "",
            "mean_conf": "",
            "median_conf": "",
        }
        for th in thresholds_samples:
            out[f"within_{th:g}_samples"] = ""
        for th in thresholds_ms:
            out[f"within_{th:g}_ms"] = ""
        return out

    abs_ms = abs_samples * float(dt_sec) * 1000.0
    err_ms = err_samples * float(dt_sec) * 1000.0

    out = {
        "n_traces": int(n_traces),
        "n_valid_fb": int(n_valid_fb),
        "n_eval": int(n_eval),
        "n_invalid_fb": int(n_invalid_fb),
        "mae_samples": float(np.mean(abs_samples)),
        "bias_samples": float(np.mean(err_samples)),
        "median_abs_samples": float(np.percentile(abs_samples, 50)),
        "p90_abs_samples": float(np.percentile(abs_samples, 90)),
        "p95_abs_samples": float(np.percentile(abs_samples, 95)),
        "p99_abs_samples": float(np.percentile(abs_samples, 99)),
        "max_abs_samples": float(np.max(abs_samples)),
        "mae_ms": float(np.mean(abs_ms)),
        "bias_ms": float(np.mean(err_ms)),
        "median_abs_ms": float(np.percentile(abs_ms, 50)),
        "p90_abs_ms": float(np.percentile(abs_ms, 90)),
        "p95_abs_ms": float(np.percentile(abs_ms, 95)),
        "p99_abs_ms": float(np.percentile(abs_ms, 99)),
        "max_abs_ms": float(np.max(abs_ms)),
    }

    if conf is not None and conf.size:
        conf_finite = conf[np.isfinite(conf)]
        out["mean_conf"] = float(np.mean(conf_finite)) if conf_finite.size else ""
        out["median_conf"] = float(np.percentile(conf_finite, 50)) if conf_finite.size else ""
    else:
        out["mean_conf"] = ""
        out["median_conf"] = ""

    for th in thresholds_samples:
        out[f"within_{th:g}_samples"] = float(np.mean(abs_samples <= float(th)))
    for th in thresholds_ms:
        out[f"within_{th:g}_ms"] = float(np.mean(abs_ms <= float(th)))

    return out


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


def parse_csv_numbers(raw: str, *, dtype=float) -> tuple:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(dtype(item))
    return tuple(out)


def stage_specs(stages: Iterable[str]) -> dict[str, tuple[str, str | None]]:
    mapping = {
        "final": ("final_pick_f", "final_conf"),
        "robust": ("robust_pick_i", "robust_conf"),
        "coarse": ("coarse_pick_i", "coarse_pmax"),
        "center": ("center_raw_i", None),
    }
    out = {}
    for stage in stages:
        stage = stage.strip()
        if not stage:
            continue
        if stage not in mapping:
            raise ValueError(f"unknown stage={stage}; valid={sorted(mapping)}")
        out[stage] = mapping[stage]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-list-root",
        default="proc/fbpick/site54/oof/fine_fold_lists",
        help="foldXX/heldout_sgy.txt and heldout_fb.txt root",
    )
    parser.add_argument(
        "--pred-root",
        default="proc/fbpick/site54/oof/fine_infer",
        help="foldXX/*.fbpick_final.npz root",
    )
    parser.add_argument(
        "--out-dir",
        default="proc/fbpick/site54/oof/fine_eval",
        help="output report directory",
    )
    parser.add_argument(
        "--stages",
        default="final,robust,coarse",
        help="comma-separated stages: final,robust,coarse,center",
    )
    parser.add_argument(
        "--thresholds-samples",
        default="1,2,4,8,16,32",
        help="coverage thresholds in samples",
    )
    parser.add_argument(
        "--thresholds-ms",
        default="1,2,5,10,20,50",
        help="coverage thresholds in ms",
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

    fold_list_root = Path(args.fold_list_root)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    thresholds_samples = parse_csv_numbers(args.thresholds_samples, dtype=float)
    thresholds_ms = parse_csv_numbers(args.thresholds_ms, dtype=float)

    specs = stage_specs(args.stages.split(","))

    per_data_rows: list[dict[str, object]] = []
    top_error_rows: list[dict[str, object]] = []

    global_abs_by_stage: dict[str, list[np.ndarray]] = {stage: [] for stage in specs}
    global_err_sum_by_stage: dict[str, float] = {stage: 0.0 for stage in specs}
    global_conf_sum_by_stage: dict[str, float] = {stage: 0.0 for stage in specs}
    global_conf_count_by_stage: dict[str, int] = {stage: 0 for stage in specs}
    global_n_eval_by_stage: dict[str, int] = {stage: 0 for stage in specs}
    global_n_traces = 0
    global_n_valid_fb = 0
    global_n_invalid_fb = 0

    if args.include_high_conf_final:
        global_abs_by_stage["final_high_conf"] = []
        global_err_sum_by_stage["final_high_conf"] = 0.0
        global_conf_sum_by_stage["final_high_conf"] = 0.0
        global_conf_count_by_stage["final_high_conf"] = 0
        global_n_eval_by_stage["final_high_conf"] = 0

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
            final_path = pred_root / fold / build_final_npz_name(segy)
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

            valid_fb = np.isfinite(fb_i) & (fb_i >= 0.0) & (fb_i < float(n_samples_orig))
            n_valid_fb = int(np.count_nonzero(valid_fb))
            n_invalid_fb = int(n_traces - n_valid_fb)

            global_n_traces += n_traces
            global_n_valid_fb += n_valid_fb
            global_n_invalid_fb += n_invalid_fb

            file_stage_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]] = {}

            for stage, (pick_key, conf_key) in specs.items():
                if pick_key not in payload:
                    raise KeyError(f"{final_path} missing key {pick_key}")

                pick = np.asarray(payload[pick_key], dtype=np.float64)
                if pick.shape != (n_traces,):
                    raise ValueError(f"{tag}: {pick_key} shape {pick.shape} != ({n_traces},)")

                valid_pick = np.isfinite(pick) & (pick >= 0.0) & (pick < float(n_samples_orig))
                mask = valid_fb & valid_pick
                err = pick[mask] - fb_i[mask]
                abs_err = np.abs(err)

                conf = None
                if conf_key is not None and conf_key in payload:
                    conf_arr = np.asarray(payload[conf_key], dtype=np.float64)
                    if conf_arr.shape == (n_traces,):
                        conf = conf_arr[mask]

                row = {
                    "fold": fold,
                    "data_id": tag,
                    "stage": stage,
                    "segy": segy,
                    "fb": fb_path,
                    "final_npz": str(final_path),
                    "dt_sec": dt_sec,
                    "n_samples_orig": n_samples_orig,
                    **metric_prefix_values(
                        err_samples=err,
                        abs_samples=abs_err,
                        dt_sec=dt_sec,
                        conf=conf,
                        n_traces=n_traces,
                        n_valid_fb=n_valid_fb,
                        n_eval=int(mask.sum()),
                        n_invalid_fb=n_invalid_fb,
                        thresholds_samples=thresholds_samples,
                        thresholds_ms=thresholds_ms,
                    ),
                }
                per_data_rows.append(row)

                global_abs_by_stage[stage].append(abs_err.astype(np.float32, copy=False))
                global_err_sum_by_stage[stage] += float(np.sum(err, dtype=np.float64))
                global_n_eval_by_stage[stage] += int(err.size)
                if conf is not None and conf.size:
                    conf = conf[np.isfinite(conf)]
                    global_conf_sum_by_stage[stage] += float(np.sum(conf, dtype=np.float64))
                    global_conf_count_by_stage[stage] += int(conf.size)

                file_stage_cache[stage] = (err, abs_err, conf, np.flatnonzero(mask))

            if args.include_high_conf_final and "final" in file_stage_cache:
                if "high_conf_mask" not in payload:
                    raise KeyError(f"{final_path} missing high_conf_mask")
                pick = np.asarray(payload["final_pick_f"], dtype=np.float64)
                conf_arr = np.asarray(payload.get("final_conf", np.full(n_traces, np.nan)), dtype=np.float64)
                high_conf = np.asarray(payload["high_conf_mask"], dtype=np.bool_)
                mask = (
                    valid_fb
                    & high_conf
                    & np.isfinite(pick)
                    & (pick >= 0.0)
                    & (pick < float(n_samples_orig))
                )
                err = pick[mask] - fb_i[mask]
                abs_err = np.abs(err)
                conf = conf_arr[mask] if conf_arr.shape == (n_traces,) else None

                per_data_rows.append(
                    {
                        "fold": fold,
                        "data_id": tag,
                        "stage": "final_high_conf",
                        "segy": segy,
                        "fb": fb_path,
                        "final_npz": str(final_path),
                        "dt_sec": dt_sec,
                        "n_samples_orig": n_samples_orig,
                        **metric_prefix_values(
                            err_samples=err,
                            abs_samples=abs_err,
                            dt_sec=dt_sec,
                            conf=conf,
                            n_traces=n_traces,
                            n_valid_fb=n_valid_fb,
                            n_eval=int(mask.sum()),
                            n_invalid_fb=n_invalid_fb,
                            thresholds_samples=thresholds_samples,
                            thresholds_ms=thresholds_ms,
                        ),
                    }
                )
                global_abs_by_stage["final_high_conf"].append(abs_err.astype(np.float32, copy=False))
                global_err_sum_by_stage["final_high_conf"] += float(np.sum(err, dtype=np.float64))
                global_n_eval_by_stage["final_high_conf"] += int(err.size)
                if conf is not None and conf.size:
                    conf = conf[np.isfinite(conf)]
                    global_conf_sum_by_stage["final_high_conf"] += float(np.sum(conf, dtype=np.float64))
                    global_conf_count_by_stage["final_high_conf"] += int(conf.size)

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

                    final_pick = np.asarray(payload["final_pick_f"], dtype=np.float64)
                    for j in sel:
                        trace_idx = int(valid_indices[j])
                        top_error_rows.append(
                            {
                                "fold": fold,
                                "data_id": tag,
                                "trace_idx": trace_idx,
                                "fb_i": float(fb_i[trace_idx]),
                                "final_pick_f": float(final_pick[trace_idx]),
                                "error_samples": float(err[j]),
                                "abs_error_samples": float(abs_err[j]),
                                "error_ms": float(err[j] * dt_sec * 1000.0),
                                "abs_error_ms": float(abs_err[j] * dt_sec * 1000.0),
                                "final_conf": (
                                    float(payload["final_conf"][trace_idx])
                                    if "final_conf" in payload
                                    else ""
                                ),
                                "ffid": (
                                    int(payload["ffid_values"][trace_idx])
                                    if "ffid_values" in payload
                                    else ""
                                ),
                                "chno": (
                                    int(payload["chno_values"][trace_idx])
                                    if "chno_values" in payload
                                    else ""
                                ),
                                "segy": segy,
                                "fb": fb_path,
                                "final_npz": str(final_path),
                            }
                        )

            n_files += 1
            print(f"[eval] {fold} {tag} done n_valid_fb={n_valid_fb}")

    summary_rows: list[dict[str, object]] = []

    for stage in global_abs_by_stage:
        n_eval = int(global_n_eval_by_stage[stage])
        if n_eval > 0:
            abs_all = np.concatenate(global_abs_by_stage[stage], axis=0).astype(np.float64, copy=False)
            err_mean = global_err_sum_by_stage[stage] / float(n_eval)
            mae = float(np.mean(abs_all))
            dt_note = "mixed_dt"  # ms summary computed from sample metrics is not globally exact if dt differs.
            conf_mean = (
                global_conf_sum_by_stage[stage] / float(global_conf_count_by_stage[stage])
                if global_conf_count_by_stage[stage] > 0
                else ""
            )
            row = {
                "stage": stage,
                "n_files": n_files,
                "n_traces": global_n_traces,
                "n_valid_fb": global_n_valid_fb,
                "n_eval": n_eval,
                "n_invalid_fb": global_n_invalid_fb,
                "mae_samples": mae,
                "bias_samples": err_mean,
                "median_abs_samples": float(np.percentile(abs_all, 50)),
                "p90_abs_samples": float(np.percentile(abs_all, 90)),
                "p95_abs_samples": float(np.percentile(abs_all, 95)),
                "p99_abs_samples": float(np.percentile(abs_all, 99)),
                "max_abs_samples": float(np.max(abs_all)),
                "mean_conf": conf_mean,
                "dt_note": dt_note,
            }
            for th in thresholds_samples:
                row[f"within_{th:g}_samples"] = float(np.mean(abs_all <= float(th)))
        else:
            row = {
                "stage": stage,
                "n_files": n_files,
                "n_traces": global_n_traces,
                "n_valid_fb": global_n_valid_fb,
                "n_eval": 0,
                "n_invalid_fb": global_n_invalid_fb,
            }
        summary_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_data.csv", per_data_rows)
    write_csv(out_dir / "summary_by_stage.csv", summary_rows)
    write_csv(out_dir / "top_errors_final.csv", top_error_rows)

    meta = {
        "fold_list_root": str(fold_list_root),
        "pred_root": str(pred_root),
        "out_dir": str(out_dir),
        "n_files": n_files,
        "stages": list(global_abs_by_stage),
        "thresholds_samples": list(thresholds_samples),
        "thresholds_ms": list(thresholds_ms),
        "include_high_conf_final": bool(args.include_high_conf_final),
        "outputs": {
            "per_data_csv": str(out_dir / "per_data.csv"),
            "summary_by_stage_csv": str(out_dir / "summary_by_stage.csv"),
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
    print(" ", out_dir / "summary_by_stage.csv")
    print(" ", out_dir / "top_errors_final.csv")
    print(" ", out_dir / "eval_meta.json")
    print()
    print("summary:")
    for row in summary_rows:
        print(
            row["stage"],
            "n_eval=", row.get("n_eval"),
            "mae_samples=", row.get("mae_samples"),
            "p90_abs_samples=", row.get("p90_abs_samples"),
            "p95_abs_samples=", row.get("p95_abs_samples"),
            "p99_abs_samples=", row.get("p99_abs_samples"),
            "within_4_samples=", row.get("within_4_samples"),
            "within_8_samples=", row.get("within_8_samples"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
