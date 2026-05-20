#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

FOLDS = [f"fold{i:02d}" for i in range(6)]
DEFAULT_CV_ROOT = Path("/workspace/proc/fbpick/site54/oof")
DEFAULT_RUN_ID = "baseline_physical_center"
FINE_TRAIN_HELDOUT_FORBIDDEN_KEYS = (
    "segy_files",
    "fb_files",
    "robust_npz_files",
    "infer_segy_files",
    "infer_fb_files",
    "infer_robust_npz_files",
)
FINE_INFER_REQUIRED_SINGLE_ENTRY_KEYS = (
    "segy_files",
    "viewer_fb_files",
    "robust_npz_files",
    "coarse_npz_files",
)
FINE_INFER_ALLOWED_PATH_KEYS = set(FINE_INFER_REQUIRED_SINGLE_ENTRY_KEYS) | {
    "fb_files",
    "out_dir",
}
STRICT_PHYSICS_KEYS = (
    "fine_center_i",
    "fine_window_valid_mask",
    "fine_window_physical_lo_i",
    "fine_window_physical_hi_i",
    "physical_model_status",
)
STRICT_FINAL_KEYS = (
    "final_pick_f",
    "n_traces",
    "n_samples_orig",
    "dt_sec",
    "reject_mask",
    "reason_mask",
)
PHYSICAL_MODEL_STATUS_LABELS = {
    0: "two_piece_ok",
    1: "relaxed_segment_ok",
    2: "fallback_existing_trend",
    3: "fallback_feasible_clip",
    4: "fallback_robust",
    5: "geometry_invalid",
    6: "insufficient_observations",
    7: "fit_failed",
    8: "physical_disabled",
    9: "single_line_ok",
    10: "neighbor_physical_fit_reuse",
    11: "coarse_in_band_fallback",
    12: "reject_physics_no_valid_window",
    13: "reject_physics_coarse_outside_band",
    14: "reject_physics_no_neighbor_fit",
}
RECOGNIZED_REJECT_PHYSICS_STATUSES = {12, 13, 14}
STRICT_FORBIDDEN_PHYSICAL_MODEL_STATUSES = {3, 4}


def read_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def fbpick_tag(segy_path: str | Path) -> str:
    p = Path(segy_path)
    return f"{p.parent.name}__{p.stem}" if p.parent.name else p.stem


def expected_output_path(out_dir: Path, segy_path: str, suffix: str) -> Path:
    return out_dir / f"{fbpick_tag(segy_path)}{suffix}"


def add_result(
    results: list[dict[str, Any]],
    *,
    fold: str,
    stage: str,
    ok: bool,
    detail: str = "",
    expected_paths: list[Path] | None = None,
) -> None:
    item: dict[str, Any] = {
        "fold": fold,
        "stage": stage,
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
    }
    if expected_paths:
        item["expected_paths"] = [str(p) for p in expected_paths]
    results.append(item)


def format_result(item: dict[str, Any]) -> str:
    line = f"{item['fold']} {item['stage']} {item['status']}"
    if item.get("detail"):
        line += f" {item['detail']}"
    return line


def load_fold_lists(
    *,
    fold_list_root: Path,
    results: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    payload: dict[str, dict[str, list[str]]] = {}
    heldout_seen: dict[str, str] = {}
    duplicate_heldout: list[str] = []
    total_heldout = 0

    for fold in FOLDS:
        fold_dir = fold_list_root / "folds" / fold
        lists: dict[str, list[str]] = {}
        failures: list[str] = []

        if not fold_dir.is_dir():
            add_result(
                results,
                fold=fold,
                stage="fold_lists",
                ok=False,
                detail=f"missing_dir={fold_dir}",
                expected_paths=[fold_dir],
            )
            payload[fold] = lists
            continue

        for name in (
            "train_sgy",
            "train_fb",
            "inner_valid_sgy",
            "inner_valid_fb",
            "heldout_sgy",
            "heldout_fb",
        ):
            path = fold_dir / f"{name}.txt"
            try:
                lists[name] = read_list(path)
            except FileNotFoundError:
                lists[name] = []
                failures.append(f"missing={path}")

        for split in ("train", "inner_valid", "heldout"):
            sgy = lists.get(f"{split}_sgy", [])
            fb = lists.get(f"{split}_fb", [])
            if len(sgy) != len(fb):
                failures.append(f"{split}_len={len(sgy)}/{len(fb)}")

        for suffix in ("sgy", "fb"):
            train = set(lists.get(f"train_{suffix}", []))
            valid = set(lists.get(f"inner_valid_{suffix}", []))
            heldout = set(lists.get(f"heldout_{suffix}", []))
            overlap = {
                "train_inner_valid": train & valid,
                "train_heldout": train & heldout,
                "inner_valid_heldout": valid & heldout,
            }
            bad = {k: sorted(v) for k, v in overlap.items() if v}
            if bad:
                failures.append(
                    f"{suffix}_overlap="
                    + ",".join(f"{k}:{len(v)}" for k, v in bad.items())
                )

        heldout_sgy = lists.get("heldout_sgy", [])
        for sgy in heldout_sgy:
            if sgy in heldout_seen:
                duplicate_heldout.append(sgy)
            heldout_seen[sgy] = fold
        total_heldout += len(heldout_sgy)

        detail = (
            ";".join(failures)
            if failures
            else f"train={len(lists.get('train_sgy', []))} "
            f"inner_valid={len(lists.get('inner_valid_sgy', []))} "
            f"heldout={len(heldout_sgy)}"
        )
        add_result(
            results,
            fold=fold,
            stage="fold_lists",
            ok=not failures,
            detail=detail,
        )
        payload[fold] = lists

    global_failures: list[str] = []
    if total_heldout != 54:
        global_failures.append(f"heldout_total={total_heldout}/54")
    if duplicate_heldout:
        global_failures.append(
            f"duplicate_heldout={len(set(duplicate_heldout))} "
            f"first={duplicate_heldout[0]}"
        )
    add_result(
        results,
        fold="all",
        stage="fold_lists_total",
        ok=not global_failures,
        detail="total_heldout=54" if not global_failures else ";".join(global_failures),
    )
    return payload


def check_ckpt(
    *,
    results: list[dict[str, Any]],
    fold: str,
    stage: str,
    run_root: Path,
    allow_smoke: bool,
) -> None:
    ckpt = run_root / fold / stage / "ckpt" / "best.pt"
    if ckpt.is_file():
        add_result(results, fold=fold, stage=stage, ok=True, detail=f"path={ckpt}")
        return

    smoke_stage = f"{stage}_smoke"
    smoke_ckpt = run_root / fold / smoke_stage / "ckpt" / "best.pt"
    if allow_smoke and smoke_ckpt.is_file():
        add_result(
            results,
            fold=fold,
            stage=stage,
            ok=True,
            detail=f"smoke_path={smoke_ckpt}",
        )
        return

    expected = [ckpt, smoke_ckpt] if allow_smoke else [ckpt]
    add_result(
        results,
        fold=fold,
        stage=stage,
        ok=False,
        detail=f"missing={ckpt}",
        expected_paths=expected,
    )


def check_npz_count(
    *,
    results: list[dict[str, Any]],
    fold: str,
    stage: str,
    out_dir: Path,
    heldout_sgy: list[str],
    suffix: str,
    strict_keys: tuple[str, ...] = (),
) -> None:
    expected_paths = [expected_output_path(out_dir, sgy, suffix) for sgy in heldout_sgy]
    if not out_dir.is_dir():
        add_result(
            results,
            fold=fold,
            stage=stage,
            ok=False,
            detail=f"missing_dir={out_dir}",
            expected_paths=[out_dir],
        )
        return

    files = sorted(out_dir.glob(f"*{suffix}"))
    found = {p.name for p in files}
    expected = {p.name for p in expected_paths}
    missing = [p for p in expected_paths if p.name not in found]
    extra = sorted(found - expected)
    failures: list[str] = []
    if missing:
        failures.append(f"missing={len(missing)} first={missing[0]}")
    if extra:
        failures.append(f"extra={len(extra)} first={extra[0]}")

    if strict_keys and not missing:
        for path in expected_paths:
            try:
                with np.load(path, allow_pickle=False) as z:
                    missing_keys = [key for key in strict_keys if key not in z.files]
            except Exception as exc:
                failures.append(f"read_error={path}:{exc}")
                break
            if missing_keys:
                failures.append(f"missing_keys={path}:{','.join(missing_keys)}")
                break

    add_result(
        results,
        fold=fold,
        stage=stage,
        ok=not failures,
        detail=f"n={len(files)}" if not failures else ";".join(failures),
        expected_paths=missing or ([] if not failures else expected_paths),
    )


def npz_n_traces(z: np.lib.npyio.NpzFile) -> int:
    """Read the scalar n_traces field from an NPZ payload."""
    return int(np.asarray(z["n_traces"]).reshape(()))


def require_trace_vector(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    n_traces: int,
    dtype: object | None = None,
) -> np.ndarray:
    """Return a 1D trace-aligned NPZ array or raise a shape error."""
    arr = np.asarray(z[key], dtype=dtype)
    if arr.ndim != 1 or int(arr.shape[0]) != int(n_traces):
        raise ValueError(f"{key}_shape={arr.shape}/{(n_traces,)}")
    return arr


def strict_check_physics_npz(path: Path) -> str | None:
    """Validate required physics NPZ keys and physical status values."""
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [key for key in STRICT_PHYSICS_KEYS if key not in z.files]
            if missing:
                return f"{path.name}:missing_keys={','.join(missing)}"
            n_traces = npz_n_traces(z)
            status = require_trace_vector(
                z,
                "physical_model_status",
                n_traces=n_traces,
                dtype=np.uint8,
            )
            for key in (
                "fine_center_i",
                "fine_window_valid_mask",
                "fine_window_physical_lo_i",
                "fine_window_physical_hi_i",
            ):
                require_trace_vector(z, key, n_traces=n_traces)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return f"{path.name}:read_error={exc}"

    unknown = sorted(
        {int(value) for value in np.unique(status)} - set(PHYSICAL_MODEL_STATUS_LABELS)
    )
    if unknown:
        return f"{path.name}:unknown_physical_model_status={unknown[0]}"

    forbidden = sorted(
        {int(value) for value in np.unique(status)}
        & STRICT_FORBIDDEN_PHYSICAL_MODEL_STATUSES
    )
    if forbidden:
        label = PHYSICAL_MODEL_STATUS_LABELS[forbidden[0]]
        return f"{path.name}:forbidden_physical_model_status={label}"

    if (
        "coarse_in_band_fallback" not in PHYSICAL_MODEL_STATUS_LABELS.values()
        or not any(
            PHYSICAL_MODEL_STATUS_LABELS[code].startswith("reject_physics")
            for code in RECOGNIZED_REJECT_PHYSICS_STATUSES
        )
    ):
        return "physical_model_status_labels_missing_new_policy"
    return None


def strict_check_final_npz(path: Path) -> str | None:
    """Validate required final NPZ keys added by the fine inference policy."""
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [key for key in STRICT_FINAL_KEYS if key not in z.files]
            if missing:
                return f"{path.name}:missing_keys={','.join(missing)}"
            n_traces = npz_n_traces(z)
            require_trace_vector(
                z,
                "reject_mask",
                n_traces=n_traces,
                dtype=np.bool_,
            )
            require_trace_vector(z, "reason_mask", n_traces=n_traces, dtype=np.uint8)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return f"{path.name}:read_error={exc}"
    return None


def strict_check_physics_final_pair(
    *,
    physics_path: Path,
    final_path: Path,
    time_len: int = 256,
) -> str | None:
    """Validate physics window masks against the paired final rejection mask."""
    try:
        with (
            np.load(physics_path, allow_pickle=False) as physics,
            np.load(final_path, allow_pickle=False) as final,
        ):
            n_traces = npz_n_traces(physics)
            final_n_traces = npz_n_traces(final)
            if final_n_traces != n_traces:
                return (
                    f"{final_path.name}:n_traces={final_n_traces}/"
                    f"{n_traces}"
                )

            valid = require_trace_vector(
                physics,
                "fine_window_valid_mask",
                n_traces=n_traces,
                dtype=np.bool_,
            )
            lo_i = require_trace_vector(
                physics,
                "fine_window_physical_lo_i",
                n_traces=n_traces,
                dtype=np.int64,
            )
            hi_i = require_trace_vector(
                physics,
                "fine_window_physical_hi_i",
                n_traces=n_traces,
                dtype=np.int64,
            )
            reject = require_trace_vector(
                final,
                "reject_mask",
                n_traces=n_traces,
                dtype=np.bool_,
            )

            invalid_not_rejected = np.flatnonzero((~valid) & (~reject))
            if invalid_not_rejected.size:
                return (
                    f"{final_path.name}:invalid_window_not_rejected="
                    f"{int(invalid_not_rejected[0])}"
                )

            if "window_start_i" in final.files and "window_end_i" in final.files:
                start = require_trace_vector(
                    final,
                    "window_start_i",
                    n_traces=n_traces,
                    dtype=np.int64,
                )
                end = require_trace_vector(
                    final,
                    "window_end_i",
                    n_traces=n_traces,
                    dtype=np.int64,
                )
                bad_window = np.flatnonzero(valid & ((start < lo_i) | (end > hi_i)))
                if bad_window.size:
                    return (
                        f"{final_path.name}:window_outside_physical_band="
                        f"{int(bad_window[0])}"
                    )
                bad_len = np.flatnonzero(valid & ((end - start) != (time_len - 1)))
                if bad_len.size:
                    return f"{final_path.name}:window_len={int(bad_len[0])}"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return f"{physics_path.name}:{final_path.name}:read_error={exc}"
    return None


def check_strict_npz_policy(
    *,
    results: list[dict[str, Any]],
    fold: str,
    run_root: Path,
    heldout_sgy: list[str],
) -> None:
    """Run strict new-policy checks for paired physics and final NPZ outputs."""
    failures: list[str] = []
    for segy_path in heldout_sgy:
        physics_path = expected_output_path(
            run_root / fold / "03_physics",
            segy_path,
            ".robust.npz",
        )
        final_path = expected_output_path(
            run_root / fold / "07_fine_infer",
            segy_path,
            ".fbpick_final.npz",
        )
        if not physics_path.is_file() or not final_path.is_file():
            continue
        error = (
            strict_check_physics_npz(physics_path)
            or strict_check_final_npz(final_path)
            or strict_check_physics_final_pair(
                physics_path=physics_path,
                final_path=final_path,
            )
        )
        if error:
            failures.append(error)
            break

    add_result(
        results,
        fold=fold,
        stage="strict_npz_policy",
        ok=not failures,
        detail=f"n={len(heldout_sgy)}" if not failures else ";".join(failures),
    )


def check_physics_qc(
    *,
    results: list[dict[str, Any]],
    fold: str,
    qc_dir: Path,
) -> None:
    if not qc_dir.is_dir():
        add_result(
            results,
            fold=fold,
            stage="04_physics_qc",
            ok=False,
            detail=f"missing_dir={qc_dir}",
            expected_paths=[qc_dir],
        )
        return

    summaries = [qc_dir / "summary_global.csv", qc_dir / "summary_per_file.csv"]
    existing = [p for p in summaries if p.is_file()]
    missing = [p for p in summaries if not p.is_file()]
    if existing and missing:
        add_result(
            results,
            fold=fold,
            stage="04_physics_qc",
            ok=False,
            detail=f"missing_summary={missing[0]}",
            expected_paths=missing,
        )
        return

    detail = "summary=present" if existing else f"dir={qc_dir}"
    add_result(results, fold=fold, stage="04_physics_qc", ok=True, detail=detail)


def check_collect_outputs(results: list[dict[str, Any]], *, run_root: Path) -> None:
    out_dir = run_root / "aggregate" / "05_collect_oof_lists"
    txt_files = [
        out_dir / "oof_train_sgy_all.txt",
        out_dir / "oof_train_fb_all.txt",
        out_dir / "oof_train_coarse_all.txt",
        out_dir / "oof_train_robust_all.txt",
    ]
    mapping = out_dir / "oof_train_mapping.csv"
    required = [*txt_files, mapping]
    missing = [path for path in required if not path.is_file()]
    if missing:
        add_result(
            results,
            fold="all",
            stage="05_collect_oof_lists",
            ok=False,
            detail=f"missing={missing[0]}",
            expected_paths=missing,
        )
        return

    lengths = {path.name: len(read_list(path)) for path in txt_files}
    mapping_rows = max(0, len(read_list(mapping)) - 1)
    failures: list[str] = []
    if len(set(lengths.values())) != 1:
        failures.append(
            "lengths=" + ",".join(f"{name}:{count}" for name, count in lengths.items())
        )
    n_rows = next(iter(lengths.values()))
    if n_rows != 54:
        failures.append(f"n={n_rows}/54")
    if mapping_rows != n_rows:
        failures.append(f"mapping_rows={mapping_rows}/{n_rows}")

    add_result(
        results,
        fold="all",
        stage="05_collect_oof_lists",
        ok=not failures,
        detail=f"n={n_rows} out_dir={out_dir}" if not failures else ";".join(failures),
        expected_paths=[] if not failures else required,
    )


def strict_check_fine_train_config(
    *,
    config_path: Path,
    fine_fold_dir: Path,
    heldout_sgy: list[str],
    heldout_fb: list[str],
) -> str | None:
    if not config_path.is_file():
        return f"missing_config={config_path}"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        return f"invalid_config={config_path}"
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        return f"missing_paths={config_path}"

    heldout_values = set(heldout_sgy) | set(heldout_fb)
    for name in ("heldout_robust", "heldout_coarse"):
        path = fine_fold_dir / f"{name}.txt"
        if path.is_file():
            heldout_values.update(read_list(path))

    for key in FINE_TRAIN_HELDOUT_FORBIDDEN_KEYS:
        raw = paths.get(key)
        if raw is None:
            continue
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if not isinstance(value, str):
                continue
            if Path(value).name.startswith("heldout_"):
                return f"{key}_uses_heldout={value}"
            value_path = Path(value)
            if not value_path.is_absolute():
                value_path = config_path.parent / value_path
            if value_path.is_file():
                overlap = set(read_list(value_path)) & heldout_values
                if overlap:
                    return (
                        f"{key}_heldout_overlap={len(overlap)} "
                        f"first={sorted(overlap)[0]}"
                    )

    return None


def strict_check_fine_infer_config(*, config_path: Path) -> str | None:
    if not config_path.is_file():
        return f"missing_config={config_path}"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        return f"invalid_config={config_path}"
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        return f"missing_paths={config_path}"

    unexpected = sorted(set(paths) - FINE_INFER_ALLOWED_PATH_KEYS)
    if unexpected:
        return f"unexpected_paths={','.join(unexpected)}"

    if paths.get("fb_files") is not None:
        return f"fb_files_set={paths['fb_files']}"

    for key in FINE_INFER_REQUIRED_SINGLE_ENTRY_KEYS:
        value = paths.get(key)
        if not (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], str)
        ):
            return f"{key}_not_single_entry={value}"

    for key, raw in paths.items():
        if key == "viewer_fb_files":
            continue
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, str) and Path(value).name == "heldout_fb.txt":
                return f"{key}_uses_heldout_fb={value}"

    return None


def check_eval(results: list[dict[str, Any]], *, run_root: Path) -> None:
    out_dir = run_root / "aggregate" / "08_eval"
    required = [
        out_dir / "per_data.csv",
        out_dir / "per_fold.csv",
        out_dir / "summary.csv",
        out_dir / "eval_meta.json",
    ]
    missing = [p for p in required if not p.is_file()]
    add_result(
        results,
        fold="all",
        stage="08_eval",
        ok=not missing,
        detail=f"out_dir={out_dir}" if not missing else f"missing={missing[0]}",
        expected_paths=missing,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check run-scoped site54 OOF CV outputs."
    )
    parser.add_argument("--cv-root", type=Path, default=DEFAULT_CV_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--fold-list-root", type=Path, default=None)
    parser.add_argument("--fine-list-root", type=Path, default=None)
    parser.add_argument("--config-root", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow smoke training checkpoints when full checkpoints are absent.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Check fine train heldout references and final NPZ schema keys.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cv_root = args.cv_root
    run_root = args.run_root or (cv_root / "runs" / args.run_id)
    fold_list_root = args.fold_list_root or (cv_root / "fold_lists")
    fine_list_root = (
        args.fine_list_root
        or run_root / "aggregate" / "05_collect_oof_lists" / "fine_fold_lists"
    )
    config_root = args.config_root or (run_root / "configs")

    results: list[dict[str, Any]] = []
    fold_lists = load_fold_lists(fold_list_root=fold_list_root, results=results)

    for fold in FOLDS:
        heldout_sgy = fold_lists.get(fold, {}).get("heldout_sgy", [])
        heldout_fb = fold_lists.get(fold, {}).get("heldout_fb", [])

        check_ckpt(
            results=results,
            fold=fold,
            stage="01_coarse_train",
            run_root=run_root,
            allow_smoke=args.smoke,
        )
        check_npz_count(
            results=results,
            fold=fold,
            stage="02_coarse_infer",
            out_dir=run_root / fold / "02_coarse_infer",
            heldout_sgy=heldout_sgy,
            suffix=".coarse.npz",
        )
        check_npz_count(
            results=results,
            fold=fold,
            stage="03_physics",
            out_dir=run_root / fold / "03_physics",
            heldout_sgy=heldout_sgy,
            suffix=".robust.npz",
            strict_keys=STRICT_PHYSICS_KEYS if args.strict else (),
        )
        check_physics_qc(
            results=results,
            fold=fold,
            qc_dir=run_root / fold / "04_physics_qc",
        )
        check_ckpt(
            results=results,
            fold=fold,
            stage="06_fine_train",
            run_root=run_root,
            allow_smoke=args.smoke,
        )
        if args.strict:
            strict_error = strict_check_fine_train_config(
                config_path=config_root / fold / "06_fine_train.yaml",
                fine_fold_dir=fine_list_root / fold,
                heldout_sgy=heldout_sgy,
                heldout_fb=heldout_fb,
            )
            if strict_error:
                add_result(
                    results,
                    fold=fold,
                    stage="06_fine_train_config",
                    ok=False,
                    detail=strict_error,
                )
            else:
                add_result(
                    results,
                    fold=fold,
                    stage="06_fine_train_config",
                    ok=True,
                    detail=f"path={config_root / fold / '06_fine_train.yaml'}",
                )
        check_npz_count(
            results=results,
            fold=fold,
            stage="07_fine_infer",
            out_dir=run_root / fold / "07_fine_infer",
            heldout_sgy=heldout_sgy,
            suffix=".fbpick_final.npz",
            strict_keys=STRICT_FINAL_KEYS if args.strict else (),
        )
        if args.strict:
            check_strict_npz_policy(
                results=results,
                fold=fold,
                run_root=run_root,
                heldout_sgy=heldout_sgy,
            )
            fine_infer_config_paths = [config_root / fold / "07_fine_infer.yaml"]
            fine_infer_config_paths.extend(
                sorted((config_root / fold).glob("07_fine_infer_*.yaml"))
            )
            strict_errors = [
                f"{path.name}:{error}"
                for path in fine_infer_config_paths
                if (error := strict_check_fine_infer_config(config_path=path))
            ]
            if len(fine_infer_config_paths) != len(heldout_sgy):
                strict_errors.append(
                    f"config_count={len(fine_infer_config_paths)}/"
                    f"{len(heldout_sgy)}"
                )
            configured_sgy: list[str] = []
            for path in fine_infer_config_paths:
                if not path.is_file():
                    continue
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                paths_cfg = cfg.get("paths") if isinstance(cfg, dict) else None
                value = (
                    paths_cfg.get("segy_files")
                    if isinstance(paths_cfg, dict)
                    else None
                )
                if isinstance(value, list) and len(value) == 1:
                    configured_sgy.append(str(value[0]))
            missing_sgy = sorted(set(heldout_sgy) - set(configured_sgy))
            extra_sgy = sorted(set(configured_sgy) - set(heldout_sgy))
            if missing_sgy:
                strict_errors.append(f"missing_sgy={len(missing_sgy)}")
            if extra_sgy:
                strict_errors.append(f"extra_sgy={len(extra_sgy)}")
            if strict_errors:
                add_result(
                    results,
                    fold=fold,
                    stage="07_fine_infer_config",
                    ok=False,
                    detail=";".join(strict_errors),
                )
            else:
                add_result(
                    results,
                    fold=fold,
                    stage="07_fine_infer_config",
                    ok=True,
                    detail=f"n={len(fine_infer_config_paths)}",
                )

    check_collect_outputs(results, run_root=run_root)
    check_eval(results, run_root=run_root)

    for item in results:
        print(format_result(item))
    passed = sum(1 for item in results if item["status"] == "PASS")
    failed = sum(1 for item in results if item["status"] == "FAIL")
    print(f"SUMMARY: failed={failed} passed={passed}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": args.run_id,
            "cv_root": str(cv_root),
            "run_root": str(run_root),
            "fold_list_root": str(fold_list_root),
            "fine_list_root": str(fine_list_root),
            "config_root": str(config_root),
            "strict": bool(args.strict),
            "smoke": bool(args.smoke),
            "summary": {"failed": failed, "passed": passed},
            "results": results,
        }
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
