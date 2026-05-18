#!/usr/bin/env python3
"""Render site54 OOF fine train/infer configs."""
from __future__ import annotations

import argparse
import copy
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import yaml

FOLDS = [f"fold{i:02d}" for i in range(6)]
VALID_POLICIES = (
    "inner_valid_from_nonheldout",
    "fixed_last",
    "heldout_metric_legacy",
)
TRAIN_HELDOUT_FORBIDDEN_KEYS = (
    "segy_files",
    "fb_files",
    "robust_npz_files",
    "infer_segy_files",
    "infer_fb_files",
    "infer_robust_npz_files",
)
INFER_REQUIRED_SINGLE_ENTRY_KEYS = (
    "segy_files",
    "robust_npz_files",
    "coarse_npz_files",
)
INFER_ALLOWED_PATH_KEYS = set(INFER_REQUIRED_SINGLE_ENTRY_KEYS) | {"fb_files", "out_dir"}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def read_plain_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def write_plain_list(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = list(values)
    path.write_text("\n".join(vals) + ("\n" if vals else ""), encoding="utf-8")


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def normalized_path_key(path_str: str, root: Path) -> str:
    return str(resolve_path(path_str, root).resolve())


def select_inner_valid_indices(
    *,
    nonheldout_idx: list[int],
    size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if size <= 0:
        raise ValueError("--fine-inner-valid-size must be positive")
    if size >= len(nonheldout_idx):
        raise ValueError(
            "--fine-inner-valid-size must be smaller than the non-heldout count "
            f"({len(nonheldout_idx)})"
        )
    shuffled = list(nonheldout_idx)
    random.Random(seed).shuffle(shuffled)
    inner_set = set(shuffled[:size])
    train_idx = [idx for idx in nonheldout_idx if idx not in inner_set]
    inner_idx = [idx for idx in nonheldout_idx if idx in inner_set]
    return train_idx, inner_idx


def load_base_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError(f"base config is not a YAML mapping: {path}")
    return cfg


def write_yaml(path: Path, data: dict, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def ensure_no_heldout_train_paths(*, cfg: dict, config_name: str, policy: str) -> None:
    if policy == "heldout_metric_legacy":
        return
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError(f"{config_name}: paths must be a mapping")
    for key in TRAIN_HELDOUT_FORBIDDEN_KEYS:
        raw = paths.get(key)
        if raw is None:
            continue
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, str) and Path(value).name.startswith("heldout_"):
                raise RuntimeError(
                    f"{config_name}: paths.{key} uses heldout list {value}"
                )


def ensure_fine_infer_heldout_policy(*, cfg: dict, config_name: str) -> None:
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError(f"{config_name}: paths must be a mapping")
    unexpected = sorted(set(paths) - INFER_ALLOWED_PATH_KEYS)
    if unexpected:
        raise RuntimeError(
            f"{config_name}: unexpected fine infer paths: {unexpected}"
        )
    if "fb_files" in paths and paths["fb_files"] is not None:
        raise RuntimeError(f"{config_name}: paths.fb_files must not be set")
    for key in INFER_REQUIRED_SINGLE_ENTRY_KEYS:
        value = paths.get(key)
        if not (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], str)
        ):
            raise RuntimeError(
                f"{config_name}: paths.{key} must be a single-entry list, "
                f"got {value}"
            )
    for raw in paths.values():
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, str) and Path(value).name == "heldout_fb.txt":
                raise RuntimeError(f"{config_name}: heldout FB is reserved for 08_eval")


def set_smoke_overrides(cfg: dict) -> None:
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = 1
    cfg["train"]["samples_per_epoch"] = 4
    cfg["train"]["batch_size"] = 1
    cfg.setdefault("infer", {})
    cfg["infer"]["batch_size"] = 1
    cfg["infer"]["max_batches"] = 1
    cfg.setdefault("vis", {})
    cfg["vis"]["n"] = 0


def fine_train_config(
    *,
    base_cfg: dict,
    paths: dict[str, Path],
    out_dir: Path,
    policy: str,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("paths", {})
    cfg["paths"]["segy_files"] = str(paths["train_sgy"])
    cfg["paths"]["fb_files"] = str(paths["train_fb"])
    cfg["paths"]["robust_npz_files"] = str(paths["train_robust"])

    if policy == "heldout_metric_legacy":
        cfg["paths"]["infer_segy_files"] = str(paths["heldout_sgy"])
        cfg["paths"]["infer_fb_files"] = str(paths["heldout_fb"])
        cfg["paths"]["infer_robust_npz_files"] = str(paths["heldout_robust"])
    elif policy == "fixed_last":
        cfg["paths"]["infer_segy_files"] = str(paths["train_sgy"])
        cfg["paths"]["infer_fb_files"] = str(paths["train_fb"])
        cfg["paths"]["infer_robust_npz_files"] = str(paths["train_robust"])
    else:
        cfg["paths"]["infer_segy_files"] = str(paths["inner_valid_sgy"])
        cfg["paths"]["infer_fb_files"] = str(paths["inner_valid_fb"])
        cfg["paths"]["infer_robust_npz_files"] = str(paths["inner_valid_robust"])

    cfg["paths"]["out_dir"] = str(out_dir)
    cfg["window_center"] = {
        "npz_key": "physical_center_i",
        "fallback_npz_key": None,
    }
    cfg.setdefault("ckpt", {})
    if policy == "fixed_last":
        cfg["ckpt"]["save_best_only"] = False
        cfg["ckpt"]["metric"] = "last"
        cfg["ckpt"]["mode"] = "max"
    else:
        cfg["ckpt"]["save_best_only"] = True
        cfg["ckpt"]["metric"] = "infer_loss"
        cfg["ckpt"]["mode"] = "min"
    return cfg


def fine_infer_config(
    *,
    base_cfg: dict,
    segy_file: str,
    robust_npz_file: str,
    coarse_npz_file: str,
    out_dir: Path,
    ckpt_path: Path,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("paths", {})
    cfg["paths"]["segy_files"] = [segy_file]
    cfg["paths"].pop("fb_files", None)
    cfg["paths"]["robust_npz_files"] = [robust_npz_file]
    cfg["paths"]["coarse_npz_files"] = [coarse_npz_file]
    cfg["paths"]["out_dir"] = str(out_dir)
    cfg.setdefault("infer", {})
    cfg["infer"]["ckpt_path"] = str(ckpt_path)
    return cfg


def fine_infer_config_name(index: int) -> str:
    if index == 0:
        return "07_fine_infer.yaml"
    return f"07_fine_infer_{index:03d}.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render run-scoped site54 fine OOF train/infer configs."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=Path("/workspace/proc/fbpick/site54/oof"),
    )
    parser.add_argument("--run-id", default="baseline_physical_center")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--fold-list-root", type=Path, default=None)
    parser.add_argument("--config-root", type=Path, default=None)
    parser.add_argument("--fine-list-root", type=Path, default=None)
    parser.add_argument(
        "--oof-list-dir",
        type=Path,
        default=None,
        help="directory containing oof_train_{sgy,fb,robust,coarse}_all.txt",
    )
    parser.add_argument(
        "--base-train-config",
        type=Path,
        default=None,
        help=(
            "fine train base config; defaults to "
            "<cv-root>/config_templates/fine_train.yaml"
        ),
    )
    parser.add_argument(
        "--base-infer-config",
        type=Path,
        default=None,
        help=(
            "fine infer base config; defaults to "
            "<cv-root>/config_templates/fine_infer.yaml"
        ),
    )
    parser.add_argument(
        "--fine-valid-policy",
        choices=VALID_POLICIES,
        default="inner_valid_from_nonheldout",
    )
    parser.add_argument("--fine-inner-valid-size", type=int, default=2)
    parser.add_argument("--fine-inner-valid-seed", type=int, default=0)
    parser.add_argument(
        "--legacy-flat-configs",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
    )
    parser.add_argument(
        "--strict-heldout-once",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    cv_root = args.cv_root.resolve()
    run_root = (args.run_root or cv_root / "runs" / args.run_id).resolve()
    fold_list_root = (args.fold_list_root or cv_root / "fold_lists").resolve()
    config_root = (args.config_root or run_root / "configs").resolve()
    fine_list_root = (
        args.fine_list_root
        or run_root / "aggregate" / "05_collect_oof_lists" / "fine_fold_lists"
    ).resolve()
    oof_list_dir = (
        args.oof_list_dir
        or run_root / "aggregate" / "05_collect_oof_lists"
    ).resolve()

    base_train_config = (
        cv_root / "config_templates" / "fine_train.yaml"
        if args.base_train_config is None
        else resolve_path(args.base_train_config, repo_root)
    )
    base_infer_config = (
        cv_root / "config_templates" / "fine_infer.yaml"
        if args.base_infer_config is None
        else resolve_path(args.base_infer_config, repo_root)
    )
    base_train_cfg = load_base_yaml(base_train_config)
    base_infer_cfg = load_base_yaml(base_infer_config)

    if args.fine_valid_policy == "heldout_metric_legacy":
        print(
            "WARNING: heldout_metric_legacy uses each fold's heldout list for "
            "fine checkpoint selection. This leaks heldout feedback into model "
            "selection and is not valid for strict CV."
        )
    elif args.fine_valid_policy == "fixed_last":
        print(
            "fixed_last requested: generated configs use ckpt.metric=last and "
            "ckpt/best.pt is overwritten by each later epoch. Heldout and "
            "inner-validation lists are not used for checkpoint selection."
        )

    all_sgy = read_plain_list(oof_list_dir / "oof_train_sgy_all.txt")
    all_fb = read_plain_list(oof_list_dir / "oof_train_fb_all.txt")
    all_robust = read_plain_list(oof_list_dir / "oof_train_robust_all.txt")
    all_coarse = read_plain_list(oof_list_dir / "oof_train_coarse_all.txt")
    lengths = {len(all_sgy), len(all_fb), len(all_robust), len(all_coarse)}
    if len(lengths) != 1:
        raise RuntimeError(
            "OOF list length mismatch: "
            f"sgy={len(all_sgy)} fb={len(all_fb)} "
            f"robust={len(all_robust)} coarse={len(all_coarse)}"
        )
    if not all_sgy:
        raise RuntimeError("OOF all lists are empty")

    all_keys = [normalized_path_key(path, repo_root) for path in all_sgy]
    dup_all = [key for key, count in Counter(all_keys).items() if count > 1]
    if dup_all:
        raise RuntimeError(f"duplicate SGY entries in OOF all list: {dup_all[:5]}")
    sgy_to_idx = {key: idx for idx, key in enumerate(all_keys)}

    generated: list[Path] = []
    heldout_all_keys: list[str] = []

    for fold_no, fold in enumerate(FOLDS):
        fold_root = fold_list_root / "folds" / fold
        heldout_sgy_from_fold = read_plain_list(fold_root / "heldout_sgy.txt")
        held_keys = [
            normalized_path_key(path, repo_root) for path in heldout_sgy_from_fold
        ]
        unknown = sorted(set(held_keys) - set(sgy_to_idx))
        if unknown:
            raise RuntimeError(
                f"{fold}: heldout SGY not found in OOF all list. "
                f"First unknown entries: {unknown[:5]}"
            )
        held_idx = [sgy_to_idx[key] for key in held_keys]
        if len(held_idx) != len(set(held_idx)):
            raise RuntimeError(f"{fold}: duplicate heldout SGY entries")

        held_set = set(held_idx)
        nonheldout_idx = [idx for idx in range(len(all_sgy)) if idx not in held_set]
        if args.fine_valid_policy in {"heldout_metric_legacy", "fixed_last"}:
            train_idx = nonheldout_idx
            inner_idx: list[int] = []
        else:
            train_idx, inner_idx = select_inner_valid_indices(
                nonheldout_idx=nonheldout_idx,
                size=args.fine_inner_valid_size,
                seed=args.fine_inner_valid_seed + fold_no,
            )

        fold_list_dir = fine_list_root / fold
        paths = {
            "train_sgy": fold_list_dir / "train_sgy.txt",
            "train_fb": fold_list_dir / "train_fb.txt",
            "train_robust": fold_list_dir / "train_robust.txt",
            "train_coarse": fold_list_dir / "train_coarse.txt",
            "inner_valid_sgy": fold_list_dir / "inner_valid_sgy.txt",
            "inner_valid_fb": fold_list_dir / "inner_valid_fb.txt",
            "inner_valid_robust": fold_list_dir / "inner_valid_robust.txt",
            "inner_valid_coarse": fold_list_dir / "inner_valid_coarse.txt",
            "heldout_sgy": fold_list_dir / "heldout_sgy.txt",
            "heldout_fb": fold_list_dir / "heldout_fb.txt",
            "heldout_robust": fold_list_dir / "heldout_robust.txt",
            "heldout_coarse": fold_list_dir / "heldout_coarse.txt",
        }

        values = {
            "train_sgy": [all_sgy[idx] for idx in train_idx],
            "train_fb": [all_fb[idx] for idx in train_idx],
            "train_robust": [all_robust[idx] for idx in train_idx],
            "train_coarse": [all_coarse[idx] for idx in train_idx],
            "inner_valid_sgy": [all_sgy[idx] for idx in inner_idx],
            "inner_valid_fb": [all_fb[idx] for idx in inner_idx],
            "inner_valid_robust": [all_robust[idx] for idx in inner_idx],
            "inner_valid_coarse": [all_coarse[idx] for idx in inner_idx],
            "heldout_sgy": [all_sgy[idx] for idx in held_idx],
            "heldout_fb": [all_fb[idx] for idx in held_idx],
            "heldout_robust": [all_robust[idx] for idx in held_idx],
            "heldout_coarse": [all_coarse[idx] for idx in held_idx],
        }

        train_cfg = fine_train_config(
            base_cfg=base_train_cfg,
            paths=paths,
            out_dir=run_root / fold / "06_fine_train",
            policy=args.fine_valid_policy,
        )
        smoke_cfg = fine_train_config(
            base_cfg=base_train_cfg,
            paths=paths,
            out_dir=run_root / fold / "06_fine_train_smoke",
            policy=args.fine_valid_policy,
        )
        set_smoke_overrides(smoke_cfg)
        ensure_no_heldout_train_paths(
            cfg=train_cfg,
            config_name=f"{fold}/06_fine_train.yaml",
            policy=args.fine_valid_policy,
        )
        ensure_no_heldout_train_paths(
            cfg=smoke_cfg,
            config_name=f"{fold}/06_fine_train_smoke.yaml",
            policy=args.fine_valid_policy,
        )

        fold_config_dir = config_root / fold
        out_train_cfg = fold_config_dir / "06_fine_train.yaml"
        out_smoke_cfg = fold_config_dir / "06_fine_train_smoke.yaml"
        infer_cfgs: list[tuple[Path, dict]] = []
        ckpt_path = run_root / fold / "06_fine_train" / "ckpt" / "best.pt"
        for infer_idx, all_idx in enumerate(held_idx):
            out_infer_cfg = fold_config_dir / fine_infer_config_name(infer_idx)
            infer_cfg = fine_infer_config(
                base_cfg=base_infer_cfg,
                segy_file=all_sgy[all_idx],
                robust_npz_file=all_robust[all_idx],
                coarse_npz_file=all_coarse[all_idx],
                out_dir=run_root / fold / "07_fine_infer",
                ckpt_path=ckpt_path,
            )
            ensure_fine_infer_heldout_policy(
                cfg=infer_cfg,
                config_name=f"{fold}/{out_infer_cfg.name}",
            )
            infer_cfgs.append((out_infer_cfg, infer_cfg))

        print(
            f"{fold}: train={len(train_idx)} inner_valid={len(inner_idx)} "
            f"heldout={len(held_idx)} fine_infer_configs={len(infer_cfgs)} "
            f"-> {fold_config_dir}"
        )
        if not args.dry_run:
            for key, path in paths.items():
                write_plain_list(path, values[key])
        write_yaml(out_train_cfg, train_cfg, dry_run=args.dry_run)
        write_yaml(out_smoke_cfg, smoke_cfg, dry_run=args.dry_run)
        for out_infer_cfg, infer_cfg in infer_cfgs:
            write_yaml(out_infer_cfg, infer_cfg, dry_run=args.dry_run)
        generated.extend([out_train_cfg, out_smoke_cfg])
        generated.extend(path for path, _ in infer_cfgs)

        if args.legacy_flat_configs:
            write_yaml(
                config_root / f"config_train_fbpick_fine_oof_{fold}.yaml",
                train_cfg,
                dry_run=args.dry_run,
            )
            write_yaml(
                config_root / f"config_train_fbpick_fine_oof_{fold}_smoke.yaml",
                smoke_cfg,
                dry_run=args.dry_run,
            )
            write_yaml(
                config_root / f"config_infer_fbpick_fine_oof_{fold}.yaml",
                infer_cfgs[0][1],
                dry_run=args.dry_run,
            )
        heldout_all_keys.extend(held_keys)

    held_counter = Counter(heldout_all_keys)
    missing = sorted(set(all_keys) - set(heldout_all_keys))
    duplicated = sorted(key for key, count in held_counter.items() if count > 1)

    print()
    print("coverage summary")
    print(f"  n_all:          {len(all_sgy)}")
    print(f"  n_held_total:   {len(heldout_all_keys)}")
    print(f"  n_held_unique:  {len(set(heldout_all_keys))}")
    print(f"  missing:        {len(missing)}")
    print(f"  duplicated:     {len(duplicated)}")

    if args.strict_heldout_once and (
        missing or duplicated or len(heldout_all_keys) != len(all_sgy)
    ):
        raise RuntimeError("heldout coverage is not one-to-one over the OOF all list")

    print()
    if args.dry_run:
        print("dry-run only; no files written")
    else:
        print("wrote fine configs:")
        for path in generated:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
