#!/usr/bin/env python3
"""Create fold-wise fine OOF training configs for site54 fbpick.

This script uses:
  - proc/fbpick/site54/oof/lists/oof_train_{sgy,fb,robust}_all.txt
  - proc/fbpick/site54/oof/configs/config_run_fbpick_physics_fold??_heldout.yaml
  - proc/fbpick/site54/configs/config_train_fbpick_fine_oof.yaml

It writes:
  - proc/fbpick/site54/oof/fine_fold_lists/foldXX/{train,heldout}_{sgy,fb,robust}.txt
  - proc/fbpick/site54/oof/configs/config_train_fbpick_fine_oof_foldXX.yaml
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml


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


def resolve_maybe_relative(value: str, root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def read_list_or_values(value, root: Path) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if not isinstance(value, str):
        raise TypeError(f"expected list or str for file list, got {type(value).__name__}: {value!r}")

    p = resolve_maybe_relative(value, root)
    if p.is_file():
        return read_plain_list(p)
    return [value]


def normalized_path_key(path_str: str, root: Path) -> str:
    return str(resolve_maybe_relative(path_str, root).resolve())


def extract_fold_name(path: Path) -> str:
    name = path.name
    prefix = "config_run_fbpick_physics_"
    suffix = "_heldout.yaml"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"unexpected physics config name: {path}")
    return name[len(prefix) : -len(suffix)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create fold-wise fine OOF train configs from existing fbpick OOF lists."
    )
    parser.add_argument("--root", default="/workspace", help="repository root, default: /workspace")
    parser.add_argument(
        "--base-config",
        default="proc/fbpick/site54/configs/config_train_fbpick_fine_oof.yaml",
        help="base fine OOF train config",
    )
    parser.add_argument(
        "--oof-list-dir",
        default="proc/fbpick/site54/oof/lists",
        help="directory containing oof_train_*_all.txt",
    )
    parser.add_argument(
        "--physics-config-glob",
        default="proc/fbpick/site54/oof/configs/config_run_fbpick_physics_fold??_heldout.yaml",
        help="glob for fold heldout physics configs",
    )
    parser.add_argument(
        "--out-list-dir",
        default="proc/fbpick/site54/oof/fine_fold_lists",
        help="output directory for fold-wise fine train/heldout lists",
    )
    parser.add_argument(
        "--out-config-dir",
        default="proc/fbpick/site54/oof/configs",
        help="output directory for generated fine fold configs",
    )
    parser.add_argument(
        "--out-dir-prefix",
        default="proc/fbpick/site54/fbpick_fine_train_oof",
        help=(
            "prefix for each fine fold train output dir; generated dirs are "
            "<root>/<prefix>_foldXX_out"
        ),
    )
    parser.add_argument(
        "--strict-heldout-once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require heldout union to cover all OOF SGY exactly once, default: true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned outputs without writing files",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    base_config = resolve_maybe_relative(args.base_config, root)
    oof_list_dir = resolve_maybe_relative(args.oof_list_dir, root)
    out_list_dir = resolve_maybe_relative(args.out_list_dir, root)
    out_config_dir = resolve_maybe_relative(args.out_config_dir, root)

    all_sgy = read_plain_list(oof_list_dir / "oof_train_sgy_all.txt")
    all_fb = read_plain_list(oof_list_dir / "oof_train_fb_all.txt")
    all_robust = read_plain_list(oof_list_dir / "oof_train_robust_all.txt")

    if not (len(all_sgy) == len(all_fb) == len(all_robust)):
        raise RuntimeError(
            "OOF list length mismatch: "
            f"sgy={len(all_sgy)} fb={len(all_fb)} robust={len(all_robust)}"
        )
    if len(all_sgy) == 0:
        raise RuntimeError("OOF all lists are empty")

    all_keys = [normalized_path_key(s, root) for s in all_sgy]
    dup_all = [k for k, n in Counter(all_keys).items() if n > 1]
    if dup_all:
        raise RuntimeError(f"duplicate SGY entries in OOF all list: {dup_all[:5]}")

    sgy_to_idx = {k: i for i, k in enumerate(all_keys)}

    physics_config_paths = sorted(root.glob(args.physics_config_glob))
    if not physics_config_paths:
        physics_config_paths = sorted(Path("/").glob(str(args.physics_config_glob).lstrip("/")))
    if not physics_config_paths:
        raise RuntimeError(f"no physics configs found for glob: {args.physics_config_glob}")

    base_cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(base_cfg, dict):
        raise RuntimeError(f"base config is not a YAML mapping: {base_config}")

    generated_configs: list[Path] = []
    heldout_all_keys: list[str] = []

    for phys_cfg_path in physics_config_paths:
        fold = extract_fold_name(phys_cfg_path)
        phys_cfg = yaml.safe_load(phys_cfg_path.read_text(encoding="utf-8"))
        held_sgy = read_list_or_values(phys_cfg["paths"]["segy_files"], root)
        held_keys = [normalized_path_key(s, root) for s in held_sgy]

        unknown = sorted(set(held_keys) - set(sgy_to_idx))
        if unknown:
            raise RuntimeError(
                f"{fold}: heldout SGY not found in OOF all list. "
                f"First unknown entries: {unknown[:5]}"
            )

        held_idx = [sgy_to_idx[k] for k in held_keys]
        held_idx_set = set(held_idx)
        if len(held_idx) != len(held_idx_set):
            raise RuntimeError(f"{fold}: duplicate heldout SGY entries in physics config")

        train_idx = [i for i in range(len(all_sgy)) if i not in held_idx_set]

        fold_list_dir = out_list_dir / fold
        paths = {
            "train_sgy": fold_list_dir / "train_sgy.txt",
            "train_fb": fold_list_dir / "train_fb.txt",
            "train_robust": fold_list_dir / "train_robust.txt",
            "heldout_sgy": fold_list_dir / "heldout_sgy.txt",
            "heldout_fb": fold_list_dir / "heldout_fb.txt",
            "heldout_robust": fold_list_dir / "heldout_robust.txt",
        }

        train_sgy = [all_sgy[i] for i in train_idx]
        train_fb = [all_fb[i] for i in train_idx]
        train_robust = [all_robust[i] for i in train_idx]
        heldout_sgy = [all_sgy[i] for i in held_idx]
        heldout_fb = [all_fb[i] for i in held_idx]
        heldout_robust = [all_robust[i] for i in held_idx]

        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("paths", {})
        cfg["paths"]["segy_files"] = str(paths["train_sgy"])
        cfg["paths"]["fb_files"] = str(paths["train_fb"])
        cfg["paths"]["robust_npz_files"] = str(paths["train_robust"])
        cfg["paths"]["infer_segy_files"] = str(paths["heldout_sgy"])
        cfg["paths"]["infer_fb_files"] = str(paths["heldout_fb"])
        cfg["paths"]["infer_robust_npz_files"] = str(paths["heldout_robust"])
        cfg["paths"]["out_dir"] = str(root / f"{args.out_dir_prefix}_{fold}_out")

        cfg["window_center"] = {
            "npz_key": "physical_center_i",
            "fallback_npz_key": None,
        }

        out_cfg = out_config_dir / f"config_train_fbpick_fine_oof_{fold}.yaml"

        print(f"{fold}: train={len(train_idx)} heldout={len(held_idx)} -> {out_cfg}")

        if not args.dry_run:
            write_plain_list(paths["train_sgy"], train_sgy)
            write_plain_list(paths["train_fb"], train_fb)
            write_plain_list(paths["train_robust"], train_robust)
            write_plain_list(paths["heldout_sgy"], heldout_sgy)
            write_plain_list(paths["heldout_fb"], heldout_fb)
            write_plain_list(paths["heldout_robust"], heldout_robust)
            out_config_dir.mkdir(parents=True, exist_ok=True)
            out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        generated_configs.append(out_cfg)
        heldout_all_keys.extend(held_keys)

    held_counter = Counter(heldout_all_keys)
    missing = sorted(set(all_keys) - set(heldout_all_keys))
    duplicated = sorted(k for k, n in held_counter.items() if n > 1)

    print()
    print("coverage summary")
    print(f"  n_all:          {len(all_sgy)}")
    print(f"  n_held_total:   {len(heldout_all_keys)}")
    print(f"  n_held_unique:  {len(set(heldout_all_keys))}")
    print(f"  missing:        {len(missing)}")
    print(f"  duplicated:     {len(duplicated)}")

    if missing:
        print("  first missing:")
        for item in missing[:10]:
            print(f"    {item}")
    if duplicated:
        print("  first duplicated:")
        for item in duplicated[:10]:
            print(f"    {item} count={held_counter[item]}")

    if args.strict_heldout_once and (missing or duplicated or len(heldout_all_keys) != len(all_sgy)):
        raise RuntimeError("heldout coverage is not one-to-one over the OOF all list")

    print()
    if args.dry_run:
        print("dry-run only; no files written")
    else:
        print("wrote fine fold configs:")
        for p in generated_configs:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
