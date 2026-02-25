#!/usr/bin/env python3
"""Run stage1 -> stage2 -> stage4 pipeline (without stage3).

Examples:
  # directory input
  # python proc/jogsarar/run_stage124.py --in /data/jogsarar --stage1-ckpt /w/stage1.pth --stage4-ckpt /w/stage4.pt --out-root /data/out
  # single file input
  # python proc/jogsarar/run_stage124.py --in /data/jogsarar/a.sgy --stage1-ckpt /w/stage1.pth --stage4-ckpt /w/stage4.pt --stage4-cfg-yaml proc/jogsarar/configs/config_convnext_prestage2_drop005.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import stage1_fbp_infer_raw as stage1
import stage2_make_psn512_windows as stage2
import stage4_psn512_infer_to_raw as stage4
from jogsarar_shared import find_segy_files


def _parse_segy_exts(text: str) -> tuple[str, ...]:
    vals = [x.strip() for x in text.split(',')]
    out: list[str] = []
    for v in vals:
        if not v:
            continue
        e = v.lower()
        if not e.startswith('.'):
            e = '.' + e
        out.append(e)
    if len(out) == 0:
        msg = f'--segy-exts produced empty list: {text!r}'
        raise ValueError(msg)
    return tuple(out)


def _resolve_input_paths(
    in_path: Path,
    *,
    segy_exts: tuple[str, ...],
) -> tuple[Path, list[Path]]:
    p = Path(in_path).expanduser().resolve()
    if not p.exists():
        msg = f'--in not found: {p}'
        raise FileNotFoundError(msg)

    if p.is_file():
        if p.suffix.lower() not in segy_exts:
            msg = f'--in file extension must be one of {segy_exts}, got {p.suffix}'
            raise ValueError(msg)
        return p.parent, [p]

    if not p.is_dir():
        msg = f'--in must be file or directory: {p}'
        raise NotADirectoryError(msg)

    segys = find_segy_files(p, exts=segy_exts, recursive=True)
    if len(segys) == 0:
        msg = f'no segy files found under: {p}'
        raise RuntimeError(msg)
    return p, segys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Run stage1 -> stage2 -> stage4 pipeline without stage3.'
    )
    p.add_argument('--in', dest='in_path', type=Path, required=True)
    p.add_argument('--stage1-ckpt', type=Path, required=True)
    p.add_argument('--stage4-ckpt', type=Path, required=True)
    p.add_argument('--stage4-cfg-yaml', type=Path, default=None)
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--segy-exts', type=str, default=None)
    p.add_argument('--thresh-mode', choices=('global', 'per_segy'), default=None)
    p.add_argument('--viz-every-n-shots', type=int, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    stage1_ckpt = Path(args.stage1_ckpt).expanduser().resolve()
    if not stage1_ckpt.is_file():
        msg = f'--stage1-ckpt not found: {stage1_ckpt}'
        raise FileNotFoundError(msg)

    stage4_ckpt = Path(args.stage4_ckpt).expanduser().resolve()
    if not stage4_ckpt.is_file():
        msg = f'--stage4-ckpt not found: {stage4_ckpt}'
        raise FileNotFoundError(msg)

    stage4_cfg_yaml: Path | None = None
    if args.stage4_cfg_yaml is not None:
        stage4_cfg_yaml = Path(args.stage4_cfg_yaml).expanduser().resolve()
        if not stage4_cfg_yaml.is_file():
            msg = f'--stage4-cfg-yaml not found: {stage4_cfg_yaml}'
            raise FileNotFoundError(msg)

    if args.segy_exts is None:
        segy_exts = tuple(stage2.DEFAULT_STAGE2_CFG.segy_exts)
    else:
        segy_exts = _parse_segy_exts(args.segy_exts)

    in_root, segy_paths = _resolve_input_paths(args.in_path, segy_exts=segy_exts)

    if args.out_root is not None:
        out_root = Path(args.out_root).expanduser().resolve()
        if out_root.exists() and not out_root.is_dir():
            msg = f'--out-root must be a directory path: {out_root}'
            raise NotADirectoryError(msg)
        stage1_out = out_root / 'stage1'
        stage2_out = out_root / 'stage2_win512'
        stage4_out = out_root / 'stage4_pred'
    else:
        stage1_out = Path(stage2.DEFAULT_STAGE2_CFG.in_infer_root)
        stage2_out = Path(stage2.DEFAULT_STAGE2_CFG.out_segy_root)
        stage4_out = Path(stage4.DEFAULT_STAGE4_CFG.out_pred_root)

    print(f'[PIPELINE] input_root={in_root} files={len(segy_paths)}')
    print(f'[PIPELINE] stage1_out={stage1_out}')
    print(f'[PIPELINE] stage2_out={stage2_out}')
    print(f'[PIPELINE] stage4_out={stage4_out}')

    print(f'[PIPELINE] stage1 start target={len(segy_paths)} skip=0')
    stage1.run_stage1(
        in_segy_root=in_root,
        out_infer_root=stage1_out,
        weights_path=stage1_ckpt,
        segy_paths=segy_paths,
        segy_exts=segy_exts,
        recursive=True,
        viz_every_n_shots=stage1.VIZ_EVERY_N_SHOTS,
        viz_dirname=stage1.VIZ_DIRNAME,
    )
    print(f'[PIPELINE] stage1 done processed={len(segy_paths)} skip=0')

    stage2_cfg = replace(
        stage2.DEFAULT_STAGE2_CFG,
        in_segy_root=in_root,
        in_infer_root=stage1_out,
        out_segy_root=stage2_out,
        segy_exts=segy_exts,
    )
    if args.thresh_mode is not None:
        stage2_cfg = replace(stage2_cfg, thresh_mode=str(args.thresh_mode))

    stage2_skip = 0
    for p in segy_paths:
        infer_npz = stage2.infer_npz_path_for_segy(p, cfg=stage2_cfg)
        if not infer_npz.exists():
            stage2_skip += 1
    stage2_target = len(segy_paths) - stage2_skip

    print(f'[PIPELINE] stage2 start target={stage2_target} skip={stage2_skip}')
    stage2.run_stage2(cfg=stage2_cfg, segy_paths=segy_paths)
    print(f'[PIPELINE] stage2 done processed={stage2_target} skip={stage2_skip}')

    stage4_cfg = replace(
        stage4.DEFAULT_STAGE4_CFG,
        in_raw_segy_root=in_root,
        in_win512_segy_root=stage2_out,
        out_pred_root=stage4_out,
        segy_exts=segy_exts,
        cfg_yaml=stage4_cfg_yaml,
        ckpt_path=stage4_ckpt,
    )
    if args.viz_every_n_shots is not None:
        stage4_cfg = replace(stage4_cfg, viz_every_n_shots=int(args.viz_every_n_shots))

    win_lookup = stage4._build_win512_lookup(stage4_cfg.in_win512_segy_root, cfg=stage4_cfg)
    stage4_skip = 0
    for raw_path in segy_paths:
        rel = raw_path.relative_to(stage4_cfg.in_raw_segy_root)
        key = (rel.parent.as_posix(), rel.stem)
        win_path = win_lookup.get(key)
        if win_path is None:
            stage4_skip += 1
            continue
        if stage4._resolve_sidecar_path(win_path) is None:
            stage4_skip += 1
            continue
    stage4_target = len(segy_paths) - stage4_skip

    print(f'[PIPELINE] stage4 start target={stage4_target} skip={stage4_skip}')
    stage4.run_stage4(cfg=stage4_cfg, raw_paths=segy_paths)
    print(f'[PIPELINE] stage4 done processed={stage4_target} skip={stage4_skip}')
    print('[PIPELINE] completed stage1->stage2->stage4')


if __name__ == '__main__':
    main()
