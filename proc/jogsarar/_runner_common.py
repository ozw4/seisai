from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from common.iter_layout import resolve_iter_layout
from common.stage3_iter_config import build_iter_stage3_config, write_stage3_listfiles
from config_io import (
    coerce_optional_bool as _coerce_optional_bool,
    coerce_optional_float as _coerce_optional_float,
    coerce_optional_int as _coerce_optional_int,
    coerce_path as _coerce_path,
    load_yaml_dict as _load_yaml_dict,
    normalize_segy_exts as _normalize_segy_exts,
    parse_args_with_yaml_defaults as _parse_args_with_yaml_defaults,
)
import stage1_fbp_infer_raw as stage1
import stage2_make_psn512_windows as stage2
import stage4_psn512_infer_to_raw as stage4
from jogsarar_shared import find_segy_files

REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE_NAME_BY_INT = {
    1: 'stage1',
    2: 'stage2',
    3: 'stage3',
    4: 'stage4',
}
_STAGE_NAMES = {'stage1', 'stage2', 'stage3', 'stage4'}


def load_config(config_path: Path) -> dict[str, object]:
    return _load_yaml_dict(config_path)


def parse_args_with_yaml_defaults(
    parser: argparse.ArgumentParser,
    *,
    load_yaml_defaults: Callable[[Path], dict[str, object]],
) -> argparse.Namespace:
    return _parse_args_with_yaml_defaults(parser, load_defaults=load_yaml_defaults)


def coerce_path_value(
    key: str, value: object, *, allow_none: bool = False
) -> Path | None:
    return _coerce_path(key, value, allow_none=allow_none)


def coerce_optional_int_value(key: str, value: object) -> int | None:
    return _coerce_optional_int(key, value)


def coerce_optional_bool_value(key: str, value: object) -> bool | None:
    return _coerce_optional_bool(key, value)


def coerce_optional_float_value(key: str, value: object) -> float | None:
    return _coerce_optional_float(key, value)


def normalize_segy_exts(value: object) -> tuple[str, ...]:
    return _normalize_segy_exts(value)


def collect_inputs(
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


def resolve_out_root(
    out_root: Path | None,
    *,
    iter_id: int | None,
) -> tuple[Path, Path, Path, Path | None]:
    if out_root is not None:
        iter_value = 0 if iter_id is None else int(iter_id)
        layout = resolve_iter_layout(Path(out_root), iter_id=iter_value)
        return (
            layout.stage1_out,
            layout.stage2_out,
            layout.stage4_out,
            layout.stage3_out,
        )

    return (
        Path(stage2.DEFAULT_STAGE2_CFG.in_infer_root),
        Path(stage2.DEFAULT_STAGE2_CFG.out_segy_root),
        Path(stage4.DEFAULT_STAGE4_CFG.out_pred_root),
        None,
    )


def resolve_existing_file(path: Path, *, context: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        msg = f'{context} not found: {p}'
        raise FileNotFoundError(msg)
    return p


def _normalize_stages(stages: tuple[int, ...] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in stages:
        if isinstance(raw, int):
            if raw not in _STAGE_NAME_BY_INT:
                msg = f'unknown stage id: {raw}'
                raise ValueError(msg)
            normalized.append(_STAGE_NAME_BY_INT[raw])
            continue

        if isinstance(raw, str):
            if raw not in _STAGE_NAMES:
                msg = f'unknown stage name: {raw}'
                raise ValueError(msg)
            normalized.append(raw)
            continue

        msg = f'stage value must be int or str, got {type(raw).__name__}'
        raise TypeError(msg)
    return tuple(normalized)


def _run_stage3_train(*, stage3_config: Path) -> None:
    cli_path = REPO_ROOT / 'cli' / 'run_psn_train.py'
    if not cli_path.is_file():
        msg = f'stage3 CLI not found: {cli_path}'
        raise FileNotFoundError(msg)

    cmd = [sys.executable, str(cli_path), '--config', str(stage3_config)]
    print(f'[PIPELINE] stage3 start config={stage3_config}')
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    print('[PIPELINE] stage3 done')


def run_pipeline(
    *,
    in_root: Path,
    segy_paths: list[Path],
    stage1_ckpt: Path,
    stage1_out: Path,
    stage2_out: Path,
    stage4_out: Path,
    stage3_out: Path | None = None,
    segy_exts: tuple[str, ...],
    stages: tuple[int, ...] | tuple[str, ...],
    iter_id: int | None = None,
    stage1_cfg_yaml: Path | None = None,
    mode: str | None = None,
    skip_stage4: bool = False,
    stage2_thresh_mode: str | None = None,
    stage2_emit_training_artifacts: bool = False,
    run_stage3: bool = False,
    stage3_config: Path | None = None,
    stage3_skip_message: str | None = None,
    stage4_ckpt: Path | None = None,
    stage4_cfg_yaml: Path | None = None,
    stage4_viz_every_n_shots: int | None = None,
    stage4_standardize_eps: float | None = None,
    completion_message: str | None = None,
    completion_message_no_stage4: str | None = None,
) -> None:
    stage_names = _normalize_stages(stages)
    iter_value = 0
    if stage3_out is not None:
        iter_value = 0 if iter_id is None else int(iter_id)
    stage1_source_model_id = str(stage1_ckpt.name)

    if mode is not None:
        print(f'[PIPELINE] mode={mode} skip_stage4={int(skip_stage4)}')
    print(f'[PIPELINE] input_root={in_root} files={len(segy_paths)}')
    if stage3_out is not None:
        print(f'[PIPELINE] iter_id={iter_value} iter_root={stage3_out.parent}')
    print(f'[PIPELINE] stage1_out={stage1_out}')
    print(f'[PIPELINE] stage2_out={stage2_out}')
    if ('stage4' in stage_names) and (not skip_stage4):
        print(f'[PIPELINE] stage4_out={stage4_out}')

    if 'stage1' in stage_names:
        print(f'[PIPELINE] stage1 start target={len(segy_paths)} skip=0')
        if stage1_cfg_yaml is None:
            stage1_cfg = replace(
                stage1.DEFAULT_STAGE1_CFG,
                in_segy_root=in_root,
                out_infer_root=stage1_out,
                weights_path=stage1_ckpt,
                segy_exts=segy_exts,
                recursive=True,
                viz_every_n_shots=stage1.VIZ_EVERY_N_SHOTS,
                viz_dirname=stage1.VIZ_DIRNAME,
                iter_id=iter_value,
                source_model_id=stage1_source_model_id,
            )
        else:
            print(f'[PIPELINE] stage1 cfg_yaml={stage1_cfg_yaml}')
            stage1_cfg = stage1.load_stage1_cfg_yaml(stage1_cfg_yaml)
            stage1_cfg = replace(
                stage1_cfg,
                in_segy_root=in_root,
                out_infer_root=stage1_out,
                weights_path=stage1_ckpt,
                segy_exts=segy_exts,
                recursive=True,
                iter_id=iter_value,
                source_model_id=stage1_source_model_id,
            )
        stage1.run_stage1_cfg(stage1_cfg, segy_paths=segy_paths)
        print(f'[PIPELINE] stage1 done processed={len(segy_paths)} skip=0')

    if 'stage2' in stage_names:
        stage2_cfg = replace(
            stage2.DEFAULT_STAGE2_CFG,
            in_segy_root=in_root,
            in_infer_root=stage1_out,
            out_segy_root=stage2_out,
            segy_exts=segy_exts,
            emit_training_artifacts=stage2_emit_training_artifacts,
            iter_id=iter_value,
            source_model_id=stage1_source_model_id,
        )
        if stage2_thresh_mode is not None:
            stage2_cfg = replace(stage2_cfg, thresh_mode=str(stage2_thresh_mode))

        stage2_skip = 0
        for p in segy_paths:
            infer_npz = stage2.infer_npz_path_for_segy(p, cfg=stage2_cfg)
            if not infer_npz.exists():
                stage2_skip += 1
        stage2_target = len(segy_paths) - stage2_skip

        print(f'[PIPELINE] stage2 start target={stage2_target} skip={stage2_skip}')
        stage2.run_stage2(cfg=stage2_cfg, segy_paths=segy_paths)
        print(f'[PIPELINE] stage2 done processed={stage2_target} skip={stage2_skip}')

    if 'stage3' in stage_names:
        if run_stage3:
            if stage3_config is None:
                msg = 'internal error: stage3_config must not be None in train mode'
                raise RuntimeError(msg)
            stage3_config_arg = stage3_config
            if stage3_out is not None:
                segy_list, phase_pick_list = write_stage3_listfiles(
                    stage2_out=stage2_out,
                    out_dir=stage3_out,
                )
                stage3_config_arg = build_iter_stage3_config(
                    base_config=stage3_config,
                    out_dir=stage3_out,
                    segy_list=segy_list,
                    phase_pick_list=phase_pick_list,
                    iter_id=iter_value,
                )
            _run_stage3_train(stage3_config=stage3_config_arg)
        elif stage3_skip_message is not None:
            print(stage3_skip_message)

    if 'stage4' in stage_names:
        if skip_stage4:
            print('[PIPELINE] stage4 skipped by config')
            if completion_message_no_stage4 is not None:
                print(completion_message_no_stage4)
            return

        stage4_source_model_id = ''
        if stage4_ckpt is not None:
            stage4_source_model_id = str(stage4_ckpt.name)
        elif stage4_cfg_yaml is not None:
            stage4_source_model_id = str(stage4_cfg_yaml.name)

        stage4_cfg = replace(
            stage4.DEFAULT_STAGE4_CFG,
            in_raw_segy_root=in_root,
            in_win512_segy_root=stage2_out,
            out_pred_root=stage4_out,
            segy_exts=segy_exts,
            cfg_yaml=stage4_cfg_yaml,
            ckpt_path=stage4_ckpt,
            iter_id=iter_value,
            source_model_id=stage4_source_model_id,
        )
        if stage4_viz_every_n_shots is not None:
            stage4_cfg = replace(
                stage4_cfg, viz_every_n_shots=int(stage4_viz_every_n_shots)
            )
        if stage4_cfg_yaml is None and stage4_standardize_eps is not None:
            stage4_cfg = replace(
                stage4_cfg, standardize_eps=float(stage4_standardize_eps)
            )

        win_lookup = stage4._build_win512_lookup(
            stage4_cfg.in_win512_segy_root, cfg=stage4_cfg
        )
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

    if completion_message is not None:
        print(completion_message)


def main_common(
    args: argparse.Namespace,
    *,
    stages: tuple[int, ...] | tuple[str, ...],
    stage1_ckpt: Path,
    stage1_cfg_yaml: Path | None = None,
    mode: str | None = None,
    skip_stage4: bool = False,
    stage2_thresh_mode: str | None = None,
    stage2_emit_training_artifacts: bool = False,
    run_stage3: bool = False,
    stage3_config: Path | None = None,
    stage3_skip_message: str | None = None,
    stage4_ckpt: Path | None = None,
    stage4_cfg_yaml: Path | None = None,
    stage4_standardize_eps: float | None = None,
    completion_message: str | None = None,
    completion_message_no_stage4: str | None = None,
) -> None:
    segy_value = args.segy_exts
    if segy_value is None:
        segy_exts = tuple(stage2.DEFAULT_STAGE2_CFG.segy_exts)
    elif isinstance(segy_value, tuple):
        segy_exts = tuple(segy_value)
    else:
        segy_exts = normalize_segy_exts(segy_value)

    in_root, segy_paths = collect_inputs(args.in_path, segy_exts=segy_exts)
    iter_id_arg = getattr(args, 'iter_id', None)
    stage1_out, stage2_out, stage4_out, stage3_out = resolve_out_root(
        args.out_root,
        iter_id=iter_id_arg,
    )
    stage1_cfg_yaml_arg = stage1_cfg_yaml
    if stage1_cfg_yaml_arg is None:
        stage1_cfg_yaml_arg = getattr(args, 'stage1_cfg_yaml', None)

    run_pipeline(
        in_root=in_root,
        segy_paths=segy_paths,
        stage1_ckpt=stage1_ckpt,
        stage1_cfg_yaml=stage1_cfg_yaml_arg,
        stage1_out=stage1_out,
        stage2_out=stage2_out,
        stage4_out=stage4_out,
        stage3_out=stage3_out,
        segy_exts=segy_exts,
        stages=stages,
        iter_id=iter_id_arg,
        mode=mode,
        skip_stage4=skip_stage4,
        stage2_thresh_mode=stage2_thresh_mode,
        stage2_emit_training_artifacts=stage2_emit_training_artifacts,
        run_stage3=run_stage3,
        stage3_config=stage3_config,
        stage3_skip_message=stage3_skip_message,
        stage4_ckpt=stage4_ckpt,
        stage4_cfg_yaml=stage4_cfg_yaml,
        stage4_viz_every_n_shots=args.viz_every_n_shots,
        stage4_standardize_eps=stage4_standardize_eps,
        completion_message=completion_message,
        completion_message_no_stage4=completion_message_no_stage4,
    )
