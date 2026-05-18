#!/usr/bin/env python3
"""Unified entry point for run-scoped site54 OOF CV reruns."""
# ruff: noqa: C901,D101,D103,PLR0911,S603

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

FOLDS = [f"fold{i:02d}" for i in range(6)]
DEFAULT_CV_ROOT = Path("/workspace/proc/fbpick/site54/oof")
DEFAULT_RUN_ID = "baseline_physical_center"
HIGH_LEVEL_STAGES = (
    "prepare_configs",
    "coarse_train",
    "coarse_infer",
    "physics",
    "collect",
    "fine_configs",
    "fine_train",
    "fine_infer",
    "eval",
    "check",
)
MANIFEST_STAGES = (
    "01_coarse_train",
    "02_coarse_infer",
    "03_physics",
    "04_physics_qc",
    "05_collect_oof_lists",
    "06_fine_train",
    "07_fine_infer",
    "08_eval",
)
MANIFEST_STAGE_BY_RUNNER_STAGE = {
    "coarse_train": ("01_coarse_train",),
    "coarse_infer": ("02_coarse_infer",),
    "physics": ("03_physics", "04_physics_qc"),
    "collect": ("05_collect_oof_lists",),
    "fine_train": ("06_fine_train",),
    "fine_infer": ("07_fine_infer",),
    "eval": ("08_eval",),
}
ALL_FOLD_ONLY_STAGES = {"collect", "fine_configs", "eval", "check"}


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    cv_root: Path
    run_id: str
    run_root: Path
    config_root: Path
    fold_list_root: Path
    collect_dir: Path
    fine_list_root: Path
    eval_dir: Path
    log_root: Path
    scripts_dir: Path


@dataclass(frozen=True)
class StageCommand:
    argv: list[str | Path]
    env: dict[str, str]


def command(argv: list[str | Path], env: dict[str, str] | None = None) -> StageCommand:
    return StageCommand(argv=argv, env=env or {})


def shell_join(args: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def format_command(argv: list[str | Path], env: dict[str, str]) -> str:
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    command = shell_join(argv)
    return f"{env_prefix} {command}" if env_prefix else command


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def selected_stages(args: argparse.Namespace) -> list[str]:
    stages = list(HIGH_LEVEL_STAGES) if args.stage == "all" else [args.stage]

    if args.from_stage or args.to_stage:
        start = HIGH_LEVEL_STAGES.index(args.from_stage or stages[0])
        end = HIGH_LEVEL_STAGES.index(args.to_stage or stages[-1])
        if start > end:
            raise SystemExit(
                f"--from-stage must not come after --to-stage: "
                f"{args.from_stage} > {args.to_stage}"
            )
        stages = list(HIGH_LEVEL_STAGES[start : end + 1])

    return stages


def selected_folds(raw_fold: str) -> list[str]:
    if raw_fold == "all":
        return list(FOLDS)
    if raw_fold not in FOLDS:
        raise SystemExit(f"--fold must be all or one of: {', '.join(FOLDS)}")
    return [raw_fold]


def build_paths(args: argparse.Namespace) -> RunPaths:
    repo_root = args.repo_root.resolve()
    cv_root = args.cv_root.resolve()
    run_root = (args.run_root or cv_root / "runs" / args.run_id).resolve()
    return RunPaths(
        repo_root=repo_root,
        cv_root=cv_root,
        run_id=args.run_id,
        run_root=run_root,
        config_root=(args.config_root or run_root / "configs").resolve(),
        fold_list_root=(args.fold_list_root or cv_root / "fold_lists").resolve(),
        collect_dir=(
            args.collect_dir
            or run_root / "aggregate" / "05_collect_oof_lists"
        ).resolve(),
        fine_list_root=(
            args.fine_list_root
            or run_root / "aggregate" / "05_collect_oof_lists" / "fine_fold_lists"
        ).resolve(),
        eval_dir=(args.eval_dir or run_root / "aggregate" / "08_eval").resolve(),
        log_root=(args.log_root or run_root / "logs").resolve(),
        scripts_dir=Path(__file__).resolve().parent,
    )


def initial_manifest(paths: RunPaths, folds: list[str]) -> dict:
    return {
        "run_id": paths.run_id,
        "cv_root": str(paths.cv_root),
        "run_root": str(paths.run_root),
        "fold_list_root": str(paths.fold_list_root),
        "config_root": str(paths.config_root),
        "collect_dir": str(paths.collect_dir),
        "fine_list_root": str(paths.fine_list_root),
        "eval_dir": str(paths.eval_dir),
        "log_root": str(paths.log_root),
        "folds": folds,
        "stages": dict.fromkeys(MANIFEST_STAGES, "pending"),
        "stage_timestamps": {},
    }


def load_or_create_manifest(paths: RunPaths, folds: list[str]) -> dict:
    manifest_path = paths.run_root / "manifest.yaml"
    if not manifest_path.is_file():
        return initial_manifest(paths, folds)
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return initial_manifest(paths, folds)
    payload = initial_manifest(paths, folds)
    payload.update(loaded)
    stages = payload.setdefault("stages", {})
    for stage in MANIFEST_STAGES:
        stages.setdefault(stage, "pending")
    payload.setdefault("stage_timestamps", {})
    return payload


def write_manifest(paths: RunPaths, manifest: dict) -> None:
    paths.run_root.mkdir(parents=True, exist_ok=True)
    (paths.run_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def update_manifest_stage(
    *,
    paths: RunPaths,
    manifest: dict,
    stage: str,
    status: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    manifest.setdefault("stages", {})[stage] = status
    manifest.setdefault("stage_timestamps", {}).setdefault(stage, {})[
        f"{status}_at"
    ] = now_utc()
    write_manifest(paths, manifest)


def common_env(paths: RunPaths) -> dict[str, str]:
    return {
        "OOF_ROOT": str(paths.cv_root),
        "RUN_ID": paths.run_id,
        "RUN_ROOT": str(paths.run_root),
        "CONFIG_ROOT": str(paths.config_root),
        "FOLD_LIST_ROOT": str(paths.fold_list_root),
        "COLLECT_DIR": str(paths.collect_dir),
        "FINE_LIST_ROOT": str(paths.fine_list_root),
        "EVAL_DIR": str(paths.eval_dir),
        "LOG_ROOT": str(paths.log_root),
        "REPO_ROOT": str(paths.repo_root),
    }


def run_command(stage_command: StageCommand, *, paths: RunPaths, dry_run: bool) -> None:
    env = common_env(paths)
    env.update(stage_command.env)
    print(format_command(stage_command.argv, env), flush=True)
    if dry_run:
        return
    run_env = os.environ.copy()
    run_env.update(env)
    subprocess.run(
        [str(arg) for arg in stage_command.argv],
        cwd=paths.repo_root,
        env=run_env,
        check=True,
    )


def python_script(paths: RunPaths, script_name: str) -> list[str | Path]:
    return [sys.executable, paths.scripts_dir / script_name]


def prepare_config_commands(paths: RunPaths) -> list[StageCommand]:
    return [
        command(
            [
                *python_script(paths, "check_fold_lists.py"),
                "--fold-list-root",
                paths.fold_list_root,
            ]
        ),
        command(
            [
                *python_script(paths, "make_coarse_fold_configs.py"),
                "--cv-root",
                paths.cv_root,
                "--run-id",
                paths.run_id,
                "--run-root",
                paths.run_root,
                "--fold-list-root",
                paths.fold_list_root,
                "--config-root",
                paths.config_root,
                "--legacy-flat-configs",
                "false",
            ]
        ),
        command(
            [
                *python_script(paths, "make_physics_fold_configs.py"),
                "--cv-root",
                paths.cv_root,
                "--run-id",
                paths.run_id,
                "--run-root",
                paths.run_root,
                "--fold-list-root",
                paths.fold_list_root,
                "--config-root",
                paths.config_root,
                "--legacy-flat-configs",
                "false",
                "--overwrite",
            ]
        ),
    ]


def commands_for_stage(
    *,
    stage: str,
    paths: RunPaths,
    folds: list[str],
    gpu: str,
    smoke: bool,
) -> list[StageCommand]:
    train_mode = "smoke" if smoke else "full"
    if stage == "prepare_configs":
        return prepare_config_commands(paths)
    if stage == "coarse_train":
        return [
            command(
                [
                    "bash",
                    paths.scripts_dir / "run_coarse_train_fold.sh",
                    fold,
                    gpu,
                    train_mode,
                ],
                {
                    "CONFIG_PATH": str(
                        paths.config_root
                        / fold
                        / (
                            "01_coarse_train_smoke.yaml"
                            if smoke
                            else "01_coarse_train.yaml"
                        )
                    )
                },
            )
            for fold in folds
        ]
    if stage == "coarse_infer":
        return [
            command(
                ["bash", paths.scripts_dir / "run_coarse_infer_fold.sh", fold, gpu],
                {
                    "CONFIG_PATH": str(
                        paths.config_root / fold / "02_coarse_infer.yaml"
                    )
                },
            )
            for fold in folds
        ]
    if stage == "physics":
        return [
            command(
                ["bash", paths.scripts_dir / "run_physics_fold.sh", fold],
                {
                    "PHYSICS_CONFIG": str(
                        paths.config_root / fold / "03_physics.yaml"
                    ),
                    "QC_CONFIG": str(
                        paths.config_root / fold / "04_physics_qc.yaml"
                    ),
                },
            )
            for fold in folds
        ]
    if stage == "collect":
        return [
            command(
                [
                    *python_script(paths, "collect_oof_robust_lists.py"),
                    "--cv-root",
                    paths.cv_root,
                    "--run-id",
                    paths.run_id,
                    "--run-root",
                    paths.run_root,
                    "--fold-list-root",
                    paths.fold_list_root,
                    "--out-dir",
                    paths.collect_dir,
                ]
            )
        ]
    if stage == "fine_configs":
        return [
            command(
                [
                    *python_script(paths, "make_fine_fold_configs.py"),
                    "--repo-root",
                    paths.repo_root,
                    "--cv-root",
                    paths.cv_root,
                    "--run-id",
                    paths.run_id,
                    "--run-root",
                    paths.run_root,
                    "--fold-list-root",
                    paths.fold_list_root,
                    "--config-root",
                    paths.config_root,
                    "--fine-list-root",
                    paths.fine_list_root,
                    "--oof-list-dir",
                    paths.collect_dir,
                    "--legacy-flat-configs",
                    "false",
                ]
            )
        ]
    if stage == "fine_train":
        return [
            command(
                [
                    "bash",
                    paths.scripts_dir / "run_fine_train_fold.sh",
                    fold,
                    gpu,
                    train_mode,
                ],
                {
                    "CONFIG_PATH": str(
                        paths.config_root
                        / fold
                        / (
                            "06_fine_train_smoke.yaml"
                            if smoke
                            else "06_fine_train.yaml"
                        )
                    )
                },
            )
            for fold in folds
        ]
    if stage == "fine_infer":
        return [
            command(
                ["bash", paths.scripts_dir / "run_fine_infer_fold.sh", fold, gpu],
                {
                    "CONFIG_PATH": str(
                        paths.config_root / fold / "07_fine_infer.yaml"
                    )
                },
            )
            for fold in folds
        ]
    if stage == "eval":
        return [
            command(
                [
                    *python_script(paths, "evaluate_fine_oof.py"),
                    "--cv-root",
                    paths.cv_root,
                    "--run-id",
                    paths.run_id,
                    "--fold-list-root",
                    paths.fine_list_root,
                    "--pred-root",
                    paths.run_root,
                    "--pred-stage-subdir",
                    "07_fine_infer",
                    "--out-dir",
                    paths.eval_dir,
                ]
            )
        ]
    if stage == "check":
        cmd = [
            *python_script(paths, "check_cv_outputs.py"),
            "--cv-root",
            paths.cv_root,
            "--run-id",
            paths.run_id,
            "--run-root",
            paths.run_root,
            "--fold-list-root",
            paths.fold_list_root,
            "--fine-list-root",
            paths.fine_list_root,
            "--config-root",
            paths.config_root,
        ]
        if smoke:
            cmd.append("--smoke")
        return [command(cmd)]
    raise ValueError(stage)


def mark_runner_stage(
    *,
    stage: str,
    status: str,
    paths: RunPaths,
    manifest: dict,
    dry_run: bool,
) -> None:
    for manifest_stage in MANIFEST_STAGE_BY_RUNNER_STAGE.get(stage, ()):
        update_manifest_stage(
            paths=paths,
            manifest=manifest,
            stage=manifest_stage,
            status=status,
            dry_run=dry_run,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run site54 OOF CV stages with run-scoped paths."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--cv-root", type=Path, default=DEFAULT_CV_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--config-root", type=Path, default=None)
    parser.add_argument("--fold-list-root", type=Path, default=None)
    parser.add_argument("--collect-dir", type=Path, default=None)
    parser.add_argument("--fine-list-root", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=("all", *HIGH_LEVEL_STAGES),
        default="all",
    )
    parser.add_argument("--from-stage", choices=HIGH_LEVEL_STAGES, default=None)
    parser.add_argument("--to-stage", choices=HIGH_LEVEL_STAGES, default=None)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = build_paths(args)
    stages = selected_stages(args)
    folds = selected_folds(args.fold)

    all_fold_stages = sorted(ALL_FOLD_ONLY_STAGES.intersection(stages))
    if args.fold != "all" and all_fold_stages:
        raise SystemExit(
            "single-fold runs are not supported for stages that require all folds: "
            + ", ".join(all_fold_stages)
        )

    manifest = load_or_create_manifest(paths, folds)
    if not args.dry_run:
        manifest["created_at"] = manifest.get("created_at") or now_utc()
        write_manifest(paths, manifest)

    print(f"run_id={paths.run_id}")
    print(f"cv_root={paths.cv_root}")
    print(f"run_root={paths.run_root}")
    print(f"config_root={paths.config_root}")
    print(f"fold_list_root={paths.fold_list_root}")
    print(f"fine_list_root={paths.fine_list_root}")
    print(f"stages={','.join(stages)}")
    print(f"folds={','.join(folds)}")

    for stage in stages:
        if stage == "prepare_configs" and not args.dry_run:
            manifest["config_prepare_started_at"] = now_utc()
            write_manifest(paths, manifest)
        commands = commands_for_stage(
            stage=stage,
            paths=paths,
            folds=folds,
            gpu=args.gpu,
            smoke=args.smoke,
        )
        try:
            for command in commands:
                run_command(command, paths=paths, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            mark_runner_stage(
                stage=stage,
                status="failed",
                paths=paths,
                manifest=manifest,
                dry_run=args.dry_run,
            )
            return exc.returncode
        mark_runner_stage(
            stage=stage,
            status="completed",
            paths=paths,
            manifest=manifest,
            dry_run=args.dry_run,
        )
        if stage == "prepare_configs" and not args.dry_run:
            manifest["config_prepared_at"] = now_utc()
            write_manifest(paths, manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
