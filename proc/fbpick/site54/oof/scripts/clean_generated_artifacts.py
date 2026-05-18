# ruff: noqa: INP001
"""Remove generated site54 OOF artifacts selected for a clean rerun."""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

LEGACY_OOF_NAMES = (
	'folds',
	'site54_oof_6fold_lists',
	'fine_fold_lists',
	'configs',
	'lists',
	'logs',
	'coarse_oof',
	'robust_oof',
	'physics_qc',
	'fine_infer',
	'fine_eval',
)
LEGACY_OOF_GLOBS = ('coarse_fold*_train_out',)
LEGACY_SITE54_GLOBS = (
	'fbpick_fine_train_oof*_out',
	'fbpick_fine_infer*_out',
)
PROTECTED_OOF_NAMES = {'README.md', 'config_templates', 'fold_lists', 'scripts'}


def _parse_args() -> argparse.Namespace:
	"""Parse CLI arguments and enforce destructive-action guardrails."""
	parser = argparse.ArgumentParser(
		description='Safely remove generated site54 OOF rerun artifacts.',
	)
	parser.add_argument(
		'--cv-root',
		type=Path,
		default=Path('/workspace/proc/fbpick/site54/oof'),
		help='Canonical OOF root. Defaults to /workspace/proc/fbpick/site54/oof.',
	)
	parser.add_argument('--run-id', help='Remove only this run under <cv-root>/runs.')
	parser.add_argument(
		'--all-runs',
		action='store_true',
		help='Remove every direct child under <cv-root>/runs.',
	)
	parser.add_argument(
		'--legacy-only',
		action='store_true',
		help='Remove only deprecated pre-runs-layout artifacts.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Print selected paths without deleting anything.',
	)
	parser.add_argument(
		'--yes',
		action='store_true',
		help='Required for deletion. Omit with --dry-run.',
	)
	args = parser.parse_args()

	if not (args.legacy_only or args.run_id or args.all_runs):
		parser.error('select at least one of --legacy-only, --run-id, or --all-runs')
	if args.run_id and args.all_runs:
		parser.error('--run-id and --all-runs are mutually exclusive')
	if args.run_id:
		_validate_run_id(parser, args.run_id)
	if not args.dry_run and not args.yes:
		parser.error('refusing to delete without --yes; use --dry-run to inspect')

	return args


def _validate_run_id(parser: argparse.ArgumentParser, run_id: str) -> None:
	"""Reject run ids that would escape the runs directory."""
	run_path = Path(run_id)
	if run_id in {'', '.', '..'} or run_path.name != run_id:
		parser.error(f'unsafe --run-id value: {run_id!r}')


def _legacy_candidates(cv_root: Path) -> list[Path]:
	"""Return allowlisted legacy artifact candidates for the OOF root."""
	site54_root = cv_root.parent
	candidates = [cv_root / name for name in LEGACY_OOF_NAMES]
	for pattern in LEGACY_OOF_GLOBS:
		candidates.extend(cv_root.glob(pattern))
	for pattern in LEGACY_SITE54_GLOBS:
		candidates.extend(site54_root.glob(pattern))
	return candidates


def _run_candidates(cv_root: Path, *, run_id: str | None, all_runs: bool) -> list[Path]:
	"""Return selected run-scoped artifact candidates."""
	runs_root = cv_root / 'runs'
	if run_id:
		return [runs_root / run_id]
	if not all_runs or not runs_root.exists():
		return []
	return list(runs_root.iterdir())


def _validate_candidate(path: Path, cv_root: Path) -> None:
	"""Ensure a candidate matches the generated-artifact allowlist."""
	if path.parent == cv_root and path.name in PROTECTED_OOF_NAMES:
		raise ValueError(f'protected OOF path selected for deletion: {path}')

	site54_root = cv_root.parent
	allowed = False
	if path.parent == cv_root:
		allowed = path.name in LEGACY_OOF_NAMES or any(
			fnmatch.fnmatch(path.name, pattern) for pattern in LEGACY_OOF_GLOBS
		)
	elif path.parent == site54_root:
		allowed = any(
			fnmatch.fnmatch(path.name, pattern) for pattern in LEGACY_SITE54_GLOBS
		)
	elif path.parent == cv_root / 'runs':
		allowed = True

	if not allowed:
		raise ValueError(f'path is outside the generated artifact allowlist: {path}')


def _selected_candidates(args: argparse.Namespace) -> list[Path]:
	"""Build and validate the unique deletion candidate list."""
	cv_root = args.cv_root.resolve()
	candidates: list[Path] = []
	if args.legacy_only:
		candidates.extend(_legacy_candidates(cv_root))
	candidates.extend(
		_run_candidates(cv_root, run_id=args.run_id, all_runs=args.all_runs),
	)

	unique = sorted({path.resolve(strict=False) for path in candidates})
	for path in unique:
		_validate_candidate(path, cv_root)
	return unique


def _path_exists(path: Path) -> bool:
	"""Return true for existing paths and broken symlinks."""
	return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> bool:
	"""Remove one path without following directory symlinks."""
	if not _path_exists(path):
		return False
	if path.is_symlink() or path.is_file():
		path.unlink()
		return True
	if path.is_dir():
		shutil.rmtree(path)
		return True
	path.unlink()
	return True


def _print_candidates(candidates: list[Path], *, dry_run: bool) -> None:
	"""Print selected candidates and whether each currently exists."""
	mode = 'dry-run' if dry_run else 'delete'
	existing = sum(_path_exists(path) for path in candidates)
	print(f'{mode}: {existing} existing generated artifact(s) selected')
	for path in candidates:
		status = 'would remove' if _path_exists(path) else 'missing'
		if not dry_run and _path_exists(path):
			status = 'remove'
		print(f'[{status}] {path}')


def main() -> int:
	"""Run the cleaner CLI."""
	args = _parse_args()
	candidates = _selected_candidates(args)
	_print_candidates(candidates, dry_run=args.dry_run)
	if args.dry_run:
		return 0

	removed = 0
	for path in candidates:
		if _remove_path(path):
			removed += 1
	print(f'removed: {removed}')
	return 0


if __name__ == '__main__':
	sys.exit(main())
