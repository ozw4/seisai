from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
	REPO_ROOT
	/ 'proc'
	/ 'fbpick'
	/ 'site54'
	/ 'oof'
	/ 'scripts'
	/ 'clean_generated_artifacts.py'
)


def run_clean(*args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, str(SCRIPT), *args],
		check=False,
		capture_output=True,
		text=True,
	)


def test_legacy_dry_run_does_not_select_protected_paths(tmp_path: Path) -> None:
	cv_root = tmp_path / 'proc' / 'fbpick' / 'site54' / 'oof'
	cv_root.mkdir(parents=True)
	(cv_root / 'README.md').write_text('runbook\n', encoding='utf-8')
	(cv_root / 'fold_lists').mkdir()
	(cv_root / 'scripts').mkdir()
	(cv_root / 'configs').mkdir()
	(cv_root / 'coarse_fold00_train_out').mkdir()
	(cv_root.parent / 'fbpick_fine_train_oof_fold00_out').mkdir()

	result = run_clean(
		'--cv-root',
		str(cv_root),
		'--legacy-only',
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert str(cv_root / 'configs') in result.stdout
	assert str(cv_root / 'coarse_fold00_train_out') in result.stdout
	assert str(cv_root.parent / 'fbpick_fine_train_oof_fold00_out') in result.stdout
	assert str(cv_root / 'README.md') not in result.stdout
	assert str(cv_root / 'fold_lists') not in result.stdout
	assert str(cv_root / 'scripts') not in result.stdout
	assert (cv_root / 'configs').exists()


def test_legacy_yes_removes_generated_paths_only(tmp_path: Path) -> None:
	cv_root = tmp_path / 'proc' / 'fbpick' / 'site54' / 'oof'
	cv_root.mkdir(parents=True)
	(cv_root / 'README.md').write_text('runbook\n', encoding='utf-8')
	(cv_root / 'fold_lists').mkdir()
	(cv_root / 'scripts').mkdir()
	(cv_root / 'configs').mkdir()
	(cv_root / 'coarse_oof').mkdir()
	(cv_root / 'coarse_fold00_train_out').mkdir()
	(cv_root.parent / 'fbpick_fine_infer_fold00_out').mkdir()

	result = run_clean(
		'--cv-root',
		str(cv_root),
		'--legacy-only',
		'--yes',
	)

	assert result.returncode == 0, result.stderr
	assert sorted(path.name for path in cv_root.iterdir()) == [
		'README.md',
		'fold_lists',
		'scripts',
	]
	assert not (cv_root.parent / 'fbpick_fine_infer_fold00_out').exists()


def test_run_id_removes_only_selected_run(tmp_path: Path) -> None:
	cv_root = tmp_path / 'proc' / 'fbpick' / 'site54' / 'oof'
	(cv_root / 'runs' / 'baseline').mkdir(parents=True)
	(cv_root / 'runs' / 'other').mkdir(parents=True)
	(cv_root / 'configs').mkdir()

	result = run_clean(
		'--cv-root',
		str(cv_root),
		'--run-id',
		'baseline',
		'--yes',
	)

	assert result.returncode == 0, result.stderr
	assert not (cv_root / 'runs' / 'baseline').exists()
	assert (cv_root / 'runs' / 'other').exists()
	assert (cv_root / 'configs').exists()
