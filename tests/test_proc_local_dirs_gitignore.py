from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_proc_local_work_dirs_are_ignored_by_git() -> None:
    ignore_lines = (REPO_ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()
    assert '/proc/arakawa/' in ignore_lines
    assert '/proc/jogsarar/' in ignore_lines

    try:
        repo_result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip('git is not available')

    if repo_result.returncode != 0:
        pytest.skip('git ignore rules are only testable inside a git checkout')

    for path in ('proc/arakawa/local-only.txt', 'proc/jogsarar/local-only.txt'):
        result = subprocess.run(
            ['git', 'check-ignore', '--quiet', '--', path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, path
