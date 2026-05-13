from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PICK_SRC = REPO_ROOT / 'packages/seisai-pick/src'


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = (
        str(PICK_SRC) if not existing else os.pathsep.join([str(PICK_SRC), existing])
    )
    return env


def test_import_io_grstat_does_not_import_numba_detector_path() -> None:
    code = r"""
import importlib
import importlib.abc
import sys


class BlockNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numba' or fullname.startswith('numba.'):
            raise AssertionError(f'unexpected numba import: {fullname}')
        return None


sys.meta_path.insert(0, BlockNumba())
mod = importlib.import_module('seisai_pick.pickio.io_grstat')
assert mod.GrstatMatrix.__name__ == 'GrstatMatrix'

unexpected = [
    name
    for name in ('numba', 'seisai_pick.detectors', 'seisai_pick.stalta')
    if name in sys.modules
]
if unexpected:
    raise AssertionError(f'unexpected eager imports: {unexpected}')
"""
    subprocess.run(
        [sys.executable, '-c', code],
        check=True,
        env=_pythonpath_env(),
        text=True,
        capture_output=True,
    )


def test_top_level_legacy_exports_resolve_when_numba_is_available() -> None:
    code = r"""
import sys

try:
    import numba  # noqa: F401
except Exception as exc:
    print(f'NUMBA_UNAVAILABLE:{type(exc).__name__}:{exc}', file=sys.stderr)
    raise SystemExit(77)

from seisai_pick import (
    detect_event_pick_cluster,
    detect_event_stalta_majority,
    stalta_1d,
)

assert callable(detect_event_pick_cluster)
assert callable(detect_event_stalta_majority)
assert callable(stalta_1d)
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        env=_pythonpath_env(),
        text=True,
        capture_output=True,
    )
    if result.returncode == 77:
        pytest.skip(result.stderr.strip())
    assert result.returncode == 0, result.stderr
