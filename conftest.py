from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent
VENDOR_SMOKE_NODEID = (
    'vendor/issue_forge/tests/test_codex_smoke_harness.py::'
    'test_codex_smoke_harness'
)
VENDOR_SMOKE_HARNESS = (
    REPO_ROOT / 'vendor' / 'issue_forge' / 'tools' / 'codex' / 'smoke_harness.sh'
)
VENDOR_ISSUE_FORGE_ROOT = REPO_ROOT / 'vendor' / 'issue_forge'


def _is_vendor_smoke_harness_invocation(args: Any) -> bool:
    if not isinstance(args, (list, tuple)) or not args:
        return False
    return Path(str(args[0])) == VENDOR_SMOKE_HARNESS


@pytest.fixture(autouse=True)
def _clear_issue_forge_skip_publish_for_smoke(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.nodeid == VENDOR_SMOKE_NODEID:
        monkeypatch.delenv('CODEX_FLOW_SKIP_PUBLISH', raising=False)
        monkeypatch.delenv('CODEX_FLOW_LIGHT_ISSUE_REVIEW', raising=False)
        monkeypatch.delenv('CODEX_RUN_REASONING_EFFORT', raising=False)

        original_run = subprocess.run

        def run_with_workspace_cwd(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
            command = kwargs.get('args', args[0] if args else None)
            cwd = kwargs.get('cwd')
            if (
                _is_vendor_smoke_harness_invocation(command)
                and cwd is not None
                and Path(cwd) == VENDOR_ISSUE_FORGE_ROOT
            ):
                kwargs = {**kwargs, 'cwd': REPO_ROOT}
            return original_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, 'run', run_with_workspace_cwd)
