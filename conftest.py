from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_issue_forge_skip_publish_for_smoke(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.nodeid == (
        'vendor/issue_forge/tests/test_codex_smoke_harness.py::'
        'test_codex_smoke_harness'
    ):
        monkeypatch.delenv('CODEX_FLOW_SKIP_PUBLISH', raising=False)
