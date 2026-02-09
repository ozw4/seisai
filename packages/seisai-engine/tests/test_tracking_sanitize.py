from __future__ import annotations

import pytest

from seisai_engine.tracking.sanitize import sanitize_key


def test_sanitize_key_replaces_invalid_chars() -> None:
    raw = 'a b$c/.-_Z9'
    assert sanitize_key(raw) == 'a_b_c/.-_Z9'


def test_sanitize_key_empty_raises() -> None:
    with pytest.raises(ValueError):
        sanitize_key('')


def test_sanitize_key_invalid_raises() -> None:
    with pytest.raises(ValueError):
        sanitize_key('---')
