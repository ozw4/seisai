from __future__ import annotations

import pytest

from seisai_dataset.file_info import normalize_segy_endian


def test_normalize_segy_endian_defaults_to_big() -> None:
    assert normalize_segy_endian(None) == 'big'


def test_normalize_segy_endian_accepts_big_and_little_case_insensitive() -> None:
    assert normalize_segy_endian('big') == 'big'
    assert normalize_segy_endian('BIG') == 'big'
    assert normalize_segy_endian('little') == 'little'
    assert normalize_segy_endian(' LiTTle ') == 'little'


def test_normalize_segy_endian_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match='segy_endian must be "big" or "little"'):
        normalize_segy_endian('auto')


def test_normalize_segy_endian_rejects_non_string() -> None:
    with pytest.raises(TypeError, match='segy_endian must be str'):
        normalize_segy_endian(123)  # type: ignore[arg-type]
