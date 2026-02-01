import pytest
from seisai_utils.config import (
	optional_tuple2_float,
	optional_value,
	require_list_str,
	require_value,
)


def test_require_value_missing_key_raises_value_error() -> None:
	data = {}
	with pytest.raises(ValueError, match='missing config key: port'):
		require_value(data, 'port', int)


def test_require_value_missing_key_uses_custom_message() -> None:
	data = {}
	with pytest.raises(ValueError, match='custom missing'):
		require_value(data, 'port', int, missing_message='custom missing')


def test_require_value_type_mismatch_raises_type_error() -> None:
	data = {'port': 'not-int'}
	with pytest.raises(TypeError, match=r'config\.port must be int'):
		require_value(data, 'port', int, type_message='config.port must be int')


def test_require_value_allow_none_accepts_none() -> None:
	data = {'value': None}
	assert require_value(data, 'value', str, allow_none=True) is None


def test_require_value_validator_returns_result() -> None:
	data = {'value': 4}
	result = require_value(data, 'value', int, validator=lambda _, v: v * 2)
	assert result == 8


def test_optional_value_missing_returns_default() -> None:
	data = {}
	assert optional_value(data, 'name', 'fallback', str) == 'fallback'


def test_optional_value_coerces_default_when_requested() -> None:
	data = {}
	assert (
		optional_value(
			data,
			'ratio',
			'3.5',
			float,
			coerce=float,
			coerce_default=True,
		)
		== 3.5
	)


def test_require_list_str_empty_list_raises_value_error() -> None:
	data = {'tags': []}
	with pytest.raises(ValueError, match=r'config.tags must be non-empty'):
		require_list_str(data, 'tags')


def test_require_list_str_type_mismatch_raises_type_error() -> None:
	data = {'tags': ['ok', 1]}
	with pytest.raises(TypeError, match=r'config.tags must be list\[str\]'):
		require_list_str(data, 'tags')


def test_optional_tuple2_float_len_mismatch_raises_value_error() -> None:
	data = {'range': [1.0]}
	with pytest.raises(ValueError, match=r'config.range must be \[float, float\]'):
		optional_tuple2_float(data, 'range', (0.0, 1.0))


def test_optional_tuple2_float_type_mismatch_raises_type_error() -> None:
	data = {'range': ['bad', 2.0]}
	with pytest.raises(TypeError, match=r'config.range must be \[float, float\]'):
		optional_tuple2_float(data, 'range', (0.0, 1.0))
