import pytest
from seisai_dataset.sample_flow import SampleFlow
from seisai_dataset.transform_contract import validate_transform_rng_meta
from seisai_transforms.augment import ViewCompose


class _DummyPlan:
	def run(self, _sample, *, rng):
		return None


def test_validate_transform_rng_meta_accepts_viewcompose():
	transform = ViewCompose([])
	validated = validate_transform_rng_meta(transform, name='transform')
	assert validated is transform


def test_validate_transform_rng_meta_rejects_missing_rng():
	def bad_transform(x):
		return x

	with pytest.raises(TypeError):
		validate_transform_rng_meta(bad_transform, name='transform')


def test_sample_flow_rejects_bad_transform():
	def bad_transform(x):
		return x

	with pytest.raises(TypeError):
		SampleFlow(bad_transform, _DummyPlan())
