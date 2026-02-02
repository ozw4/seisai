import os
from pathlib import Path

import numpy as np
import pytest
import torch
from seisai_dataset import (
	BuildPlan,
	FirstBreakGate,
	FirstBreakGateConfig,
	SegyGatherPipelineDataset,
)
from seisai_dataset.builder.builder import IdentitySignal, MaskedSignal, SelectStack
from seisai_transforms.augment import (
	DeterministicCropOrPad,
	PerTraceStandardize,
	ViewCompose,
)
from seisai_transforms.masking import MaskGenerator

_DEFAULT_SEGY = Path(
	'/home/dcuser/data/ActiveSeisField/aso19-2/input_TRCTAB_ml_fbpick_Aso19-2_wolmo.sgy'
)
_DEFAULT_FBNP = Path('/home/dcuser/data/ActiveSeisField/aso19-2/fb_Aso19-2.npy')

SEGY = Path(os.environ.get('FBP_TEST_SEGY', str(_DEFAULT_SEGY)))
FBNP = Path(os.environ.get('FBP_TEST_FB', str(_DEFAULT_FBNP)))

pytestmark = [
	pytest.mark.integration,
	pytest.mark.skipif(
		not (SEGY.exists() and FBNP.exists()),
		reason=(
			'Integration test data not found. '
			f'Expected SEG-Y at {SEGY} and FB at {FBNP}. '
			'Optionally set FBP_TEST_SEGY / FBP_TEST_FB to override.'
		),
	),
]


def _build_ds() -> SegyGatherPipelineDataset:
	transform = ViewCompose([PerTraceStandardize(), DeterministicCropOrPad(2048)])

	fbgate = FirstBreakGate(
		FirstBreakGateConfig(
			apply_on='off',  # FBLC gate 無効(契約テストを安定化)
			min_pick_ratio=0.0,  # min_pick も無効
		)
	)

	gen = MaskGenerator.traces(ratio=0.25, width=1, mode='replace', noise_std=1.0)
	mask_op = MaskedSignal(gen, src='x_view', dst='x_masked', mask_key='mask_bool')

	plan = BuildPlan(
		wave_ops=[
			IdentitySignal(src='x_view', dst='x_orig', copy=True),
			mask_op,
		],
		label_ops=[],
		input_stack=SelectStack(keys='x_masked', dst='input'),
		target_stack=SelectStack(keys='x_orig', dst='target'),
	)

	return SegyGatherPipelineDataset(
		segy_files=[str(SEGY)],
		fb_files=[str(FBNP)],
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		use_superwindow=False,
		valid=False,
		verbose=False,
	)


def test_output_contract_v0():
	ds = _build_ds()
	try:
		out = ds[None]
	finally:
		ds.close()

	# --- required keys ---
	required = {
		'input',
		'target',
		'trace_valid',
		'fb_idx',
		'offsets',
		'dt_sec',
		'indices',
		'meta',
		'file_path',
		'key_name',
		'secondary_key',
		'primary_unique',
		'did_superwindow',
	}
	missing = required.difference(out.keys())
	assert not missing, f'missing keys: {sorted(missing)}'

	x = out['input']
	y = out['target']
	trace_valid = out['trace_valid']
	fb_idx = out['fb_idx']
	offsets = out['offsets']
	dt_sec = out['dt_sec']
	indices = out['indices']
	meta = out['meta']

	# --- input/target ---
	assert isinstance(x, torch.Tensor)
	assert x.dtype == torch.float32
	assert x.ndim == 3
	C_in, H, W = x.shape
	assert C_in >= 1

	assert isinstance(y, torch.Tensor)
	assert y.dtype == torch.float32
	assert y.ndim == 3
	assert y.shape[1] == H
	assert y.shape[2] == W

	# --- trace_valid ---
	assert isinstance(trace_valid, torch.Tensor)
	assert trace_valid.dtype == torch.bool
	assert trace_valid.ndim == 1
	assert trace_valid.shape[0] == H

	# --- fb_idx / offsets / dt_sec ---
	assert isinstance(fb_idx, torch.Tensor)
	assert fb_idx.dtype == torch.int64
	assert fb_idx.shape == (H,)

	assert isinstance(offsets, torch.Tensor)
	assert offsets.dtype == torch.float32
	assert offsets.shape == (H,)

	assert isinstance(dt_sec, torch.Tensor)
	assert dt_sec.dtype == torch.float32
	assert dt_sec.ndim == 0

	# --- indices ---
	assert isinstance(indices, np.ndarray)
	assert indices.dtype == np.int64
	assert indices.shape == (H,)

	# indices と trace_valid の整合(-1 が pad)
	tv_np = trace_valid.cpu().numpy()
	np.testing.assert_array_equal(tv_np, indices != -1)

	# --- meta (required minimal fields) ---
	assert isinstance(meta, dict)
	meta_required = {
		'time_view',
		'offsets_view',
		'fb_idx_view',
		'dt_eff_sec',
		'trace_valid',
	}
	missing_meta = meta_required.difference(meta.keys())
	assert not missing_meta, f'missing meta keys: {sorted(missing_meta)}'

	time_view = meta['time_view']
	offsets_view = meta['offsets_view']
	fb_idx_view = meta['fb_idx_view']
	dt_eff_sec = meta['dt_eff_sec']
	trace_valid_meta = meta['trace_valid']

	assert isinstance(time_view, np.ndarray)
	assert time_view.dtype == np.float32
	assert time_view.shape == (W,)

	assert isinstance(offsets_view, np.ndarray)
	assert offsets_view.dtype == np.float32
	assert offsets_view.shape == (H,)

	assert isinstance(fb_idx_view, np.ndarray)
	assert fb_idx_view.dtype == np.int64
	assert fb_idx_view.shape == (H,)
	assert np.all((fb_idx_view == -1) | (fb_idx_view > 0))

	assert isinstance(dt_eff_sec, float)

	assert isinstance(trace_valid_meta, np.ndarray)
	assert trace_valid_meta.dtype == np.bool_
	assert trace_valid_meta.shape == (H,)
	np.testing.assert_array_equal(trace_valid_meta, tv_np)

	# --- optional key: mask_bool (present in this plan) ---
	assert 'mask_bool' in out
	mask_bool = out['mask_bool']
	assert isinstance(mask_bool, np.ndarray)
	assert mask_bool.dtype == np.bool_
	assert mask_bool.shape == (H, W)

	# --- misc ---
	assert isinstance(out['file_path'], str)
	assert isinstance(out['key_name'], str)
	assert isinstance(out['secondary_key'], str)
	assert isinstance(out['primary_unique'], str)
	assert isinstance(out['did_superwindow'], bool)
