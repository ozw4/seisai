from __future__ import annotations

import numpy as np
from seisai_dataset import BuildPlan
from seisai_dataset.builder.builder import (
	IdentitySignal,
	MakeOffsetChannel,
	MakeTimeChannel,
	MaskedSignal,
	SelectStack,
)
from seisai_transforms.masking import MaskGenerator


def build_plan(
	*,
	mask_ratio: float,
	mask_mode: str,
	noise_std: float,
	use_offset_ch: bool,
	offset_normalize: bool,
	use_time_ch: bool,
) -> BuildPlan:
	gen = MaskGenerator.traces(
		ratio=float(mask_ratio),
		width=1,
		mode=str(mask_mode),
		noise_std=float(noise_std),
	)

	wave_ops: list = [
		IdentitySignal(src='x_view', dst='y_wave', copy=True),
		MaskedSignal(
			generator=gen,
			src='x_view',
			dst='x_wave',
			mask_key='mask_bool',
			mode=str(mask_mode),
		),
	]

	if use_offset_ch:
		wave_ops.append(
			MakeOffsetChannel(dst='x_offset_ch', normalize=bool(offset_normalize))
		)
	if use_time_ch:
		wave_ops.append(MakeTimeChannel(dst='x_time_ch'))

	input_keys = ['x_wave']
	if use_offset_ch:
		input_keys.append('x_offset_ch')
	if use_time_ch:
		input_keys.append('x_time_ch')

	return BuildPlan(
		wave_ops=wave_ops,
		label_ops=[],
		input_stack=SelectStack(
			keys=input_keys,
			dst='input',
			dtype=np.float32,
			to_torch=True,
		),
		target_stack=SelectStack(
			keys=['y_wave'],
			dst='target',
			dtype=np.float32,
			to_torch=True,
		),
	)
