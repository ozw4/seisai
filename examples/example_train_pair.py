# %%
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_engine.train_loop import train_one_epoch
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
from torch.utils.data import DataLoader, Subset

# -----------------
# User config
# -----------------
# noisy / clean のペア（同じ枚数・同じ tracecount/nsamples/dt が必須）
INPUT_SEGY_FILES = [
	'/path/noisy_001.sgy',
	'/path/noisy_002.sgy',
]
TARGET_SEGY_FILES = [
	'/path/clean_001.sgy',
	'/path/clean_002.sgy',
]

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

SUBSET_TRACES = 128
TIME_LEN = 4096
SAMPLES_PER_EPOCH = 256  # 1epochの長さを短くしたい時に使う

LOSS_KIND = 'l1'  # 'l1' or 'mse'


def criterion(pred: torch.Tensor, target: torch.Tensor, batch: dict) -> torch.Tensor:
	if LOSS_KIND == 'l1':
		return F.l1_loss(pred, target)
	if LOSS_KIND == 'mse':
		return F.mse_loss(pred, target)
	raise ValueError(f'unknown LOSS_KIND: {LOSS_KIND}')


def main() -> None:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	transform = ViewCompose(
		[
			RandomCropOrPad(target_len=TIME_LEN),
			PerTraceStandardize(eps=1e-8),
		]
	)

	# PairDataset は plan に x_view_input / x_view_target が渡ってくるよ
	plan = BuildPlan(
		wave_ops=[
			IdentitySignal(source_key='x_view_input', dst='x_in', copy=False),
			IdentitySignal(source_key='x_view_target', dst='x_tg', copy=False),
		],
		label_ops=[],
		input_stack=SelectStack(
			keys=['x_in'],
			dst='input',
			dtype=np.float32,
			to_torch=True,
		),
		target_stack=SelectStack(
			keys=['x_tg'],
			dst='target',
			dtype=np.float32,
			to_torch=True,
		),
	)

	ds = SegyGatherPairDataset(
		input_segy_files=INPUT_SEGY_FILES,
		target_segy_files=TARGET_SEGY_FILES,
		transform=transform,
		plan=plan,
		subset_traces=SUBSET_TRACES,
		valid=False,
		verbose=True,
		max_trials=2048,
		use_header_cache=True,
	)

	train_ds = Subset(ds, range(SAMPLES_PER_EPOCH))
	loader = DataLoader(
		train_ds,
		batch_size=BATCH_SIZE,
		shuffle=False,  # dataset自体がランダムサンプルなので不要
		num_workers=0,
		pin_memory=(device.type == 'cuda'),
	)

	model = EncDec2D(
		backbone='resnet18',
		in_chans=1,
		out_chans=1,
		pretrained=False,
	)
	model.use_tta = False
	model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

	for epoch in range(EPOCHS):
		stats = train_one_epoch(
			model,
			loader,
			optimizer,
			criterion,
			device=device,
			lr_scheduler=None,
			gradient_accumulation_steps=1,
			max_norm=1.0,
			use_amp=True,
			scaler=None,
			ema=None,
			step_offset=0,
			print_freq=10,
			on_step=None,
		)
		print(
			f'epoch={epoch} loss={stats["loss"]:.6f} steps={int(stats["steps"])} samples={int(stats["samples"])}'
		)

	ds.close()


if __name__ == '__main__':
	main()
