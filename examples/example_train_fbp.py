# %%
from __future__ import annotations

import numpy as np
import torch
from seisai_dataset import (
	BuildPlan,
	FirstBreakGate,
	FirstBreakGateConfig,
	SegyGatherPipelineDataset,
)
from seisai_dataset.builder.builder import (
	FBGaussMap,
	MakeOffsetChannel,
	MakeTimeChannel,
	SelectStack,
)
from seisai_engine.loss.fbsegKLLoss import FbSegKLLossView
from seisai_engine.train_loop import train_one_epoch
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
from torch.utils.data import DataLoader, Subset

# -----------------
# User config
# -----------------
SEGY_PATH = '/home/dcuser/data/ActiveSeisField/TSTKRES/shotgath.sgy'
FB_PATH = (
	'/home/dcuser/data/ActiveSeisField/TSTKRES/fb_time_all_1341ch.crd.0613.ReMerge.npy'
)

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

SUBSET_TRACES = 128
TIME_LEN = 4096
SAMPLES_PER_EPOCH = 256  # reduces one epoch length (Dataset.__len__ is fixed)

LOSS_FN = FbSegKLLossView(tau=1.0, eps=0.0)


def criterion(pred: torch.Tensor, target: torch.Tensor, batch: dict) -> torch.Tensor:
	# train_loop passes `target` as a device tensor, but `batch['target']` is still on CPU.
	# FbSegKLLossView expects pred/target to share device and reads target from batch.
	batch_dev = dict(batch)
	batch_dev['target'] = target
	return LOSS_FN(pred, batch_dev, reduction='mean')


def main() -> None:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	transform = ViewCompose(
		[
			RandomCropOrPad(target_len=TIME_LEN),
			PerTraceStandardize(eps=1e-8),
		]
	)

	# Start simple: disable gating, only keep fb_map supervision.
	fbgate = FirstBreakGate(
		FirstBreakGateConfig(
			percentile=95.0,
			thresh_ms=8.0,
			min_pairs=16,
			apply_on='off',
			min_pick_ratio=0.4,
			verbose=False,
		)
	)

	plan = BuildPlan(
		wave_ops=[
			MakeTimeChannel(dst='time_ch'),
			MakeOffsetChannel(dst='offset_ch', normalize=True),
		],
		label_ops=[
			FBGaussMap(dst='fb_map', sigma=10.0),
		],
		input_stack=SelectStack(
			keys=['x_view', 'offset_ch', 'time_ch'],
			dst='input',
			dtype=np.float32,
			to_torch=True,
		),
		target_stack=SelectStack(
			keys=['fb_map'],
			dst='target',
			dtype=np.float32,
			to_torch=True,
		),
	)

	ds = SegyGatherPipelineDataset(
		segy_files=[SEGY_PATH],
		fb_files=[FB_PATH],
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=SUBSET_TRACES,
		valid=True,
		verbose=True,
		max_trials=2048,
	)

	train_ds = Subset(ds, range(SAMPLES_PER_EPOCH))
	loader = DataLoader(
		train_ds,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=0,
		pin_memory=(device.type == 'cuda'),
	)

	model = EncDec2D(
		backbone='resnet18',
		in_chans=3,
		out_chans=1,
		pretrained=False,
	)
	model.use_tta = False  # 例ではTTA不要ならOFF
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
