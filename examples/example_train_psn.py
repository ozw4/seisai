# %%
"""Train a P/S/Noise (3-class) soft-label classifier on phase-pick CSR data.

Contract:
  - input:  (B,1,H,W) waveform only
  - target: (B,3,H,W) [P,S,Noise] per-pixel distribution (sum==1)
  - logits: (B,3,H,W) (no softmax in the model)

This example uses:
  - SegyGatherPhasePipelineDataset + PhasePSNMap (target builder)
  - EncDec2D (in_chans=1, out_chans=3)
  - soft_label_ce_masked_mean (trace_valid & label_valid & optional mask_bool)
  - compute_ps_metrics_from_batch (P/S pick error summary)
  - save_psn_debug_png (waveform + target/pred panels)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from seisai_dataset import (
	BuildPlan,
	FirstBreakGate,
	FirstBreakGateConfig,
	SegyGatherPhasePipelineDataset,
)
from seisai_dataset.builder.builder import IdentitySignal, PhasePSNMap, SelectStack
from seisai_engine.loss.soft_label_ce import (
	build_pixel_mask_from_batch,
	soft_label_ce_masked_mean,
)
from seisai_engine.metrics.phase_pick_metrics import compute_ps_metrics_from_batch
from seisai_engine.train_loop import train_one_epoch
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
from seisai_utils.viz_phase import make_title_from_batch_meta, save_psn_debug_png
from torch.utils.data import DataLoader, Subset

# -----------------
# User config
# -----------------
TRAIN_SEGY_PATH = '/home/dcuser/data/ridgecrest_das/event/20200623002546.sgy'
TRAIN_PHASE_PICK_PATH = '/home/dcuser/data/ridgecrest_das/event/npz/20200623002546_phase_picks.npz'  # CSR npz (p_indptr/p_data/s_indptr/s_data)

BATCH_SIZE = 4
EPOCHS = 5
LR = 2e-4

SUBSET_TRACES = 128
TRAIN_TIME_LEN = 6016
SAMPLES_PER_EPOCH = 256  # Dataset.__len__ is nominal; Subset controls epoch length

PSN_SIGMA = 1.5

# Visualization (debug)
VIS_OUT_DIR = './_psn_vis'


def _build_plan(*, sigma: float) -> BuildPlan:
	return BuildPlan(
		wave_ops=[
			IdentitySignal(src='x_view', dst='x', copy=False),
		],
		label_ops=[
			PhasePSNMap(dst='psn_map', sigma=float(sigma)),
		],
		input_stack=SelectStack(keys='x', dst='input', dtype=np.float32, to_torch=True),
		target_stack=SelectStack(
			keys='psn_map', dst='target', dtype=np.float32, to_torch=True
		),
	)


def criterion(logits: torch.Tensor, target: torch.Tensor, batch: dict) -> torch.Tensor:
	pixel_mask = build_pixel_mask_from_batch(
		batch, use_trace_valid=True, use_label_valid=True, mask_bool_key='mask_bool'
	)
	return soft_label_ce_masked_mean(logits, target, pixel_mask)


@torch.no_grad()
def run_epoch_debug(
	model: torch.nn.Module,
	loader: DataLoader,
	*,
	device: torch.device,
	epoch: int,
	out_dir: str,
) -> None:
	model.eval()
	batch = next(iter(loader))

	x = batch['input']
	y = batch['target']
	if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
		msg = "batch['input']/batch['target'] must be torch.Tensor"
		raise TypeError(msg)
	if x.ndim != 4 or y.ndim != 4:
		msg = f'expected input/target batched tensors: input={tuple(x.shape)} target={tuple(y.shape)}'
		raise ValueError(msg)

	x_dev = x.to(device=device, non_blocking=(device.type == 'cuda'))
	logits_dev = model(x_dev)

	# Shape log (acceptance criteria)
	print(
		f'[debug] epoch={epoch} input={tuple(x.shape)} target={tuple(y.shape)} logits={tuple(logits_dev.shape)}'
	)

	# Metrics (keep logits on device; avoid CPU->GPU round-trip)
	metrics = compute_ps_metrics_from_batch(logits_dev, batch, thresholds=(5, 10, 20))
	print(
		'[metrics] '
		+ ' '.join(
			f'{k}={v:.4f}' if np.isfinite(v) else f'{k}=nan' for k, v in metrics.items()
		)
	)

	# Smoke-check: pixel_mask==0 should yield finite loss==0 (no NaN / no crash).
	pixel_mask = build_pixel_mask_from_batch(batch)
	pixel_mask_sum = int(pixel_mask.sum().item())
	y_dev = y.to(device=device, non_blocking=(device.type == 'cuda'))
	loss_masked = soft_label_ce_masked_mean(logits_dev, y_dev, pixel_mask)
	loss_empty = soft_label_ce_masked_mean(
		logits_dev, y_dev, torch.zeros_like(pixel_mask, dtype=torch.bool)
	)
	print(
		f'[debug] pixel_mask_sum={pixel_mask_sum} loss_masked={float(loss_masked.detach().cpu().item()):.6f} '
		f'loss_empty_mask={float(loss_empty.detach().cpu().item()):.6f}'
	)

	# Visualization runs on CPU.
	logits = logits_dev.detach().cpu()

	title = make_title_from_batch_meta(batch, b=0)
	outp = Path(out_dir)
	outp.mkdir(parents=True, exist_ok=True)
	png = outp / f'psn_debug_epoch{int(epoch):04d}.png'
	save_psn_debug_png(
		png,
		x_bchw=x,
		target_b3hw=y,
		logits_b3hw=logits,
		b=0,
		title=title,
	)
	print(f'[saved] {png}')


def main() -> None:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	for p in (TRAIN_SEGY_PATH, TRAIN_PHASE_PICK_PATH):
		if not Path(p).exists():
			raise FileNotFoundError(f'file not found: {p}')

	train_transform = ViewCompose(
		[
			RandomCropOrPad(target_len=TRAIN_TIME_LEN),
			PerTraceStandardize(eps=1e-8),
		]
	)

	# Start stable: disable FBLC/min-pick gates (phase dataset still uses P-first as legacy fb).
	fbgate = FirstBreakGate(
		FirstBreakGateConfig(
			apply_on='off',
			min_pick_ratio=0.0,
		)
	)

	plan = _build_plan(sigma=PSN_SIGMA)

	ds_full = SegyGatherPhasePipelineDataset(
		segy_files=[TRAIN_SEGY_PATH],
		phase_pick_files=[TRAIN_PHASE_PICK_PATH],
		transform=train_transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(SUBSET_TRACES),
		include_empty_gathers=False,
		use_header_cache=True,
		primary_keys=('ffid',),
		valid=False,
		verbose=True,
		max_trials=2048,
	)

	try:
		train_ds = Subset(ds_full, range(int(SAMPLES_PER_EPOCH)))
		train_loader = DataLoader(
			train_ds,
			batch_size=int(BATCH_SIZE),
			shuffle=True,
			num_workers=0,
			pin_memory=(device.type == 'cuda'),
		)

		model = EncDec2D(
			backbone='resnet18',
			in_chans=1,
			out_chans=3,
			pretrained=False,
		)
		model.use_tta = False
		model.to(device)
		optimizer = torch.optim.AdamW(model.parameters(), lr=float(LR))

		Path(VIS_OUT_DIR).mkdir(parents=True, exist_ok=True)

		for epoch in range(int(EPOCHS)):
			stats = train_one_epoch(
				model,
				train_loader,
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
			run_epoch_debug(
				model,
				train_loader,
				device=device,
				epoch=epoch,
				out_dir=VIS_OUT_DIR,
			)
	finally:
		ds_full.close()


if __name__ == '__main__':
	main()
