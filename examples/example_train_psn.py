# %%
# %%
"""Train a P/S/Noise (3-class) soft-label classifier on phase-pick CSR data.

Contract:
  - input:  (B,1,H,W) waveform only
  - target: (B,3,H,W) [P,S,Noise] per-pixel distribution (sum==1)
  - logits: (B,3,H,W) (no softmax in the model)

Config:
  - YAML is ALWAYS loaded.
  - If --config is omitted, defaults to a YAML next to this script:
      examples/config_train_psn.yaml
  - Any relative paths inside YAML are resolved relative to the YAML file location.
  - --vis_out_dir overrides vis.out_dir in YAML (useful for tests).
"""

from __future__ import annotations

import argparse
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
from seisai_utils.config import (
	load_config,
	optional_bool,
	optional_float,
	optional_int,
	optional_str,
	require_dict,
	require_float,
	require_int,
	require_list_str,
)
from seisai_utils.viz_phase import make_title_from_batch_meta, save_psn_debug_png
from torch.utils.data import DataLoader, Subset

DEFAULT_CONFIG_PATH = Path(__file__).with_name('config_train_psn.yaml')


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
		raise TypeError("batch['input']/batch['target'] must be torch.Tensor")
	if x.ndim != 4 or y.ndim != 4:
		raise ValueError(
			f'expected input/target batched tensors: input={tuple(x.shape)} target={tuple(y.shape)}'
		)

	x_dev = x.to(device=device, non_blocking=(device.type == 'cuda'))
	logits_dev = model(x_dev)

	print(
		f'[debug] epoch={epoch} input={tuple(x.shape)} target={tuple(y.shape)} logits={tuple(logits_dev.shape)}'
	)

	metrics = compute_ps_metrics_from_batch(logits_dev, batch, thresholds=(5, 10, 20))
	print(
		'[metrics] '
		+ ' '.join(
			f'{k}={v:.4f}' if np.isfinite(v) else f'{k}=nan' for k, v in metrics.items()
		)
	)

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


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
	parser.add_argument('--vis_out_dir', default=None)
	args, _unknown = parser.parse_known_args(argv)

	cfg_path = Path(args.config).expanduser().resolve()
	if not cfg_path.is_file():
		msg = f'config not found: {cfg_path}'
		raise FileNotFoundError(msg)

	cfg = load_config(str(cfg_path))
	if cfg is None:
		msg = f'config is empty or failed to load: {cfg_path}'
		raise ValueError(msg)

	base_dir = cfg_path.parent

	def _resolve_path(p: str) -> str:
		pp = Path(p).expanduser()
		if not pp.is_absolute():
			pp = base_dir / pp
		return str(pp.resolve())

	paths = require_dict(cfg, 'paths')
	ds_cfg = require_dict(cfg, 'dataset')
	train_cfg = require_dict(cfg, 'train')
	transform_cfg = require_dict(cfg, 'transform')
	vis_cfg = require_dict(cfg, 'vis')
	model_cfg = require_dict(cfg, 'model')

	segy_files = [_resolve_path(p) for p in require_list_str(paths, 'segy_files')]
	phase_pick_files = [
		_resolve_path(p) for p in require_list_str(paths, 'phase_pick_files')
	]
	if len(segy_files) != len(phase_pick_files):
		raise ValueError(
			'paths.segy_files and paths.phase_pick_files must have same length'
		)

	max_trials = optional_int(ds_cfg, 'max_trials', 2048)
	use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
	verbose = optional_bool(ds_cfg, 'verbose', default=True)
	include_empty_gathers = optional_bool(
		ds_cfg, 'include_empty_gathers', default=False
	)
	valid = optional_bool(ds_cfg, 'valid', default=False)
	primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
	if not isinstance(primary_keys_list, list) or not all(
		isinstance(x, str) for x in primary_keys_list
	):
		raise ValueError('dataset.primary_keys must be list[str]')
	primary_keys = tuple(primary_keys_list)

	batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	subset_traces = require_int(train_cfg, 'subset_traces')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	psn_sigma = optional_float(train_cfg, 'psn_sigma', 1.5)
	max_norm = optional_float(train_cfg, 'max_norm', 1.0)
	use_amp = optional_bool(train_cfg, 'use_amp', default=True)
	num_workers = optional_int(train_cfg, 'num_workers', 0)

	# transform is required by design
	target_len = require_int(transform_cfg, 'target_len')
	standardize_eps = optional_float(transform_cfg, 'standardize_eps', 1.0e-8)

	vis_out_dir = optional_str(vis_cfg, 'out_dir', './_psn_vis')
	if args.vis_out_dir is not None:
		vis_out_dir = args.vis_out_dir
	vis_out_dir = _resolve_path(vis_out_dir)

	backbone = optional_str(model_cfg, 'backbone', 'resnet18')
	pretrained = optional_bool(model_cfg, 'pretrained', default=False)
	in_chans = optional_int(model_cfg, 'in_chans', 1)
	out_chans = optional_int(model_cfg, 'out_chans', 3)

	if int(in_chans) != 1:
		msg = 'model.in_chans must be 1 (waveform only)'
		raise ValueError(msg)
	if int(out_chans) != 3:
		raise ValueError('model.out_chans must be 3 (P/S/Noise)')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	for p in list(segy_files) + list(phase_pick_files):
		if not Path(p).is_file():
			raise FileNotFoundError(f'file not found: {p}')

	train_transform = ViewCompose(
		[
			RandomCropOrPad(target_len=int(target_len)),
			PerTraceStandardize(eps=float(standardize_eps)),
		]
	)

	fbgate = FirstBreakGate(
		FirstBreakGateConfig(
			apply_on='off',
			min_pick_ratio=0.0,
		)
	)

	plan = _build_plan(sigma=float(psn_sigma))

	ds_full = SegyGatherPhasePipelineDataset(
		segy_files=segy_files,
		phase_pick_files=phase_pick_files,
		transform=train_transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(subset_traces),
		include_empty_gathers=bool(include_empty_gathers),
		use_header_cache=bool(use_header_cache),
		primary_keys=primary_keys,
		valid=bool(valid),
		verbose=bool(verbose),
		max_trials=int(max_trials),
	)

	try:
		train_ds = Subset(ds_full, range(int(samples_per_epoch)))
		train_loader = DataLoader(
			train_ds,
			batch_size=int(batch_size),
			shuffle=True,
			num_workers=int(num_workers),
			pin_memory=(device.type == 'cuda'),
		)

		model = EncDec2D(
			backbone=str(backbone),
			in_chans=int(in_chans),
			out_chans=int(out_chans),
			pretrained=bool(pretrained),
		)
		model.use_tta = False
		model.to(device)
		optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

		Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

		for epoch in range(int(epochs)):
			stats = train_one_epoch(
				model,
				train_loader,
				optimizer,
				criterion,
				device=device,
				lr_scheduler=None,
				gradient_accumulation_steps=1,
				max_norm=float(max_norm),
				use_amp=bool(use_amp),
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
				out_dir=vis_out_dir,
			)
	finally:
		ds_full.close()


if __name__ == '__main__':
	main()
