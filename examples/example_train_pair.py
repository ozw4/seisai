# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_engine.predict import _run_tiled
from seisai_engine.train_loop import train_one_epoch
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
from seisai_utils.viz import ImshowPanel, save_imshow_row
from torch.utils.data import DataLoader, Subset

# -----------------
# User config
# -----------------
# noisy/clean は 1対1で対応（同じ tracecount/nsamples/dt、同じ並びが前提）
INPUT_SEGY_FILES = [
	'/home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.30.00_bpf.sgy',
	'/home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.31.00_bpf.sgy',
	'/home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.32.00_bpf.sgy',
	'/home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.33.00_bpf.sgy',
	'/home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.34.00_bpf.sgy',
]
TARGET_SEGY_FILES = [
	'/home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.30.00_NR.sgy',
	'/home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.31.00_NR.sgy',
	'/home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.32.00_NR.sgy',
	'/home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.33.00_NR.sgy',
	'/home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.34.00_NR.sgy',
]

BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4

SUBSET_TRACES = 128
TIME_LEN = 6016
SAMPLES_PER_EPOCH = 256

LOSS_KIND = 'l1'  # 'l1' or 'mse'

# inference/vis
INFER_BATCH_SIZE = 1
INFER_MAX_BATCHES = 12
VIS_N = 5
VIS_OUT_DIR = './_pair_vis'


INFER_SUBSET_TRACES = 4096

# tiled inference (Trace方向タイル)
TILE_H = 128
OVERLAP_H = 64
TILES_PER_BATCH = 16

# vis style
VIS_CMAP = 'seismic'
VIS_TRANSPOSE_FOR_TRACE_TIME = True  # x=Trace, y=Time にそろえる

VIS_PER_TRACE_NORM = True
VIS_PER_TRACE_EPS = 1e-8


def per_trace_zscore_hw(a_hw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
	"""(H,W) をトレース毎（行ごと）に z-score。可視化専用。"""
	m = a_hw.mean(axis=1, keepdims=True)
	s = a_hw.std(axis=1, keepdims=True)
	return (a_hw - m) / (s + eps)


def criterion(pred: torch.Tensor, target: torch.Tensor, _batch: dict) -> torch.Tensor:
	if LOSS_KIND == 'l1':
		return F.l1_loss(pred, target)
	if LOSS_KIND == 'mse':
		return F.mse_loss(pred, target)
	raise ValueError(f'unknown LOSS_KIND: {LOSS_KIND}')


def _first(v):
	return v[0] if isinstance(v, list) and len(v) > 0 else v


def save_pair_triptych(
	*,
	x_in_bchw: torch.Tensor,
	x_tg_bchw: torch.Tensor,
	x_pr_bchw: torch.Tensor,
	meta: dict,
	step: int,
	out_dir: str,
	batch_index: int = 0,
) -> None:
	"""入力/正解/予測を横3枚で保存（seisai_utils.vizへ委譲）"""
	if x_in_bchw.ndim != 4 or x_tg_bchw.ndim != 4 or x_pr_bchw.ndim != 4:
		raise ValueError('x_in/x_tg/x_pr must be (B,C,H,W)')
	B, C, H, W = x_in_bchw.shape
	if x_tg_bchw.shape != (B, C, H, W) or x_pr_bchw.shape != (B, C, H, W):
		raise ValueError('shape mismatch among input/target/pred')
	if not (0 <= batch_index < B):
		raise ValueError(f'batch_index out of range: {batch_index} for B={B}')
	if C < 1:
		raise ValueError('C must be >= 1')

	in_hw = x_in_bchw[batch_index, 0].detach().cpu().numpy()
	tg_hw = x_tg_bchw[batch_index, 0].detach().cpu().numpy()
	pr_hw = x_pr_bchw[batch_index, 0].detach().cpu().numpy()

	if VIS_PER_TRACE_NORM:
		in_hw = per_trace_zscore_hw(in_hw, eps=VIS_PER_TRACE_EPS)
		tg_hw = per_trace_zscore_hw(tg_hw, eps=VIS_PER_TRACE_EPS)
		pr_hw = per_trace_zscore_hw(pr_hw, eps=VIS_PER_TRACE_EPS)

	# 3枚で同一スケール（あなたの調整を尊重して固定）
	vmin = -3
	vmax = 3

	fp_in = str(meta.get('file_path_input', ''))
	fp_tg = str(meta.get('file_path_target', ''))
	key = str(meta.get('key_name', ''))
	sec = str(meta.get('secondary_key', ''))

	suptitle = f'step={step}  key={key}  sec={sec}\ninput={Path(fp_in).name}  target={Path(fp_tg).name}'

	out_path = Path(out_dir) / f'pair_triptych_step{step:04d}.png'
	save_imshow_row(
		out_path,
		[
			ImshowPanel(
				title='Input (noisy)',
				data_hw=in_hw,
				cmap=VIS_CMAP,
				vmin=vmin,
				vmax=vmax,
			),
			ImshowPanel(
				title='Target (clean)',
				data_hw=tg_hw,
				cmap=VIS_CMAP,
				vmin=vmin,
				vmax=vmax,
			),
			ImshowPanel(
				title='Pred (model)',
				data_hw=pr_hw,
				cmap=VIS_CMAP,
				vmin=vmin,
				vmax=vmax,
			),
		],
		suptitle=suptitle,
		transpose_for_trace_time=VIS_TRANSPOSE_FOR_TRACE_TIME,
		figsize=(20.0, 15.0),
		dpi=300,
	)


@torch.no_grad()
def run_infer_and_vis(
	model: torch.nn.Module,
	loader: DataLoader,
	*,
	device: torch.device,
	out_dir: str,
	max_batches: int,
	vis_n: int,
) -> None:
	model.eval()

	for step, batch in enumerate(loader):
		if step >= max_batches:
			break

		x_in = batch['input'].to(device=device, non_blocking=(device.type == 'cuda'))
		x_tg = batch['target'].to(device=device, non_blocking=(device.type == 'cuda'))

		# ★Trace方向(H)だけタイル推論：tile_w は「W全域」にする
		W = int(x_in.shape[-1])
		x_pr = _run_tiled(
			model,
			x_in,
			tile=(int(TILE_H), W),
			overlap=(int(OVERLAP_H), 0),
			amp=True,
			use_tqdm=False,
			tiles_per_batch=int(TILES_PER_BATCH),
			tile_transform=None,
			post_tile_transform=None,
		)

		l1 = float(F.l1_loss(x_pr, x_tg).detach().cpu())
		print(f'[infer] step={step} l1={l1:.6f}  (H={int(x_in.shape[-2])}, W={W})')

		if step >= vis_n:
			continue

		meta = batch.get('meta', {})
		if isinstance(meta, list):
			meta0 = meta[0] if len(meta) > 0 else {}
		elif isinstance(meta, dict):
			meta0 = meta
		else:
			meta0 = {}

		meta0 = dict(meta0)
		fp_in = _first(batch.get('file_path_input'))
		fp_tg = _first(batch.get('file_path_target'))
		key = _first(batch.get('key_name'))
		sec = _first(batch.get('secondary_key'))
		if fp_in is not None:
			meta0['file_path_input'] = fp_in
		if fp_tg is not None:
			meta0['file_path_target'] = fp_tg
		if key is not None:
			meta0['key_name'] = key
		if sec is not None:
			meta0['secondary_key'] = sec

		save_pair_triptych(
			x_in_bchw=x_in.detach().cpu(),
			x_tg_bchw=x_tg.detach().cpu(),
			x_pr_bchw=x_pr.detach().cpu(),
			meta=meta0,
			step=step,
			out_dir=out_dir,
			batch_index=0,
		)
		print(f'[infer] saved: {out_dir}/pair_triptych_step{step:04d}.png')


def main() -> None:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# -----------------
	# Train dataset (crop/padあり)
	# -----------------
	train_transform = ViewCompose(
		[
			RandomCropOrPad(target_len=TIME_LEN),
			PerTraceStandardize(eps=1e-8),
		]
	)

	# -----------------
	# Infer dataset (cropなし)
	# -----------------
	infer_transform = ViewCompose(
		[
			PerTraceStandardize(eps=1e-8),
		]
	)

	# PairDataset は sample に x_view_input / x_view_target が入る前提
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

	ds_train_full = SegyGatherPairDataset(
		input_segy_files=INPUT_SEGY_FILES,
		target_segy_files=TARGET_SEGY_FILES,
		transform=train_transform,
		plan=plan,
		subset_traces=SUBSET_TRACES,
		primary_keys=('ffid,'),
		valid=False,
		verbose=True,
		max_trials=2048,
		use_header_cache=True,
	)

	train_ds = Subset(ds_train_full, range(SAMPLES_PER_EPOCH))
	train_loader = DataLoader(
		train_ds,
		batch_size=BATCH_SIZE,
		shuffle=False,
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

	# -----------------
	# Train
	# -----------------
	for epoch in range(EPOCHS):
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

	# -----------------
	# Inference + Visualization
	# -----------------
	ds_infer_full = SegyGatherPairDataset(
		input_segy_files=INPUT_SEGY_FILES,
		target_segy_files=TARGET_SEGY_FILES,
		transform=infer_transform,
		plan=plan,
		subset_traces=int(INFER_SUBSET_TRACES),
		valid=True,
		verbose=True,
		max_trials=2048,
		use_header_cache=True,
	)

	infer_ds = Subset(ds_infer_full, range(INFER_BATCH_SIZE * INFER_MAX_BATCHES))
	infer_loader = DataLoader(
		infer_ds,
		batch_size=INFER_BATCH_SIZE,
		shuffle=False,
		num_workers=0,
		pin_memory=(device.type == 'cuda'),
	)

	run_infer_and_vis(
		model,
		infer_loader,
		device=device,
		out_dir=VIS_OUT_DIR,
		max_batches=INFER_MAX_BATCHES,
		vis_n=VIS_N,
	)

	ds_train_full.close()
	ds_infer_full.close()


if __name__ == '__main__':
	main()
