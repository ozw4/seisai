# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
	InputOnlyPlan,
	MakeOffsetChannel,
	MakeTimeChannel,
	SelectStack,
)
from seisai_dataset.infer_window_dataset import (
	InferenceGatherWindowsConfig,
	InferenceGatherWindowsDataset,
	collate_pad_w_right,
)
from seisai_engine.infer.runner import TiledWConfig, infer_batch_tiled_w
from seisai_engine.loss.fbsegKLLoss import FbSegKLLossView
from seisai_engine.train_loop import train_one_epoch
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
from torch.utils.data import DataLoader, Subset

# -----------------
# User config
# -----------------
TRAIN_SEGY_PATH = '/home/dcuser/data/ActiveSeisField/TSTKRES/shotgath.sgy'
TRAIN_FB_PATH = (
	'/home/dcuser/data/ActiveSeisField/TSTKRES/fb_time_all_1341ch.crd.0613.ReMerge.npy'
)

# いまは訓練データで推論(後で分けやすいように変数名だけ分ける)
INFER_SEGY_PATH = TRAIN_SEGY_PATH
INFER_FB_PATH = TRAIN_FB_PATH

BATCH_SIZE = 8
EPOCHS = 20
LR = 2e-4

SUBSET_TRACES = 128
TRAIN_TIME_LEN = 4096
SAMPLES_PER_EPOCH = 256  # reduces one epoch length (Dataset.__len__ is fixed)

# Inference config(pad-only + W tile)
INFER_TARGET_LEN = 6016
INFER_TILE_W = 6016
INFER_OVERLAP_W = 1024
INFER_TILES_PER_BATCH = 16
INFER_BATCH_SIZE = 1
INFER_MAX_BATCHES = 32

# Visualization
VIS_OUT_DIR = './_infer_vis'
VIS_N = 3
VIS_SIGMA = 10.0

LOSS_FN = FbSegKLLossView(tau=1.0, eps=0.0)


def criterion(pred: torch.Tensor, target: torch.Tensor, batch: dict) -> torch.Tensor:
	# train_loop passes `target` as a device tensor, but `batch['target']` is still on CPU.
	# FbSegKLLossView expects pred/target to share device and reads target from batch.
	batch_dev = dict(batch)
	batch_dev['target'] = target
	return LOSS_FN(pred, batch_dev, reduction='mean')


def fb_gauss_map_from_idx(
	fb_idx_view: np.ndarray,
	*,
	W: int,
	sigma: float = 10.0,
	trace_valid: np.ndarray | None = None,
) -> np.ndarray:
	"""fb_idx_view (H,) -> (H,W) ガウスheatmap(無効は0)"""
	fb = np.asarray(fb_idx_view, dtype=np.int64)
	H = int(fb.shape[0])

	if trace_valid is None:
		tv = np.ones(H, dtype=np.bool_)
	else:
		tv = np.asarray(trace_valid, dtype=np.bool_)
		if tv.shape[0] != H:
			raise ValueError(f'trace_valid shape mismatch: {tv.shape} vs H={H}')

	t = np.arange(W, dtype=np.float32)[None, :]  # (1,W)
	c = fb.astype(np.float32)[:, None]  # (H,1)
	m = np.exp(-0.5 * ((t - c) / float(sigma)) ** 2)  # (H,W)

	valid_fb = tv & (fb > 0) & (fb < W)
	m[~valid_fb, :] = 0.0
	return m.astype(np.float32, copy=False)


def save_infer_triptych_no_lines(
	*,
	x_bchw: torch.Tensor,
	logits_b1hw: torch.Tensor,
	metas: list[dict],
	step: int,
	out_dir: str,
	batch_index: int = 0,
	sigma: float = 10.0,
) -> None:
	"""横3枚(線なし・間引きなし):
	(1) input gather(ch0)
	(2) GT FB heatmap(FBGaussMap: fb_idx_view -> gauss)
	(3) Pred FB heatmap(FBGaussMap: pred_idx_view -> gauss)
	"""
	if x_bchw.ndim != 4:
		raise ValueError(f'x_bchw must be (B,C,H,W), got {tuple(x_bchw.shape)}')
	if logits_b1hw.ndim != 4 or int(logits_b1hw.shape[1]) != 1:
		raise ValueError(
			f'logits_b1hw must be (B,1,H,W), got {tuple(logits_b1hw.shape)}'
		)

	B, C, H, Wmax = x_bchw.shape
	B2, _, H2, Wmax2 = logits_b1hw.shape
	if B != B2 or H != H2 or Wmax != Wmax2:
		raise ValueError('x/logits shape mismatch')
	if not (0 <= batch_index < B):
		raise ValueError(f'batch_index out of range: {batch_index} for B={B}')
	if C <= 0:
		raise ValueError('C must be positive')

	meta = metas[batch_index]
	time_view = np.asarray(meta['time_view'], dtype=np.float32)
	W = int(time_view.shape[0])
	if W <= 0 or Wmax < W:
		raise ValueError(f'invalid W from meta time_view: W={W}, Wmax={Wmax}')

	# (1) gather: input ch0  (H,W)
	gather_hw = x_bchw[batch_index, 0, :, :W].detach().cpu()

	# (2) GT gauss map(既存の FBGaussMap を使う)
	gt_op = FBGaussMap(dst='gt_map', sigma=float(sigma), src='fb_idx_view')
	gt_sample = {'x_view': gather_hw, 'meta': meta}
	gt_op(gt_sample)
	gt_map = gt_sample['gt_map']  # (H,W) np.float32

	# (3) Pred gauss map(pred_idx_view を meta に足して同じ op で生成)
	logits_hw = logits_b1hw[batch_index, 0, :, :W].detach().cpu()  # (H,W)
	pred_idx = logits_hw.argmax(dim=-1).numpy().astype(np.int64)  # (H,)

	pred_op = FBGaussMap(dst='pred_map', sigma=float(sigma), src='pred_idx_view')
	pred_meta = dict(meta)
	pred_meta['pred_idx_view'] = pred_idx
	pred_sample = {'x_view': gather_hw, 'meta': pred_meta}
	pred_op(pred_sample)
	pred_map = pred_sample['pred_map']  # (H,W) np.float32

	outp = Path(out_dir)
	outp.mkdir(parents=True, exist_ok=True)

	fig, axes = plt.subplots(1, 3, figsize=(12, 10), sharey=True)

	axes[0].imshow(
		gather_hw.T,
		aspect='auto',
		cmap='seismic',
		vmin=-3,
		vmax=3,
		interpolation='none',
	)
	axes[0].set_title('Input gather (ch0)')
	axes[0].set_xlabel('time index')
	axes[0].set_ylabel('trace')

	axes[1].imshow(
		gt_map.T, aspect='auto', vmin=0, vmax=gt_map.max(), interpolation='none'
	)
	axes[1].set_title(f'GT FB heatmap (sigma={sigma})')
	axes[1].set_xlabel('time index')

	axes[2].imshow(
		pred_map.T, aspect='auto', vmin=0, vmax=gt_map.max(), interpolation='none'
	)
	axes[2].set_title('Pred FB heatmap (softmax)')
	axes[2].set_xlabel('time index')

	group_id = str(meta.get('group_id', ''))
	domain = str(meta.get('domain', ''))
	fig.suptitle(f'infer step={step}  domain={domain}  group={group_id}  H={H}  W={W}')
	fig.tight_layout()
	fig.savefig(outp / f'infer_triptych_step{step:04d}.png', dpi=150)
	plt.close(fig)


@torch.no_grad()
def run_inference_with_vis(
	model: torch.nn.Module,
	*,
	device: torch.device,
	infer_segy_files: list[str],
	infer_fb_files: list[str],
	train_plan: BuildPlan,
) -> None:
	# 推論では target を作らない(= InputOnlyPlan)
	infer_plan = (
		train_plan
		if isinstance(train_plan, InputOnlyPlan)
		else InputOnlyPlan.from_build_plan(train_plan, include_label_ops=False)
	)

	# 推論 transform は crop無し(標準化だけ)
	infer_transform = ViewCompose([PerTraceStandardize(eps=1e-8)])

	infer_cfg = InferenceGatherWindowsConfig(
		domains=('shot',),
		secondary_sort={'shot': 'chno', 'recv': 'ffid', 'cmp': 'offset'},
		win_size_traces=SUBSET_TRACES,
		stride_traces=64,
		pad_last=True,
		target_len=INFER_TARGET_LEN,
	)

	infer_ds_full = InferenceGatherWindowsDataset(
		segy_files=infer_segy_files,
		fb_files=infer_fb_files,
		transform=infer_transform,
		plan=infer_plan,
		cfg=infer_cfg,
		use_header_cache=True,
		header_cache_dir=None,
	)

	# デモ用途で先頭だけ回す(必要ならこの Subset を外してOK)
	n_take = min(len(infer_ds_full), INFER_MAX_BATCHES * INFER_BATCH_SIZE)
	infer_ds = Subset(infer_ds_full, range(n_take))

	infer_loader = DataLoader(
		infer_ds,
		batch_size=INFER_BATCH_SIZE,
		shuffle=False,
		num_workers=0,
		pin_memory=(device.type == 'cuda'),
		collate_fn=collate_pad_w_right,
	)

	tiled_cfg = TiledWConfig(
		tile_w=INFER_TILE_W,
		overlap_w=INFER_OVERLAP_W,
		tiles_per_batch=INFER_TILES_PER_BATCH,
		amp=True,
		use_tqdm=False,
	)

	model.eval()
	print('\n[infer] start')
	print(
		f'[infer] batches={len(infer_loader)} tile_w={tiled_cfg.tile_w} overlap_w={tiled_cfg.overlap_w} tiles_per_batch={tiled_cfg.tiles_per_batch}'
	)

	for step, (x_bchw, metas) in enumerate(infer_loader):
		if step >= INFER_MAX_BATCHES:
			break

		x_dev = x_bchw.to(device=device, non_blocking=(device.type == 'cuda'))
		logits_dev = infer_batch_tiled_w(model, x_dev, cfg=tiled_cfg)  # (B,1,H,W)
		logits_cpu = logits_dev.detach().cpu()

		if step < VIS_N:
			save_infer_triptych_no_lines(
				x_bchw=x_bchw.detach().cpu(),
				logits_b1hw=logits_cpu,
				metas=metas,
				step=step,
				out_dir=VIS_OUT_DIR,
				batch_index=0,
				sigma=VIS_SIGMA,
			)
			print(f'[infer] saved: {VIS_OUT_DIR}/infer_triptych_step{step:04d}.png')

	infer_ds_full.close()
	print('[infer] done\n')


def main() -> None:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# -----------------
	# Training dataset
	# -----------------
	train_transform = ViewCompose(
		[
			RandomCropOrPad(target_len=TRAIN_TIME_LEN),
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

	train_plan = BuildPlan(
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

	ds_full = SegyGatherPipelineDataset(
		segy_files=[TRAIN_SEGY_PATH],
		fb_files=[TRAIN_FB_PATH],
		transform=train_transform,
		fbgate=fbgate,
		plan=train_plan,
		subset_traces=SUBSET_TRACES,
		secondary_key_fixed=True,
		verbose=True,
		max_trials=2048,
	)

	train_ds = Subset(ds_full, range(SAMPLES_PER_EPOCH))
	train_loader = DataLoader(
		train_ds,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=0,
		pin_memory=(device.type == 'cuda'),
	)

	# -----------------
	# Model
	# -----------------
	model = EncDec2D(
		backbone='resnet18',
		in_chans=3,
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
	run_inference_with_vis(
		model,
		device=device,
		infer_segy_files=[INFER_SEGY_PATH],
		infer_fb_files=[INFER_FB_PATH],
		train_plan=train_plan,
	)

	ds_full.close()


if __name__ == '__main__':
	main()
