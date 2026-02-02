# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from seisai_dataset.builder.builder import (
	BuildPlan,
	FBGaussMap,
	IdentitySignal,
	MakeOffsetChannel,
	MakeTimeChannel,
	MaskedSignal,
	SelectStack,
)
from seisai_dataset.config import FirstBreakGateConfig
from seisai_dataset.gate_fblc import FirstBreakGate
from seisai_dataset.segy_gather_pipeline_dataset import SegyGatherPipelineDataset

# ▼ パッケージからのインポート(必要に応じてモジュールパスを調整)
from seisai_transforms.augment import (
	PerTraceStandardize,
	ViewCompose,
)
from seisai_transforms.masking import MaskGenerator
from seisai_utils.viz import imshow_hw, imshow_overlay_hw

transform = ViewCompose(
	[
		PerTraceStandardize(eps=1e-8),
	]
)
fbgate_cfg = FirstBreakGateConfig(
	percentile=95.0,
	thresh_ms=8.0,
	min_pairs=16,
	apply_on='super_only',
	min_pick_ratio=0.9,  # ピック数率の下限を課したい場合は 0.1 などに
	verbose=True,
)
fbgate = FirstBreakGate(fbgate_cfg)
gen = MaskGenerator.traces(ratio=0.35, width=4, mode='replace', noise_std=1.0)
plan = BuildPlan(
	wave_ops=[
		IdentitySignal(src='x_view', dst='x_orig', copy=True),  # 元を退避
		MaskedSignal(generator=gen, src='x_view', dst='x_masked'),  # 破壊
	],
	label_ops=[],
	input_stack=SelectStack(
		keys=['x_masked'], dst='input', dtype=np.float32, to_torch=True
	),
	target_stack=SelectStack(
		keys=['x_orig'], dst='target', dtype=np.float32, to_torch=True
	),
)

segy_files = ['/home/dcuser/data/ActiveSeisField/TSTKRES/shotgath.sgy']
fb_files = [
	'/home/dcuser/data/ActiveSeisField/TSTKRES/fb_time_all_1341ch.crd.0613.ReMerge.npy'
]

ds = SegyGatherPipelineDataset(
	segy_files=segy_files,
	fb_files=fb_files,
	transform=transform,  # ← ViewCompose を渡す
	fbgate=fbgate,  # ← FirstBreakGate を渡す
	plan=plan,  # ← MAE(入力=出力)用
	primary_keys=('ffid,'),
	subset_traces=128,
	valid=True,
	verbose=True,
	max_trials=1024,  # 無限ループ保護(前提の実装に対応)
)

s = ds[0]
x_in = s['input'][0].cpu().numpy()  # (H,W)d
target = s['target'][0].cpu().numpy()  # (H,W)
H, W = x_in.shape

# ---- 6) 可視化(2サブプロット、タイトル=タスク名)
TASK_NAME = 'MAE: Input=x_masked / Target=x_masked'
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
fig.suptitle(TASK_NAME)

axes[0].imshow(x_in.T, aspect='auto', vmin=-3.0, vmax=+3.0, cmap='seismic')
axes[0].set_title('Input (x_masked)')
axes[0].set_xlabel('Trace (H)')
axes[0].set_ylabel('Time (samples)')

axes[1].imshow(target.T, aspect='auto', vmin=-3.0, vmax=+3.0, cmap='seismic')
axes[1].set_title('Target (x_orig)')
axes[1].set_xlabel('Trace (H)')
axes[1].set_ylabel('Time (samples)')

plt.show()

###
plan_fb = BuildPlan(
	wave_ops=[
		MakeTimeChannel(dst='time_ch'),  # (H,W) 時刻チャネル(秒)
		MakeOffsetChannel(
			dst='offset_ch', normalize=True
		),  # (H,W) オフセット(z-score)。生値にしたい場合は normalize=False
	],
	label_ops=[
		FBGaussMap(dst='fb_map', sigma=10),  # (H,W) 各行の面積=1のガウスマップ
	],
	input_stack=SelectStack(
		keys=['x_view', 'offset_ch', 'time_ch'],  # ← 3チャンネル入力
		dst='input',
		dtype=np.float32,
		to_torch=True,
	),
	target_stack=SelectStack(
		keys=['fb_map'], dst='target', dtype=np.float32, to_torch=True
	),
)

ds.plan = plan_fb  # プラン差し替え
ds.output_builder.plan = plan_fb  # 出力ビルダーのプランも差し替え
s = ds[0]
x_in = s['input'][0].cpu().numpy()  # (H,W)
oft = s['input'][1].cpu().numpy()  # (H,W)
time = s['input'][2].cpu().numpy()  # (H,W)

target = s['target'][0].cpu().numpy()  # (H,W)
H, W = x_in.shape

# ---- 6) 可視化(2サブプロット、タイトル=タスク名)
TASK_NAME = 'First-Break: Input=x_view / Target=FBGauss'
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
fig.suptitle(TASK_NAME)

imshow_hw(
	axes[0][0],
	x_in,
	title='Input (x_view)',
	cmap='seismic',
	vmin=-3.0,
	vmax=+3.0,
	transpose_for_trace_time=True,
)
imshow_hw(
	axes[0][1],
	oft,
	title='Input (Offset)',
	cmap='jet',
	transpose_for_trace_time=True,
)
imshow_hw(
	axes[1][0],
	time,
	title='Input (time)',
	cmap='jet',
	transpose_for_trace_time=True,
)
imshow_overlay_hw(
	axes[1][1],
	x_in,
	target,
	transpose_for_trace_time=True,
	base_title='Target (FB Gaussian Map)',
	base_cmap='seismic',
	base_vmin=-3.0,
	base_vmax=+3.0,
	overlay_cmap='jet',
	overlay_vmin=0.0,
	overlay_vmax=0.05,
	overlay_alpha=0.5,
)

plt.show()
