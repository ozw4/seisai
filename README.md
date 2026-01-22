# seisai

Modular, PyTorch-friendly **SEG-Y gather dataset toolkit** (monorepo).

> Note: 現状は monorepo 内のサブパッケージを editable install して使う前提です（`import seisai` はまだ用意していません）。
> Dataset の入口は **`seisai_dataset`** です。

## Install (local dev / monorepo)

このリポジトリ直下で、依存順に editable install してください。

```bash
# core utils
pip install -e packages/seisai-utils

# pick utilities (required by dataset builders)
pip install -e packages/seisai-pick

# transforms (currently required; dataset imports view_projection)
pip install -e packages/seisai-transforms

# dataset
pip install -e packages/seisai-dataset
```

## Quick Start
```python
from torch.utils.data import DataLoader

from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    SegyGatherPipelineDataset,
)
from seisai_dataset.builder.builder import (
    FBGaussMap,
    IdentitySignal,
    MaskedSignal,
    SelectStack,
)
from seisai_transforms.augment import (
    PerTraceStandardize,
    RandomCropOrPad,
    ViewCompose,
)
from seisai_transforms.masking import MaskGenerator

# ---- 1) transform: (H,W0) -> (H,W)
transform = ViewCompose([
    PerTraceStandardize(),
    RandomCropOrPad(target_len=2048),
])

# ---- 2) gate: disable FBLC for a stable quickstart
fbgate = FirstBreakGate(
    FirstBreakGateConfig(
        apply_on="off",
        min_pick_ratio=0.0,
    )
)

# ---- 3) plan: build input/target (example: recon with masking)
mask_gen = MaskGenerator.traces(
    ratio=0.25,      # masked trace ratio (per-gather)
    width=1,
    mode="replace",  # or "add"
    noise_std=1.0,
)
mask_op = MaskedSignal(
    mask_gen,
    src="x_view",
    dst="x_masked",
    mask_key="mask_bool",
)

plan = BuildPlan(
    wave_ops=[
        IdentitySignal(src="x_view", dst="x_orig", copy=True),
        mask_op,
    ],
    label_ops=[
        # fb target example (optional):
        # FBGaussMap(dst="fb_map", sigma=1.5, src="fb_idx_view"),
    ],
    input_stack=SelectStack(keys="x_masked", dst="input"),
    target_stack=SelectStack(keys="x_orig", dst="target"),   # recon target
    # fb_seg target にしたい場合は target_stack=SelectStack("fb_map","target") にする
)

ds = SegyGatherPipelineDataset(
    segy_files=["/path/input.sgy"],
    fb_files=["/path/fb.npy"],
    transform=transform,
    fbgate=fbgate,
    plan=plan,
    use_header_cache=True,
)

loader = DataLoader(ds, batch_size=2, num_workers=2)
batch = next(iter(loader))

# batch is a dict; keys include:
#   input: (B,C,H,W), target: (B,C2,H,W), mask_bool (optional), meta, fb_idx, offsets, ...
```

## Core piecies

> SegyGatherPipelineDataset – sample → load → transform → FB gate → BuildPlan で input/target を組み立て。
> TraceSubsetSampler – gather/subset の抽出（primary key / superwindow など）。
> TraceSubsetLoader – segyio mmap load + padding。
> ViewCompose + augmenters – time/space/freq 等の独立コンポーネント（meta を統合）。
> MaskGenerator / MaskedSignal – bool mask を生成して replace/add を適用（mask_bool を出力）。
> FirstBreakGate – FBLC gate（apply_on="off" で無効化可）。
> FBGaussMap – fb_idx_view から Gaussian heatmap を生成（fb_seg 用）。

## Testing
```bash
pytest -q

# Enable integration tests with real data
export FBP_TEST_SEGY=/abs/path/sample.sgy
export FBP_TEST_FB=/abs/path/fb.npy
pytest -q
```

## License
MIT
