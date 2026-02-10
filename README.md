# seisai

PyTorch で扱いやすい **SEG-Y gather データセット群 + 学習/推論ユーティリティ**のモノレポです。

- このリポジトリは `packages/*` に複数の Python パッケージを持ちます。
- いまのところ **`import seisai` のような統合トップレベルパッケージはありません**。
  - 入口は用途ごとに `seisai_dataset`, `seisai_transforms`, `seisai_engine`, `seisai_models`, `seisai_pick`, `seisai_utils` です。

動作確認は主に Python 3.10/3.11 (CI も 3.10/3.11) です。各パッケージの `requires-python` は `>=3.10`。

## リポジトリ構成

- `packages/` : 各機能を分割した Python パッケージ群（editable install 前提）
- `examples/` : 実行スクリプト（PSN / pair / blindtrace は YAML 駆動。FBP は単体スクリプト）
- `docs/` : 仕様・メモ・テスト補助ドキュメント
- `tests/` : ルートの e2e テスト（同梱サンプル `tests/data` を使用）

## パッケージ構成

| package | import | 役割 |
|---|---|---|
| `seisai-utils` | `seisai_utils` | YAML 設定ローダ/型チェック、可視化ヘルパなど |
| `seisai-transforms` | `seisai_transforms` | 2D 波形 view の augment、mask、view projection など |
| `seisai-pick` | `seisai_pick` | pick/検出のユーティリティ (gaussian, STA/LTA など) |
| `seisai-dataset` | `seisai_dataset` | SEG-Y gather Dataset 群 (pipeline / phase / pair / 推論 window) |
| `seisai-models` | `seisai_models` | モデル群 (例: `EncDec2D`) + timm backbone |
| `seisai-engine` | `seisai_engine` | train loop / loss / metrics / tiled inference / 可視化 / tracking など |

## Install (local dev / monorepo)

このリポジトリ直下で **editable install** してください。

> 依存関係: utils → transforms → pick → dataset → models → engine

```bash
python -m pip install -U pip

# まとめて editable install
for p in seisai-utils seisai-transforms seisai-pick seisai-dataset seisai-models seisai-engine; do
  python -m pip install -e "packages/$p"
done
```

メモ:
- `seisai-dataset` は `segyio` を使います（環境によってはビルド/導入に追加要件が出る場合があります）。
- `seisai-pick` は `numba` 依存があります。
- `seisai-models` は `timm` を使います。
- `seisai-engine` は `mlflow>=2.0` を依存に含みます（tracking を使わない場合でも import には入ります）。

## Quick Start: SegyGatherPipelineDataset

「SEG-Y → gather 抽出 → transform → gate → BuildPlan で input/target を組み立て」の最小例です。

```python
from torch.utils.data import DataLoader

from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    SegyGatherPipelineDataset,
)
from seisai_dataset.builder.builder import (
    IdentitySignal,
    MaskedSignal,
    SelectStack,
    # FBGaussMap,
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

# ---- 2) gate: quickstart は安定化のため FBLC を無効化
fbgate = FirstBreakGate(
    FirstBreakGateConfig(
        apply_on="off",
        min_pick_ratio=0.0,
    )
)

# ---- 3) plan: input/target を構成 (例: recon + masking)
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
        # first-break heatmap が必要なら:
        # FBGaussMap(dst="fb_map", sigma=1.5, src="fb_idx_view"),
    ],
    input_stack=SelectStack(keys="x_masked", dst="input"),
    target_stack=SelectStack(keys="x_orig", dst="target"),
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

# batch は dict
# 代表的なキー: input, target(※plan が作る), mask_bool(任意), meta, trace_valid(任意),
#              dt_sec, offsets, indices, file_path, key_name, secondary_key, primary_unique,
#              fb_idx, did_superwindow
```

## Quick Start: Phase Picks (P/S/Noise)

CSR 形式の phase pick から P/S/Noise (3-class) の soft target map を作る例です。

仕様:
- `docs/spec/phase_pick_files_spec.md`
- `docs/spec/segy_gather_phase_pipeline_dataset_output_contract.md`

```python
from torch.utils.data import DataLoader

from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    SegyGatherPhasePipelineDataset,
)
from seisai_dataset.builder.builder import (
    IdentitySignal,
    PhasePSNMap,
    SelectStack,
)
from seisai_transforms.augment import (
    PerTraceStandardize,
    RandomCropOrPad,
    ViewCompose,
)

transform = ViewCompose([
    PerTraceStandardize(),
    RandomCropOrPad(target_len=2048),
])

fbgate = FirstBreakGate(
    FirstBreakGateConfig(
        apply_on="off",
        min_pick_ratio=0.0,
    )
)

plan = BuildPlan(
    wave_ops=[IdentitySignal(src="x_view", dst="x", copy=False)],
    label_ops=[PhasePSNMap(dst="psn_map", sigma=1.5)],
    input_stack=SelectStack(keys="x", dst="input"),
    target_stack=SelectStack(keys="psn_map", dst="target"),
)

ds = SegyGatherPhasePipelineDataset(
    segy_files=["/path/input.sgy"],
    phase_pick_files=["/path/phase_picks.npz"],
    transform=transform,
    fbgate=fbgate,
    plan=plan,
    include_empty_gathers=False,
    use_header_cache=True,
)

loader = DataLoader(ds, batch_size=1, num_workers=0)
batch = next(iter(loader))
```

## Pair dataset example

入力 SEG-Y とターゲット SEG-Y のペアを同期サンプリングして `input/target` を作る例です。

仕様:
- `docs/spec/segy_gather_pair_dataset_spec.md`

```python
from torch.utils.data import DataLoader

from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_transforms.augment import RandomCropOrPad, ViewCompose

input_segy_files = ["/path/noisy_001.sgy", "/path/noisy_002.sgy"]
target_segy_files = ["/path/clean_001.sgy", "/path/clean_002.sgy"]

transform = ViewCompose([RandomCropOrPad(target_len=2048)])

plan = BuildPlan(
    wave_ops=[
        IdentitySignal(src="x_view_input", dst="x_in", copy=False),
        IdentitySignal(src="x_view_target", dst="x_tg", copy=False),
    ],
    label_ops=[],
    input_stack=SelectStack(keys=["x_in"], dst="input"),
    target_stack=SelectStack(keys=["x_tg"], dst="target"),
)

ds = SegyGatherPairDataset(
    input_segy_files=input_segy_files,
    target_segy_files=target_segy_files,
    transform=transform,
    plan=plan,
    use_header_cache=True,
)

loader = DataLoader(ds, batch_size=4, num_workers=2)
batch = next(iter(loader))
```

## Inference: gather window 列挙 + tiled 推論

推論向けに「**gather を決定論で window 列挙**」する Dataset が `seisai_dataset.infer_window_dataset` にあります。

- `InferenceGatherWindowsDataset`: H 方向を window 列挙（不足は pad）、W 方向は crop せず不足時のみ右 0pad
- `collate_pad_w_right`: 可変 W をバッチでまとめるための右 0pad collate
- `InputOnlyPlan`: `BuildPlan` から `InputOnlyPlan.from_build_plan(...)` で推論用 plan を作れます

> この Dataset は推論で RNG が呼ばれると例外を投げる設計です（決定論の崩れを検知するため）。

```python
from seisai_dataset.infer_window_dataset import (
    InferenceGatherWindowsDataset,
    InferenceGatherWindowsConfig,
    collate_pad_w_right,
)

cfg = InferenceGatherWindowsConfig(
    domains=("shot",),
    win_size_traces=128,
    stride_traces=64,
    target_len=6016,
)

ds = InferenceGatherWindowsDataset(
    segy_files=["/path/input.sgy"],
    fb_files=["/path/fb.npy"],
    plan=plan,  # BuildPlan でも OK（内部で InputOnlyPlan に変換）
    cfg=cfg,
)

# 可変Wなので collate_fn を差し替える
# (x_bchw, metas) を返す
# loader = DataLoader(ds, batch_size=4, collate_fn=collate_pad_w_right)
```

tiled 推論は `seisai_engine.infer.runner` にあり、H/W の両方向をサポートします。

- `infer_batch_tiled_w` / `iter_infer_loader_tiled_w` / `run_infer_loader_tiled_w`
- `infer_batch_tiled_h` / `iter_infer_loader_tiled_h` / `run_infer_loader_tiled_h`

※ `seisai_engine` のトップレベル export は現在 `*_tiled_w` のみです（`*_tiled_h` は `seisai_engine.infer.runner` から import）。

- 設定クラス:
  - `TiledWConfig` は `from seisai_engine import TiledWConfig` で import
  - `TiledHConfig` は `from seisai_engine.infer.runner import TiledHConfig` で import

## Examples (実行スクリプト)

### 1) データセット単体の quick check

```bash
# noise-only TraceSubset の簡易チェック
python packages/seisai-dataset/examples/noise_dataset_quick_check.py

# phase pick dataset の簡易チェック
python packages/seisai-dataset/examples/phase_dataset_quick_check.py
```

### 2) 学習スクリプト (root/examples)

このリポジトリには **YAML 設定で学習/推論/可視化まで回すサンプル**が入っています。

- `examples/example_train_psn.py` : P/S/Noise (3-class) 学習 + 推論 + 可視化
- `examples/example_train_pair.py` : paired SEG-Y 学習 + tiled 推論 + triptych 可視化
- `examples/example_train_fbp.py` : first-break 系の学習例
- `examples/examples_train_blindtrace.py` : mask/blindtrace 系の学習例

#### config の `paths` セクション（train と infer を分ける）

YAML で指定する入力ファイルは、**train 用**と **infer（評価/可視化）用**を `paths` で別々に受け取ります。
（同じファイルを使う場合でも、対応する `infer_*` を明示します。）

- PSN / blindtrace:
  - train: `paths.segy_files`, `paths.phase_pick_files`
  - infer: `paths.infer_segy_files`, `paths.infer_phase_pick_files`
- Pair:
  - train: `paths.input_segy_files`, `paths.target_segy_files`
  - infer: `paths.infer_input_segy_files`, `paths.infer_target_segy_files`
- 共通:
  - `paths.out_dir`: 出力先（相対パスは YAML ファイルの場所基準で解決）

また、各 `*_files` は `list[str]` だけでなく **listfile へのパス（`str`）** でも指定できます（1行1パス）。
listfile 内の相対パスは、**listfile のあるディレクトリ**基準で解決されます。

実行例:

```bash
# PSN (設定ファイルは examples/config_train_psn.yaml)
python examples/example_train_psn.py --config examples/config_train_psn.yaml

# Pair
python examples/example_train_pair.py --config examples/config_train_pair.yaml
```

tracking 設定（最小例）:

```yaml
tracking:
  enabled: true
  exp_name: baseline
  tracking_uri: file:./mlruns
  vis_best_only: true
  vis_max_files: 50
```

注意:
- `tracking_uri` の相対パスは **YAML ファイルの場所**基準で解決されます（詳細は `docs/spec/mlflow_tracking_spec.md`）。

出力レイアウト（共通）:
- best checkpoint: `out_dir/ckpt/best.pt`
- vis: `out_dir/<vis.out_subdir>/epoch_####/step_####.png`（`vis.out_subdir` のデフォルトは `vis`）

同梱データでの最小実行 (1 epoch / PNG 出力):

```bash
python examples/example_train_psn.py --config tests/e2e/config_train_psn.yaml
```

### 3) パッケージ内の小さな例

```bash
python packages/seisai-engine/example/example2.py
python packages/seisai-engine/example/example_mask_velocity.py
python packages/seisai-engine/example/example_trend_prior_op.py

python packages/seisai-transforms/example/example_mask.py
python packages/seisai-pick/example/example_trend_fit.py
```

## Testing

```bash
pytest -q
```

- `tests/e2e/` には **同梱サンプル (tests/data)** を使う軽量な e2e が含まれます。
- `packages/seisai-dataset/tests/` には、実データがある場合にだけ動く integration が含まれます。

ユニットテストだけ回したい場合:

```bash
pytest -q -m "not integration"
```

### dataset の integration (実データが必要)

`seisai-dataset` の一部 integration テストは大きい SEG-Y と first-break を必要とし、無い場合は skip されます。

```bash
export FBP_TEST_SEGY=/abs/path/sample.sgy
export FBP_TEST_FB=/abs/path/fb.npy
pytest -q -m integration
```

## ドキュメント

- SegyGatherPipelineDataset 出力契約: `docs/spec/segy_gather_pipeline_dataset_output_contract.md`
- SegyGatherPipelineDataset 入力前提 (SEG-Y): `docs/spec/segy_gather_pipeline_dataset_input_assumptions.md`
- Phase pick ファイル仕様: `docs/spec/phase_pick_files_spec.md`
- Phase dataset 出力契約: `docs/spec/segy_gather_phase_pipeline_dataset_output_contract.md`
- Pair dataset 仕様: `docs/spec/segy_gather_pair_dataset_spec.md`
- BuildPlan 契約: `docs/spec/build_plan_contract.md`
- Mask 契約: `docs/spec/mask_contract.md`
- Training pipeline 出力レイアウト: `docs/spec/training_pipeline_output_layout.md`
- MLflow tracking 仕様: `docs/spec/mlflow_tracking_spec.md`

## License

MIT
