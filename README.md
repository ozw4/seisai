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

## Init checkpoint (`train.init_ckpt`)

- `blindtrace` / `psn` / `pair` の全タスクで `train.init_ckpt` を共通利用できます。
- 想定する重みファイルは `train_skeleton` が保存する `best.pt` です。
- `in_chans` が不一致の場合は即時エラーで停止します（部分ロードしません）。
- `out_chans` が不一致の場合は `seg_head.*` を除外してロードします。
- 上記以外の重みは完全一致を要求し、`missing/unexpected` があればエラーで停止します。

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

## Quick Start: Pair トレーニング

Pair は **入力 SEG-Y** と **ターゲット SEG-Y** を 1:1 のペアとして扱い、同一 gather window（同一 trace subset）を切り出して **波形回帰（mapping）** を学習します。

代表ユースケース（例）:
- ノイズ抑制: `noisy → clean`
- 低周波数補完: `high-freq → low-freq`（または帯域補完後の波形）

### 0) 前提（データ整合）
各ペア（`input_segy_files[i]` と `target_segy_files[i]`）で、少なくとも以下が一致している必要があります。

- サンプル数（`n_samples`）
- トレース数（`n_traces`）
- サンプリング間隔（`dt`）

加えて、ファイルリストは **同数** かつ **同じ順序**（同じ行番号がペア）に整列させてください。

### 1) デモ実行（同梱 config）
サンプル config: `examples/config_train_pair.yaml`

```bash
python cli/run_pair_train.py --config examples/config_train_pair.yaml
```

> `paths.out_dir` が相対パスの場合、**YAML ファイルのあるディレクトリ基準**で解決されます。

### 2) 自分のデータで実行（paths だけ差し替え）
`examples/config_train_pair.yaml` の `paths.*_segy_files` を更新します。
`paths.*_segy_files` は (A) パス配列の直書き、または (B) listfile（1行1パスのテキスト）を指定できます。

例（listfile 指定）:

```yaml
paths:
  input_segy_files:  data/train_input.txt
  target_segy_files: data/train_target.txt
  infer_input_segy_files:  data/val_input.txt
  infer_target_segy_files: data/val_target.txt
  out_dir: ./_pair_out
```

学習設定の注意（最低限）:
- 学習中評価は固定サンプルのため `infer.num_workers: 0` が必須
- `tile` セクション必須、かつ `tile.tile_h <= infer.subset_traces`

### 3) 生成物
- checkpoint: `<out_dir>/ckpt/best.pt`
- 可視化（triptych: Input/Target/Pred）: `<out_dir>/vis/...`

### 4) 推論（SEG-Y → SEG-Y）
サンプル config: `examples/config_infer_pair.yaml`

```bash
python cli/run_pair_infer.py --config examples/config_infer_pair.yaml
```

詳細（paths / YAML / 制約 / 推論の入口）は `docs/examples/quick_start_pair.md` を参照してください。


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

shared_transform = ViewCompose([RandomCropOrPad(target_len=2048)])
input_transform = shared_transform
target_transform = shared_transform

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
    input_transform=input_transform,
    target_transform=target_transform,
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

### 2) 学習スクリプト (root/cli)

このリポジトリには **YAML 設定で学習/推論/可視化まで回すサンプル**が入っています。

- `cli/run_psn_train.py` : P/S/Noise (3-class) 学習 + 推論 + 可視化
- `cli/run_pair_train.py` : paired SEG-Y 学習 + tiled 推論 + triptych 可視化
- `examples/example_train_fbp.py` : first-break 系の学習例
- `cli/run_blindtrace_train.py` : mask/blindtrace 系の学習例

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
python cli/run_psn_train.py --config examples/config_train_psn.yaml

# Pair
python cli/run_pair_train.py --config examples/config_train_pair.yaml
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
python cli/run_psn_train.py --config tests/e2e/config_train_psn.yaml
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
- `run_psn_train.py` スモークテスト手順: `docs/testing/smoke_run_train_psn.md`
- `SegyGatherPairDataset` 例外系テスト技術メモ: `docs/testing/-test_segy_gather_pair_dataset_exceptions.md`
- `SegyGatherPairDataset` Pair整合性テスト技術メモ: `docs/testing/test_segy_gather_pair_dataset_pair_consistency.md`

## License

MIT
