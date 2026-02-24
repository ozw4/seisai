# seisai-engine

SeisAI 向けの **学習ループ / 推論ループ / ロス・メトリクス / 可視化 / 追跡（MLflow）** をまとめたパッケージです。

- YAML 設定を読み込み、`pipelines/*` のタスク（PSN / pair / blindtrace）を **同一の骨格（train_skeleton）** で回します
- 推論では **タイル推論**（H/W 方向）と、FFID gather 単位の **SEGY→SEGY 書き出し**ユーティリティを提供します
- 実験追跡は `tracking.enabled` を立てたときだけ有効化され、無効時は No-op になります（ただし依存として `mlflow` は入ります）

> 注: `import seisai_engine` のトップレベル export は現在 `*_tiled_w` 系のみです（`TiledHConfig`/H方向タイル等は `seisai_engine.infer.runner` から import）。

## Requirements

- Python >= 3.10
- 依存（主要）: `torch`, `numpy`, `tqdm`, `mlflow>=2.0`
- モノレポ内依存: `seisai-dataset`, `seisai-transforms`, `seisai-models`, `seisai-utils`

## Install（開発用 / monorepo）

リポジトリルートから editable install する想定です。

```bash
pip install -e ./packages/seisai-engine
```

（他パッケージも含めたまとめインストール例はリポジトリルートの README を参照）

## 何が入っているか（ざっくり）

### 1) Pipelines（YAML駆動の学習/推論/可視化）

- `seisai_engine.pipelines.psn` : P/S/Noise（3-class）
- `seisai_engine.pipelines.pair` : paired SEG-Y（noisy→clean 等）
- `seisai_engine.pipelines.blindtrace` : mask / blindtrace 系

各パイプラインは概ね以下を持ちます。

- `build_dataset.py` : dataset & dataloader 構築
- `build_model.py` : model 構築（`seisai-models` を利用）
- `build_plan.py` : `BuildPlan`（入力/教師の組み立て）
- `train.py` : エントリポイント（YAMLロード→train_skeleton 実行）
- `infer.py` / `vis.py` : 推論・可視化（タスクにより有無）

共通骨格は `seisai_engine.pipelines.common.train_skeleton` で、

- AMP / gradient accumulation / grad clip
- LR scheduler（step/epoch）
- checkpoint（best/min-based）
- EMA（任意）
- tracking（任意）

を統一的に扱います。

### 2) 学習ループ

- `seisai_engine.train_loop` : `train_one_epoch()`（AMP, grad accumulation, clip, scheduler.step など）
- `seisai_engine.optim` : optimizer 構築（`optimizer` セクションがあれば timm の factory を使用）
- `seisai_engine.schedulers` : warmup_cosine / step_lr / multistep_lr / cosine_annealing / warm_restarts など
- `seisai_engine.ema_controller` : EMA の更新と、推論で EMA を使うオプション

### 3) ロス / メトリクス

- `seisai_engine.loss.*`
  - `composite`（複数ロス合成）
  - `soft_label_ce`, `pixelwise_loss`, `shift_pertrace_mse`, `trend_prior_loss` など
- `seisai_engine.metrics.phase_pick_metrics`

パイプライン側（例: `pipelines/psn/loss.py`）で、設定から必要なロスを組み立てます。

### 4) 推論

- `seisai_engine.infer.runner`
  - `infer_batch_tiled_w` / `iter_infer_loader_tiled_w` / `run_infer_loader_tiled_w`
  - `infer_batch_tiled_h` / `iter_infer_loader_tiled_h` / `run_infer_loader_tiled_h`
  - **前提**: `model` は `out_chans` 属性を持ち、runner は `out_chans == 1` を要求します（満たさない場合は即失敗）。
- `seisai_engine.predict`
  - タイル結合は Hann 窓重みでブレンドし、カバレッジが欠ける場合は例外で落とします。
  - 推論用の「乱数禁止」ガードも用意されています（決定論が崩れたら即検知）。
- `seisai_engine.infer.ffid_segy2segy`
  - FFID gather 単位で推論し、入力と同じ形状・ヘッダ維持で SEGY を書き出します（per-trace 標準化→推論→denorm して保存）。

### 5) Tracking（MLflow）

- `seisai_engine.tracking.*`
  - `tracking.enabled: true` のときだけ MLflow を使います（false の場合は NoOp）。
  - `tracking.tracking_uri` が `file:` の相対パスの場合、**YAML のあるディレクトリ基準**で絶対化します。
  - data manifest を作り、入力データ集合を `data_id` として記録します。

仕様メモ: `docs/spec/mlflow_tracking_spec.md`

### 6) Viewer（簡易推論）

- `seisai_engine.viewer.fbpick`
  - checkpoint をロードし、(H,W) セクションから確率マップを推論する軽量ヘルパです。
  - `softmax_axis`（time/channel）や `output_ids` を checkpoint から解決します。

## 使い方

### 学習（推奨: リポジトリルートの CLI）

このパッケージ自体は「学習を回すロジック」を持ちますが、実行はリポジトリルートの thin CLI から呼ぶのが簡単です。

```bash
# PSN
python cli/run_psn_train.py --config examples/config_train_psn.yaml

# Pair
python cli/run_pair_train.py --config examples/config_train_pair.yaml

# blindtrace
python cli/run_blindtrace_train.py --config examples/config_train_blindtrace.yaml
```

設定の基本は以下のセクション（パイプラインにより多少差があります）:

- `paths`: 入力ファイル群と出力ディレクトリ
- `dataset` / `transform` / `plan`: dataset と view 変換、入力/教師の組み立て
- `model`: モデル定義（例: `EncDec2D`）
- `train`: lr / batch / epochs / AMP / accumulation など
- `infer` / `vis`: 推論バッチ数、可視化枚数など
- `optimizer` / `scheduler` / `ema`: 任意
- `tracking`: 任意（enabled=false がデフォルト）

tracking の最小例:

```yaml
tracking:
  enabled: true
  exp_name: baseline
  tracking_uri: file:./mlruns
  vis_best_only: true
  vis_max_files: 50
```

### Checkpoint

共通の出力レイアウト（train_skeleton）:

- best checkpoint: `out_dir/ckpt/best.pt`
- vis: `out_dir/<vis.out_subdir>/epoch_####/step_####.png`

`best.pt` の payload には概ね以下が入ります。

- `pipeline`, `epoch`, `global_step`
- `model_sig`, `model_state_dict`
- `optimizer_state_dict`
- `lr_scheduler_sig`, `lr_scheduler_state_dict`
- `cfg`
- （EMA 有効時）`ema_cfg`, `ema_state_dict`, `ema_step`, `infer_used_ema`

`train.init_ckpt` のロード規約（in_chans/out_chans の扱い等）はリポジトリルート README の **Init checkpoint** セクションを参照してください。

### タイル推論（W方向の例）

`seisai_dataset.infer_window_dataset` の window 列挙 + `collate_pad_w_right` と組み合わせる想定です。

```python
import torch
from torch.utils.data import DataLoader

from seisai_dataset.infer_window_dataset import (
    InferenceGatherWindowsDataset,
    InferenceGatherWindowsConfig,
    collate_pad_w_right,
)
from seisai_engine import TiledWConfig, run_infer_loader_tiled_w

# ds は (x_bchw, metas) を返すように collate を差し替える
cfg = InferenceGatherWindowsConfig(win_size_traces=128, stride_traces=64, target_len=6016)
ds = InferenceGatherWindowsDataset(segy_files=["/path/in.sgy"], fb_files=["/path/fb.npy"], cfg=cfg, plan=plan)
loader = DataLoader(ds, batch_size=4, collate_fn=collate_pad_w_right)

# model.out_chans == 1 が前提
logits_hw_list = run_infer_loader_tiled_w(
    model,
    loader,
    cfg=TiledWConfig(tile_w=6016, overlap_w=1024),
)
```

### FFID gather 推論→SEGY書き出し

```python
from seisai_engine.infer.ffid_segy2segy import run_ffid_gather_infer_and_write_segy

out_paths = run_ffid_gather_infer_and_write_segy(
    model,
    segy_files=["/path/in1.sgy", "/path/in2.sgy"],
    out_dir="/path/out",
    out_suffix="_pred.sgy",
    overwrite=False,
)
```

## 開発

テスト（パッケージ単体）:

```bash
pytest -q ./packages/seisai-engine/tests
```

