# Pairタスク トレーニング・クイックスタート（日本語 / Markdown）

このレポの **Pairタスク** は、**input SEG-Y** と **target SEG-Y** を **1:1のペア**として扱い、同じgather window（同じtrace subset）を切り出して **波形の回帰（mapping）** を学習します。

---

## 1) Pairタスクの定義、ユースケース

### Pairタスクの定義
- **教師あり回帰**：`input wave → target wave` を学習するタスク
- 1サンプルは基本的に以下の形（モデル入力/教師）
  - `input`: `(1, H, W)`（H=trace方向, W=time方向）
  - `target`: `(1, H, W)`
- 学習時は **input/targetに同じtransform（同期transform）** が適用されます（crop/augmentがズレない）。
- 推論時は **H方向（trace方向）タイル推論** を前提にしています（`tile` セクションが必須）。

### ユースケース（必須指定の2例）
#### (A) ノイズ抑制（Denoise）
- **input**：ノイズが入ったデータ（例：field noise混入）
- **target**：クリーンデータ（ノイズ無し/低ノイズ）
- 目的：`noisy → clean` の写像を学習して、推論でノイズを低減した波形を得る

#### (B) 低周波数補完（Low-frequency completion）
- **input**：高周波側が主なデータ（例：高域のみ残した/低域が欠落したデータ）
- **target**：低周波側が主なデータ（例：低域成分、または低域を含む“補完後”データ）
- 目的：`high-freq → low-freq(または補完後)` の写像を学習して、低周波を補う

> 重要：Pairは「同じgeometry/同じtrace配列」を前提に、同じindicesでinputとtargetを読みます。
> ノイズ抑制でも低周波補完でも、**inputとtargetは“対応する同一データ”として整列**している必要があります。

---

## 2) inputデータ / targetデータの設定方法

### 2.1 ペア整合性の必須条件（一致しないと即エラー）
各ペア（`input_segy_files[i]` と `target_segy_files[i]`）について、最低限次が一致している必要があります：

- **サンプル数（nsamples）**
- **トレース数（n_traces / tracecount）**
- **サンプリング間隔（dt）**

さらに実運用上は、次も揃っているのが望ましいです：
- trace順序（並び）が同一
- gather抽出に使うヘッダキー（例：FFIDなど）が整合

### 2.2 `paths.*_segy_files` の指定方法（2通り）
PairのYAMLでは、以下4つが必須です（学習用 + 推論/可視化用）：

- `paths.input_segy_files`
- `paths.target_segy_files`
- `paths.infer_input_segy_files`
- `paths.infer_target_segy_files`

指定方法は2通りあります。

#### 方法A：YAMLに直接 `list[str]` で列挙
```yaml
paths:
  input_segy_files:
    - /data/pair/train/noisy_0001.sgy
    - /data/pair/train/noisy_0002.sgy
  target_segy_files:
    - /data/pair/train/clean_0001.sgy
    - /data/pair/train/clean_0002.sgy
  infer_input_segy_files:
    - /data/pair/val/noisy_0101.sgy
  infer_target_segy_files:
    - /data/pair/val/clean_0101.sgy
  out_dir: ./_pair_out
```

#### 方法B：listfile（1行1パスのテキストファイル）を指定
`paths.*_segy_files` は **listfileへのパス（文字列）** でも指定できます。

**listfileのルール**
- 1行1パス
- 空行は無視
- `#` で始まる行はコメントとして無視
- `~` と環境変数（`$VAR`）展開あり
- listfile内の相対パスは **listfile自身のディレクトリ基準**

例：`data/train_input.txt`
```text
# noisy segy
/data/pair/train/noisy_0001.sgy
/data/pair/train/noisy_0002.sgy
```

例：`data/train_target.txt`
```text
# clean segy (same order as input)
 /data/pair/train/clean_0001.sgy
 /data/pair/train/clean_0002.sgy
```

YAML側はこう書きます：
```yaml
paths:
  input_segy_files: data/train_input.txt
  target_segy_files: data/train_target.txt
  infer_input_segy_files: data/val_input.txt
  infer_target_segy_files: data/val_target.txt
  out_dir: ./_pair_out
```

**ペア対応の作り方（超重要）**
- `input` と `target` の list は **同じ行番号がペア**です
  - 例：`train_input.txt` の1行目 ↔ `train_target.txt` の1行目

---

## 最低限の実行に必要な設定（YAMLテンプレ）

下のテンプレは「動かす」ための最小構成です（listfile方式）。

```yaml
paths:
  input_segy_files:  data/train_input.txt
  target_segy_files: data/train_target.txt
  infer_input_segy_files:  data/val_input.txt
  infer_target_segy_files: data/val_target.txt
  out_dir: ./_pair_out

dataset:
  max_trials: 2048
  use_header_cache: true
  verbose: true
  primary_keys: [ffid]          # gather抽出の主キー（まずはffid推奨）
  secondary_key_fixed: false
  # endianがbig以外なら指定（任意）
  # train_input_endian: little
  # train_target_endian: little
  # infer_input_endian: little
  # infer_target_endian: little
  # mmapにするなら workers=0 制約（任意）
  # waveform_mode: eager   # eager | mmap

train:
  device: auto                  # auto | cpu | cuda | cuda:N
  batch_size: 8
  epochs: 10
  lr: 3.0e-4
  subset_traces: 128            # 1サンプルで使うtrace本数（H）
  samples_per_epoch: 1024
  seed: 42
  use_amp: true
  max_norm: 1.0
  num_workers: 0                # mmap利用時は必ず0

  loss_scope: all
  losses:
    - kind: l1
      weight: 1.0
      scope: all
      params: {}

transform:
  time_len: 2048                # 学習時のみ時間方向crop/pad（推論は時間cropしない）

infer:
  batch_size: 1
  max_batches: 8
  subset_traces: 128            # tile.tile_h以上にすること
  seed: 43
  num_workers: 0                # 固定推論のため0必須

tile:                           # Pairでは必須
  tile_h: 64                    # trace方向タイルサイズ（<= infer.subset_traces）
  overlap_h: 32
  tiles_per_batch: 16
  amp: true
  use_tqdm: false

vis:
  out_subdir: vis
  n: 5
  cmap: seismic
  vmin: -3
  vmax: 3
  transpose_for_trace_time: true
  per_trace_norm: true
  per_trace_eps: 1.0e-8
  figsize: [13.0, 8.0]
  dpi: 150

model:
  backbone: resnet18
  pretrained: false
  in_chans: 1
  out_chans: 1

ckpt:
  save_best_only: true
  metric: infer_loss
  mode: min
```

---

## 実行手順（最短）

### 0. 依存インストール（モノレポ / editable install）
レポ直下で以下（依存順が重要）：

```bash
python -m pip install -U pip

for p in seisai-utils seisai-transforms seisai-pick seisai-dataset seisai-models seisai-engine; do
  python -m pip install -e "packages/$p"
done
```

### 1. 設定YAMLを用意
- 上のテンプレを `configs/train_pair.yaml` などに保存
- `data/train_input.txt` / `data/train_target.txt` / `data/val_input.txt` / `data/val_target.txt` を作る

### 2. トレーニング実行
```bash
python cli/run_pair_train.py --config configs/train_pair.yaml
```

---

## 出力（どこに何ができるか）
`paths.out_dir`（例：`./_pair_out`）配下に最低限こう出ます：

- `ckpt/best.pt`：推論損失（`infer_loss`）が最良の重み
- `vis/`：Input / Target / Pred のtriptych可視化画像（設定 `vis.n` 枚）

---

## つまずきポイント（エラーになりやすい条件）
- **input/targetのdt・サンプル数・トレース数が一致していない**
  - まずは同一元データからフィルタ処理してSEG-Yを作るのが安全（ノイズ付与/帯域分離など）
- `tile.tile_h > infer.subset_traces`
  - `infer.subset_traces` を増やすか、`tile.tile_h` を下げる
- `infer.num_workers != 0`
  - 固定推論のため **0必須**
- `dataset.waveform_mode: mmap` にしているのに `train.num_workers > 0` / `infer.num_workers > 0`
  - mmapは **I/O安全のためworkers=0必須**
