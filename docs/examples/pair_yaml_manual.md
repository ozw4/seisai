# Pair タスク: YAML 設定マニュアル（examples/config_train_pair.yaml）

本メモは `examples/config_train_pair.yaml`（Pair 学習パイプライン）で使用される YAML 設定の **キー・型・制約・実挙動**を、コード実装に合わせて整理したもの。
対象エントリポイントは `examples/example_train_pair.py` → `seisai_engine.pipelines.pair.train.main`。

共通事項（パス解決/listfile、augment、scheduler、ckpt、ema、tracking など）は `docs/examples/common_yaml_manual.md` を参照。

---

## 1. 目的と前提

Pair タスクは、入力 SEG-Y とターゲット SEG-Y を 1:1 対応させた **paired gather** を用い、2D Encoder-Decoder（`EncDec2D`）で **(H,W) の波形マップ**を回帰する。

- 入力: input SEG-Y gather window（`(H,W)` → `(1,H,W)`）
- 教師: target SEG-Y gather window（`(H,W)` → `(1,H,W)`）
- 損失: pixel-wise `l1` / `mse`
- 推論: H 方向タイル（`tile` セクション）で分割推論し、可視化は triptych（Input/Target/Pred）

---

## 2. 参照 YAML（examples/config_train_pair.yaml の現物）

```yaml
paths:
  input_segy_files:
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.30.00_bpf.sgy
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.31.00_bpf.sgy
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.32.00_bpf.sgy
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.33.00_bpf.sgy
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.34.00_bpf.sgy

  target_segy_files:
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.30.00_NR.sgy
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.31.00_NR.sgy
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.32.00_NR.sgy
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.33.00_NR.sgy
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.34.00_NR.sgy
  infer_input_segy_files:
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.30.00_bpf.sgy
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.31.00_bpf.sgy
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.32.00_bpf.sgy
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.33.00_bpf.sgy
    - /home/dcuser/data/kshitf22/raw/F1_2025-01-05_15.34.00_bpf.sgy

  infer_target_segy_files:
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.30.00_NR.sgy
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.31.00_NR.sgy
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.32.00_NR.sgy
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.33.00_NR.sgy
    - /home/dcuser/data/kshitf22/processed/F1_2025-01-05_15.34.00_NR.sgy
  out_dir: ./_pair_out

dataset:
  max_trials: 2048
  use_header_cache: true
  verbose: true
  primary_keys: [ffid]
  secondary_key_fixed: false

train:
  device: auto # auto | cpu | cuda | cuda:N
  batch_size: 8
  epochs: 100
  lr: 1.0e-4
  subset_traces: 128
  time_len: 6016
  samples_per_epoch: 256
  loss_kind: l1            # l1 or mse
  seed: 42
  use_amp: true
  max_norm: 1.0
  num_workers: 0


# ema:
#   enabled: true
#   decay: 0.999
#   start_step: 0
#   update_every: 1
#   use_for_infer: true
#   device: cpu  # omit -> same device as model; "cpu" saves VRAM but is slower

augment:
  hflip_prob: 0.0
  polarity_prob: 0.0
  space:
    prob: 0.0
    factor_range: [0.90, 1.10]
  time:
    prob: 0.0
    factor_range: [0.95, 1.05]
  freq:
    prob: 0.0
    kinds: [bandpass, lowpass, highpass]
    band: [0.05, 0.45]
    width: [0.10, 0.35]
    roll: 0.02
    restandardize: false

infer:
  batch_size: 1
  max_batches: 12
  subset_traces: 4096
  seed: 43
  num_workers: 0

tile:
  tile_h: 128
  overlap_h: 64
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
  figsize: [20.0, 15.0]
  dpi: 300

model:
  backbone: resnet18 # edgenext_small.usi_in1k  caformer_b36.sail_in22k_ft_in1k, resnet18
  # pretrained: false # default
  in_chans: 1
  out_chans: 1
  # stage_strides: null
  # extra_stages: 0
  # extra_stage_strides: null
  # extra_stage_channels: null
  # extra_stage_use_bn: true
  # pre_stages: 0
  # pre_stage_strides: null
  # pre_stage_kernels: null
  # pre_stage_channels: null
  # pre_stage_use_bn: true
  # decoder_channels: [256, 128, 64, 32]
  # decoder_scales: [2, 2, 2, 2]
  # upsample_mode: bilinear
  # attention_type: scse # or null
  # intermediate_conv: true

ckpt:
  save_best_only: true
  metric: infer_loss
  mode: min

tracking:
  enabled: true
  exp_name: baseline
  tracking_uri: file:./mlruns
  vis_best_only: true
  vis_max_files: 50
```

---

## 3. `paths` セクション

パス解決・listfile 展開の詳細は `docs/examples/common_yaml_manual.md` を参照。

### 3.1 必須キー
| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `paths.input_segy_files` | `list[str]` または `str` | Yes | 学習用 input SEG-Y ファイル群。`target_segy_files` と **同じ長さ**が必要。空リストは不可。 |
| `paths.target_segy_files` | `list[str]` または `str` | Yes | 学習用 target SEG-Y ファイル群。input と 1:1 対応。空リストは不可。 |
| `paths.infer_input_segy_files` | `list[str]` または `str` | Yes | 推論（評価/可視化）用 input SEG-Y ファイル群。空リストは不可。 |
| `paths.infer_target_segy_files` | `list[str]` または `str` | Yes | 推論用 target SEG-Y ファイル群。`infer_input_segy_files` と **同じ長さ**が必要。空リストは不可。 |
| `paths.out_dir` | `str` | Yes | 出力先ディレクトリ。相対指定は YAML の場所基準で解決。 |

### 3.2 整合性チェック
- `len(input_segy_files) == len(target_segy_files)` を要求
- `len(infer_input_segy_files) == len(infer_target_segy_files)` を要求
- 全ファイルの存在をチェック（存在しないと即エラー）

### 3.3 SEG-Y ペア整合性
各ペアについて以下が一致しない場合は **即エラー**。

- `n_samples`（時間サンプル数）
- `n_traces`（トレース数）
- `dt`（サンプル間隔）

---

## 4. `dataset` セクション

`SegyGatherPairDataset` の生成パラメータ。
`dataset` セクション自体は **必須**（`progress` と `waveform_mode` は optional）。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `dataset.max_trials` | `int` | Yes | - | サンプル生成のリトライ上限。小さすぎると有効サンプルを引けずエラーになり得る。 |
| `dataset.use_header_cache` | `bool` | Yes | - | SEG-Y header を `*.headers.npz` にキャッシュして高速化する。SEG-Y と同階層に sidecar が作られる。 |
| `dataset.verbose` | `bool` | Yes | - | Dataset 内部のログ/情報出力の有無。 |
| `dataset.progress` | `bool` | No | `dataset.verbose` | インデクシング時の tqdm 表示。未指定なら `verbose` と同じ。 |
| `dataset.primary_keys` | `list[str]` | Yes | - | gather 抽出の主キー（例: `ffid`）。空は禁止、重複禁止。 |
| `dataset.secondary_key_fixed` | `bool` | Yes | - | 2次整列（secondary key）ルールを固定するか。 **学習側のみ**この値が反映される（推論側は常に固定）。 |
| `dataset.waveform_mode` | `str` | No | `eager` | `eager` / `mmap`。`mmap` はメモリ節約だが `train.num_workers=0` と `infer.num_workers=0` が必須。 |

補足:
- `primary_keys=['ffid']` の場合、`secondary_key_fixed=false` なら secondary が `chno` と `offset` からランダムに選ばれ得る。
- 推論データセットは **必ず** `secondary_key_fixed=True` として生成される。

---

## 5. `train` セクション（学習ループ + Pair 固有）

`train` セクションは **必須**。

### 5.1 学習ループ系（共通）
| key | 型 | 必須 | 意味 |
|---|---:|:---:|---|
| `train.device` | `str` | No | 学習デバイス指定。`auto` / `cpu` / `cuda` / `cuda:N`。`auto` は CUDA があれば GPU、なければ CPU。 |
| `train.batch_size` | `int` | Yes | 学習 DataLoader の batch size。 |
| `train.epochs` | `int` | Yes | epoch 数。 |
| `train.samples_per_epoch` | `int` | Yes | 1 epoch あたりに使用するサンプル数（`Subset(ds_train_full, range(samples_per_epoch))`）。 |
| `train.seed` | `int` | Yes | 学習の乱数 seed（epoch ごとに `seed+epoch` を使用）。 |
| `train.use_amp` | `bool` | Yes | AMP を使うか（CUDA 時のみ有効）。 |
| `train.max_norm` | `float` | Yes | gradient clipping の max norm。 |
| `train.num_workers` | `int` | Yes | 学習 DataLoader worker 数。0 の場合は main process が dataset RNG を直接更新。 |
| `train.print_freq` | `int` | No | 学習中のログ出力頻度（step 間隔）。未指定時は `10`。 |

### 5.2 Pair 固有
| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `train.lr` | `float` | Yes | AdamW の学習率。 |
| `train.subset_traces` | `int` | Yes | 学習データセットで切り出すトレース本数（H）。 |
| `train.time_len` | `int` | Yes | 学習時の時間長（W）。`RandomCropOrPad` により W を調整。 |
| `train.loss_kind` | `str` | Yes | `l1` / `mse` のみ。pixel-wise 損失に対応。 |

### 5.3 `train.time_len` の挙動
- `train.time_len < 元W` は **ランダム crop**
- `train.time_len > 元W` は **右側ゼロパディング**
- 推論側は **時間方向の crop/pad を行わない**（元の W のまま）
- 変換は input/target に同期適用され、学習/推論ともに `PerTraceStandardize(eps=1e-8)` が固定で入る

---

## 6. `infer` セクション（評価/可視化ループ）

`infer` セクションは **必須**。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `infer.batch_size` | `int` | Yes | 推論 DataLoader の batch size。 |
| `infer.max_batches` | `int` | Yes | 推論で処理する最大 batch 数。推論データセットは `batch_size * max_batches` 件で固定 subset される。 |
| `infer.seed` | `int` | Yes | 各 epoch の推論前に dataset RNG をこの seed で再初期化し、推論サンプルを固定化する。 |
| `infer.num_workers` | `int` | Yes | **必ず 0**（0 以外はエラー）。推論サンプル固定のため。 |
| `infer.subset_traces` | `int` | Yes | 推論データセットで切り出すトレース本数（H）。 |

---

## 7. `tile` セクション（H 方向タイル推論）

Pair 推論は `infer_batch_tiled_h` を使って **H 方向タイル**で処理する。未指定は不可。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `tile.tile_h` | `int` | Yes | H 方向のタイル高さ。`tile_h <= infer.subset_traces` が必須。 |
| `tile.overlap_h` | `int` | Yes | H 方向のオーバーラップ。`0 <= overlap_h < tile_h`。 |
| `tile.tiles_per_batch` | `int` | Yes | 1バッチあたりのタイル数。`>0`。 |
| `tile.amp` | `bool` | Yes | タイル推論時に AMP を使うか（CUDA 時のみ有効）。 |
| `tile.use_tqdm` | `bool` | Yes | タイル推論の進捗表示（tqdm）。 |

補足:
- 入力 `H < tile_h` はエラー（フォールバックなし）。
- `model.out_chans` は **1 固定**（タイル推論のバリデーションで強制）。

---

## 8. `vis` セクション（triptych 可視化）

| key | 型 | 必須 | 意味 |
|---|---:|:---:|---|
| `vis.out_subdir` | `str` | Yes | `paths.out_dir` 配下の可視化出力サブディレクトリ名。 |
| `vis.n` | `int` | Yes | 推論ループの先頭 `n` batch について PNG を保存。 |
| `vis.cmap` | `str` | Yes | 可視化カラーマップ名。 |
| `vis.vmin` | `float` | Yes | imshow の vmin。 |
| `vis.vmax` | `float` | Yes | imshow の vmax。 |
| `vis.transpose_for_trace_time` | `bool` | Yes | trace/time 軸を転置して描画するか。 |
| `vis.per_trace_norm` | `bool` | Yes | 各トレースごとに z-score 正規化して描画するか。 |
| `vis.per_trace_eps` | `float` | Yes | per-trace 正規化の epsilon。 |
| `vis.figsize` | `list[float]` | Yes | 図のサイズ（例: `[20.0, 15.0]`）。 |
| `vis.dpi` | `int` | Yes | DPI。 |

出力パス規約（学習時）:
- `out_dir/<vis.out_subdir>/epoch_####/step_####.png`

補足:
- CLI 推論（`seisai_engine.pipelines.pair.infer`）では `out_dir/<vis.out_subdir>/pair_triptych_step####.png` が作られる。

---

## 9. `model` セクション

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `model.backbone` | `str` | Yes | 例: `resnet18`。`EncDec2D` の backbone 名。 |
| `model.pretrained` | `bool` | No | backbone の pretrained を使うか。未指定なら `false`。 |
| `model.in_chans` | `int` | Yes | 入力チャネル数。**現行パイプラインでは 1 固定**。 |
| `model.out_chans` | `int` | Yes | 出力チャネル数。**現行パイプラインでは 1 固定**。タイル推論で強制。 |
| `model.stage_strides` | `null` or `list[[int,int]]` | No | backbone の stage stride を上書き。未指定なら `null`。 |
| `model.extra_stages` | `int` | No | 追加 downsample 段数。未指定なら `0`。 |
| `model.extra_stage_strides` | `null` or `list[[int,int]]` | No | 追加段 stride。未指定なら `null`。 |
| `model.extra_stage_channels` | `null` or `list[int]` | No | 追加段 channel。未指定なら `null`。 |
| `model.extra_stage_use_bn` | `bool` | No | 追加段 BN 有無。未指定なら `true`。 |
| `model.pre_stages` | `int` | No | 前段 Conv+BN+ReLU 段数。未指定なら `0`。 |
| `model.pre_stage_strides` | `null` or `list[[int,int]]` | No | 前段 stride。未指定なら `null`。 |
| `model.pre_stage_kernels` | `null` or `list[int]` | No | 前段 kernel。未指定なら `null`。 |
| `model.pre_stage_channels` | `null` or `list[int]` | No | 前段 channel。未指定なら `null`。 |
| `model.pre_stage_use_bn` | `bool` | No | 前段 BN 有無。未指定なら `true`。 |
| `model.decoder_channels` | `list[int]` | No | decoder channels。未指定なら `[256,128,64,32]`。 |
| `model.decoder_scales` | `list[int]` | No | decoder scale factors。未指定なら `[2,2,2,2]`。 |
| `model.upsample_mode` | `str` | No | upsample mode。未指定なら `bilinear`。 |
| `model.attention_type` | `null` or `str` | No | attention 種別。`null` で無効。未指定なら `scse`。 |
| `model.intermediate_conv` | `bool` | No | decoder の中間 conv を使うか。未指定なら `true`。 |

---

## 10. 実行（例）

```bash
python examples/example_train_pair.py --config examples/config_train_pair.yaml
```

出力先は `paths.out_dir`。

---

## 11. よくある落とし穴（チェックリスト）

- `paths.*_segy_files` の長さ不一致（input と target）がある
- input/target で `n_samples` / `n_traces` / `dt` が一致しない
- `infer.num_workers` を 0 以外にしている（即エラー）
- `dataset.waveform_mode="mmap"` で `train.num_workers` / `infer.num_workers` が 0 以外（即エラー）
- `tile.tile_h > infer.subset_traces` または `overlap_h >= tile_h`
- `model.out_chans!=1`（タイル推論で即エラー）
