# BlindTrace タスク: YAML 設定マニュアル（examples/config_train_blindtrace.yaml）

本メモは `examples/config_train_blindtrace.yaml`（BlindTrace 学習パイプライン）で使用される YAML 設定の **キー・型・制約・実挙動**を、コード実装に合わせて整理したもの。
対象エントリポイントは `examples/examples_train_blindtrace.py` → `seisai_engine.pipelines.blindtrace.train.main`。

共通事項（パス解決/listfile、augment、scheduler、ckpt、ema、tracking など）は `docs/examples/common_yaml_manual.md` を参照。

---

## 1. 目的と前提

BlindTrace タスクは、入力 SEG-Y の一部トレースを **マスク/汚染**し、2D Encoder-Decoder（`EncDec2D`）で **元の波形を復元**する自己教師ありタスク。

- 入力: マスク済みの gather window（`(H,W)` → `(C,H,W)`）
- 教師: マスク前の元波形（`(H,W)` → `(1,H,W)`）
- 損失: mask されたトレースのみ（`masked_only`）または全トレース（`all`）
- 推論: H 方向タイル（`tile` セクション）で分割推論し、可視化は triptych（Input/Target/Pred）

---

## 2. 参照 YAML（examples/config_train_blindtrace.yaml の現物）

```yaml
paths:
  segy_files:
    - /home/dcuser/data/ActiveSeisField/TSTKRES/shotgath.sgy
  phase_pick_files:
    - /home/dcuser/data/ActiveSeisField/TSTKRES/fb_time_all_1341ch.crd.0613.ReMerge.npy
  infer_segy_files:
    - /home/dcuser/data/ActiveSeisField/TSTKRES/shotgath.sgy
  infer_phase_pick_files:
    - /home/dcuser/data/ActiveSeisField/TSTKRES/fb_time_all_1341ch.crd.0613.ReMerge.npy
  out_dir: ./_blindtrace_out

dataset:
  primary_keys: [ffid,chno,cmp]
  max_trials: 2048
  use_header_cache: true
  verbose: true
  train_endian: big
  infer_endian: big

transform:
  time_len: 5024
  per_trace_standardize: true

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

fbgate:
  apply_on: "off"
  min_pick_ratio: 0.1

mask:
  ratio: 0.5
  mode: replace
  noise_std: 1.0

input:
  use_offset_ch: false
  offset_normalize: true
  use_time_ch: false

train:
  device: auto # auto | cpu | cuda | cuda:N
  seed: 42
  loss_scope: masked_only # masked_only | all
  losses:
    - kind: shift_robust_mse
      weight: 1.0
      scope: masked_only
      params:
        shift_max: 5
    - kind: fx_mag_mse
      weight: 0.05
      params:
        use_log: true
        eps: 1.0e-6
        f_lo: 1
        f_hi: null
    - kind: mse
      weight: 0.02
      scope: all
      params: {}
  batch_size: 8
  num_workers: 0
  use_amp: true
  max_norm: 1.0
  lr: 1.0e-4
  weight_decay: 1.0e-4
  epochs: 10
  samples_per_epoch: 256
  subset_traces: 128

eval:
  loss_scope: masked_only
  losses:
    - kind: shift_robust_mse
      weight: 1.0
      scope: masked_only
      params:
        shift_max: 5


# ema:
#   enabled: true
#   decay: 0.999
#   start_step: 0
#   update_every: 1
#   use_for_infer: true
#   device: cpu  # omit -> same device as model; "cpu" saves VRAM but is slower

infer:
  seed: 43
  batch_size: 1
  num_workers: 0
  max_batches: 12
  subset_traces: 128

tile:
  tile_h: 128
  overlap_h: 64
  tiles_per_batch: 16
  amp: true
  use_tqdm: false

vis:
  n: 5
  out_subdir: vis
  cmap: seismic
  vmin: -3
  vmax: 3
  transpose_for_trace_time: true
  per_trace_norm: true
  per_trace_eps: 1.0e-8
  figsize: [20.0, 15.0]
  dpi: 300

model:
  backbone: resnet18 #resnet18, edgenext_small.usi_in1k
  # pretrained: false # default
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
| `paths.segy_files` | `list[str]` または `str` | Yes | 学習用 SEG-Y ファイル群。空リストは不可。 |
| `paths.infer_segy_files` | `list[str]` または `str` | Yes | 推論（評価/可視化）用 SEG-Y ファイル群。空リストは不可。 |
| `paths.out_dir` | `str` | Yes | 出力先ディレクトリ。相対指定は YAML の場所基準で解決。 |

### 3.2 任意キー（FB picks）
| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `paths.phase_pick_files` | `list[str]` または `str` | No | 学習用 first-break pick `.npy`。指定する場合は `segy_files` と **同じ長さ**が必要。 |
| `paths.infer_phase_pick_files` | `list[str]` または `str` | No | 推論用 first-break pick `.npy`。指定する場合は `infer_segy_files` と **同じ長さ**が必要。 |

補足:
- `phase_pick_files` が **未指定**の場合、学習・推論とも **fbgate は強制 OFF** になり、内部的にはダミー pick（全トレース=1）が生成される。

### 3.3 FB pick `.npy` の想定
- 1 ファイルにつき **1 次元配列（len = n_traces）** を想定
- pick 値は **サンプル index（int）**
- `pick > 0` が有効、`pick <= 0` は無効扱い

---

## 4. `dataset` セクション

`SegyGatherPipelineDataset` の生成パラメータ。
`dataset` セクション自体は **必須**（`progress` / `waveform_mode` / `*_endian` は optional）。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `dataset.max_trials` | `int` | No | `2048` | サンプル生成のリトライ上限。小さすぎると有効サンプルを引けずエラーになり得る。 |
| `dataset.use_header_cache` | `bool` | No | `true` | SEG-Y header を `*.headers.<endian>.npz` にキャッシュして高速化する。SEG-Y と同階層に sidecar が作られる。 |
| `dataset.verbose` | `bool` | No | `true` | Dataset 内部のログ/情報出力の有無。 |
| `dataset.progress` | `bool` | No | `dataset.verbose` | インデクシング時の tqdm 表示。未指定なら `verbose` と同じ。 |
| `dataset.primary_keys` | `list[str]` | No | `[ffid]` | gather 抽出の主キー（例: `ffid`）。空は禁止、重複禁止。 |
| `dataset.waveform_mode` | `str` | No | `eager` | `eager` / `mmap`。`mmap` はメモリ節約だが `train.num_workers=0` と `infer.num_workers=0` が必須。 |
| `dataset.train_endian` | `str` | No | `big` | 学習用 SEG-Y の読込エンディアン。`big` / `little`。 |
| `dataset.infer_endian` | `str` | No | `big` | 推論用 SEG-Y の読込エンディアン。`big` / `little`。 |

補足:
- 学習データセットは `secondary_key_fixed=False`、推論データセットは **常に** `secondary_key_fixed=True`。

---

## 5. `transform` セクション

`transform.time_len` を用いて **時間方向（W）**を crop/pad する。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `transform.time_len` | `int` | Yes | - | 出力する時間長（W）。学習はランダム crop、推論は中央 crop。`W > 元W` は右側ゼロパディング。 |
| `transform.per_trace_standardize` | `bool` | No | `true` | `PerTraceStandardize(eps=1e-8)` を適用するか。 |

補足:
- H（トレース本数）方向は dataset 側で `subset_traces` に固定され、transform は W のみを操作する想定。

---

## 6. `fbgate` セクション（必須・OFF 可）

FB pick を使ったゲート。`phase_pick_files` が未指定の場合は **強制 OFF**。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `fbgate.apply_on` | `str` | No | `on` | `any` / `super_only` / `off`。`on` は `any` と同義。 |
| `fbgate.min_pick_ratio` | `float` | No | `0.0` | 最低 pick 比率（0.0 相当は実質無効扱い）。 |

補足:
- FBLC のパラメータ（`percentile=95`, `thresh_ms=8`, `min_pairs=16`）は **固定**で YAML から変更できない。

---

## 7. `mask` セクション

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `mask.ratio` | `float` | Yes | - | トレース方向のマスク率（概ね `ratio * H` 本をマスク）。`[0,1]`。 |
| `mask.mode` | `str` | No | `replace` | `replace` / `add`。 |
| `mask.noise_std` | `float` | No | `1.0` | マスク適用時のノイズ標準偏差。`>=0`。 |

補足:
- マスクは **トレース単位**（width=1）で生成される。
- `train.loss_scope` または `eval.loss_scope` が `masked_only` の場合、該当する `subset_traces` について `round(mask.ratio * subset_traces) >= 1` が必須。

---

## 8. `input` セクション

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `input.use_offset_ch` | `bool` | No | `false` | offset チャンネルを入力に追加するか。 |
| `input.offset_normalize` | `bool` | No | `true` | offset チャンネルを valid trace のみで z-score 正規化するか。 |
| `input.use_time_ch` | `bool` | No | `false` | time チャンネルを入力に追加するか。 |

補足:
- 入力チャネル数は `1 + use_offset_ch + use_time_ch`。
- `offset_normalize` は `use_offset_ch=true` のときのみ有効。

---

## 9. `train` セクション（学習ループ + BlindTrace 固有）

`train` セクションは **必須**。

### 9.1 学習ループ系（共通）
| key | 型 | 必須 | 意味 |
|---|---:|:---:|---|
| `train.device` | `str` | No | 学習デバイス指定。`auto` / `cpu` / `cuda` / `cuda:N`。`auto` は CUDA があれば GPU、なければ CPU。 |
| `train.seed` | `int` | No | 学習の乱数 seed（epoch ごとに `seed+epoch` を使用）。 |
| `train.batch_size` | `int` | Yes | 学習 DataLoader の batch size。 |
| `train.gradient_accumulation_steps` | `int` | No | gradient accumulation のステップ数。未指定なら `1`。 |
| `train.num_workers` | `int` | No | 学習 DataLoader worker 数。0 の場合は main process が dataset RNG を直接更新。 |
| `train.use_amp` | `bool` | Yes | AMP を使うか（CUDA 時のみ有効）。 |
| `train.max_norm` | `float` | No | gradient clipping の max norm。 |
| `train.lr` | `float` | Yes | AdamW の学習率。 |
| `train.weight_decay` | `float` | Yes | AdamW の weight decay。 |
| `train.epochs` | `int` | Yes | epoch 数。 |
| `train.samples_per_epoch` | `int` | Yes | 1 epoch あたりに使用するサンプル数（`Subset(ds_train_full, range(samples_per_epoch))`）。 |
| `train.subset_traces` | `int` | Yes | 学習データセットで切り出すトレース本数（H）。 |

### 9.2 BlindTrace 固有
| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `train.losses` | `list[dict]` | Yes | 損失のリスト。`kind`/`weight`/`scope`/`params` を持つ。`kind`: `l1` / `mse` / `huber` / `shift_mse` / `shift_robust_mse` / `fx_mag_mse`。 |
| `train.loss_scope` | `str` | No | `masked_only` / `all`。各 loss で `scope` 未指定時のデフォルト。 |
| `train.loss_kind` | `str` | No | **legacy**。`train.losses` 未指定時のみ有効。`l1` / `mse` / `shift_mse` / `shift_robust_mse`。 |
| `train.shift_max` | `int` | No | **legacy**。`shift_*` のときのみ使用。`0 <= shift_max < W` 推奨。 |
| `train.fx_weight` | `float` | No | **legacy**。`>0` のとき `fx_mag_mse` を追加。 |
| `train.fx_use_log` | `bool` | No | **legacy**。`fx_mag_mse` の対数スケール有無。 |
| `train.fx_eps` | `float` | No | **legacy**。`fx_mag_mse` の epsilon（`>0`）。 |
| `train.fx_f_lo` | `int` | No | **legacy**。`fx_mag_mse` の下限周波数 bin。 |
| `train.fx_f_hi` | `int` / `null` | No | **legacy**。`fx_mag_mse` の上限周波数 bin。 |

補足:
- `train.losses` がある場合は **優先**され、legacy キーは無視される。
- 時間長（W）は `transform.time_len` のみ有効。
- 実効バッチサイズは `train.batch_size × train.gradient_accumulation_steps`。

---

## 10. `eval` セクション（評価用損失）

`eval` セクションは **任意**。未指定の場合、`train` の損失設定が評価にも使われる。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `eval.losses` | `list[dict]` | No | 評価用損失リスト。指定時は評価損失を固定できる。 |
| `eval.loss_scope` | `str` | No | `masked_only` / `all`。`eval.losses` の `scope` 未指定時のデフォルト。 |
| `eval.loss_kind` | `str` | No | **legacy**。`eval.losses` 未指定時のみ有効。 |
| `eval.shift_max` | `int` | No | **legacy**。`shift_*` のときのみ使用。 |
| `eval.fx_weight` | `float` | No | **legacy**。`>0` のとき `fx_mag_mse` を追加。 |
| `eval.fx_use_log` | `bool` | No | **legacy**。 |
| `eval.fx_eps` | `float` | No | **legacy**。 |
| `eval.fx_f_lo` | `int` | No | **legacy**。 |
| `eval.fx_f_hi` | `int` / `null` | No | **legacy**。 |

補足:
- `eval.losses` がある場合は **優先**され、legacy キーは無視される。

---

## 11. `infer` セクション（評価/可視化ループ）

`infer` セクションは **必須**。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `infer.seed` | `int` | No | 推論サンプル固定用 seed。 |
| `infer.batch_size` | `int` | Yes | 推論 DataLoader の batch size。 |
| `infer.num_workers` | `int` | No | **必ず 0**（0 以外はエラー）。推論サンプル固定のため。 |
| `infer.max_batches` | `int` | Yes | 推論で処理する最大 batch 数。推論データセットは `batch_size * max_batches` 件で固定 subset される。 |
| `infer.subset_traces` | `int` | Yes | 推論データセットで切り出すトレース本数（H）。 |

---

## 12. `tile` セクション（H 方向タイル推論）

BlindTrace 推論は `infer_batch_tiled_h` を使って **H 方向タイル**で処理する。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `tile.tile_h` | `int` | Yes | H 方向のタイル高さ。`tile_h <= infer.subset_traces` が必須。 |
| `tile.overlap_h` | `int` | Yes | H 方向のオーバーラップ。`0 <= overlap_h < tile_h`。 |
| `tile.tiles_per_batch` | `int` | Yes | 1バッチあたりのタイル数。`>0`。 |
| `tile.amp` | `bool` | No | タイル推論時に AMP を使うか（CUDA 時のみ有効）。 |
| `tile.use_tqdm` | `bool` | No | タイル推論の進捗表示（tqdm）。 |

補足:
- 入力 `H < tile_h` はエラー（フォールバックなし）。
- `model.out_chans` は **1 固定**（タイル推論のバリデーションで強制）。

---

## 13. `vis` セクション（triptych 可視化）

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

---

## 14. `model` セクション

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `model.backbone` | `str` | Yes | 例: `resnet18`。`EncDec2D` の backbone 名。 |
| `model.pretrained` | `bool` | No | backbone の pretrained を使うか。未指定なら `false`。 |
| `model.in_chans` | `int` | No | `input` から算出される値と一致必須（不一致はエラー）。 |
| `model.out_chans` | `int` | No | `1` 固定。指定した場合は `1` と一致必須。 |
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

補足:
- 入力チャネル数は `input` セクションから算出され、`out_chans=1` が固定。`model.in_chans` / `model.out_chans` を指定する場合は一致必須。

---

## 14. 実行（例）

```bash
python examples/examples_train_blindtrace.py --config examples/config_train_blindtrace.yaml
```

出力先は `paths.out_dir`。

---

## 15. よくある落とし穴（チェックリスト）

- `phase_pick_files` と `segy_files` の長さ不一致
- FB pick `.npy` が `n_traces` と一致しない
- `mask.ratio` が小さすぎて `masked_only` でエラー（`round(ratio*subset_traces) < 1`）
- `infer.num_workers` を 0 以外にしている（即エラー）
- `dataset.waveform_mode="mmap"` で `train.num_workers` / `infer.num_workers` が 0 以外（即エラー）
- `tile.tile_h > infer.subset_traces` または `overlap_h >= tile_h`
- `model.out_chans!=1`（タイル推論で即エラー）
