# PSN タスク: YAML 設定マニュアル（examples/config_train_psn.yaml）

本メモは `examples/config_train_psn.yaml`（PSN 学習パイプライン）で使用される YAML 設定の **キー・型・制約・実挙動**を、コード実装に合わせて整理したもの。
対象エントリポイントは `examples/example_train_psn.py` → `seisai_engine.pipelines.psn.train.main`。

共通事項（パス解決/listfile、augment、scheduler、ckpt、ema、tracking など）は `docs/examples/common_yaml_manual.md` を参照。

---

## 1. 目的と前提

PSN（P/S/Noise）タスクは、CSR 形式の位相ピック（P/S）から **(3, H, W) の確率ターゲットマップ**を生成し、2D Encoder-Decoder（`EncDec2D`）で **(P,S,Noise)** を画素単位（トレース×時間）に推定する。

- 入力: SEG-Y gather window（`(H,W)` を `(1,H,W)` としてモデル入力へ）
- 教師: `PhasePSNMap` により `(3,H,W)` の soft-label（P/S/Noise の確率分布）
- 損失: soft-label CE（`trace_valid` と `label_valid` 等で画素マスク）

---

## 2. 参照 YAML（examples/config_train_psn.yaml の現物）

```yaml
paths:
  segy_files: [/workspace/tests/data/20200623002546.sgy]
  phase_pick_files: [/workspace/tests/data/20200623002546_phase_picks.npz]
  infer_segy_files: [/workspace/tests/data/20200623002546.sgy]
  infer_phase_pick_files: [/workspace/tests/data/20200623002546_phase_picks.npz]
  out_dir: ./_psn_out

dataset:
  max_trials: 2048
  use_header_cache: true
  verbose: true
  primary_keys: [ffid]
  include_empty_gathers: false
  secondary_key_fixed: false
  train_endian: big
  infer_endian: big

train:
  device: auto # auto | cpu | cuda | cuda:N
  batch_size: 4
  epochs: 10
  lr: 1.0e-4
  subset_traces: 128
  time_len: 6016
  samples_per_epoch: 256
  psn_sigma: 1.5
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

transform:
  target_len: 6016
  standardize_eps: 1.0e-8

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

vis:
  out_subdir: vis
  n: 1

infer:
  batch_size: 1
  max_batches: 4
  subset_traces: 128
  seed: 43
  num_workers: 0

ckpt:
  save_best_only: true
  metric: infer_loss
  mode: min

model:
  backbone: resnet18
  # pretrained: false # default
  in_chans: 1
  out_chans: 3
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
| `paths.segy_files` | `list[str]` または `str` | Yes | 学習用 SEG-Y ファイル群。`phase_pick_files` と **同じ長さ**が必要。空リストは不可。 |
| `paths.phase_pick_files` | `list[str]` または `str` | Yes | 学習用 CSR 位相ピック `.npz` 群。SEG-Y と 1:1 対応。空リストは不可。 |
| `paths.infer_segy_files` | `list[str]` または `str` | Yes | 推論（評価/可視化）用 SEG-Y ファイル群。空リストは不可。 |
| `paths.infer_phase_pick_files` | `list[str]` または `str` | Yes | 推論用 CSR 位相ピック `.npz` 群。`infer_segy_files` と **同じ長さ**が必要。空リストは不可。 |
| `paths.out_dir` | `str` | Yes | 出力先ディレクトリ。相対指定は YAML の場所基準で解決。 |

### 3.2 整合性チェック
- `len(segy_files) == len(phase_pick_files)` を要求
- `len(infer_segy_files) == len(infer_phase_pick_files)` を要求
- 全ファイルの存在をチェック（存在しないと即エラー）

---

## 4. 位相ピック（CSR `.npz`）入力仕様（要点）

`phase_pick_files` / `infer_phase_pick_files` は CSR 形式の `.npz`。

必須キー:
- `p_indptr: int, shape (n_traces+1,)`
- `p_data: int, shape (nnz_p,)`
- `s_indptr: int, shape (n_traces+1,)`
- `s_data: int, shape (nnz_s,)`

制約:
- `n_traces` は対応する SEG-Y のトレース数と一致する必要がある（不一致は即エラー）
- pick 値は raw time 軸のサンプル index（整数）
- `pick <= 0` は invalid
- `pick > 0` は valid

---

## 5. `dataset` セクション

`SegyGatherPhasePipelineDataset` の生成パラメータの一部。
`dataset` セクション自体は **必須**（ただし多くのキーは optional でデフォルト有り）。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `dataset.max_trials` | `int` | No | `2048` | サンプル生成のリトライ上限。小さすぎると「有効サンプルを引けず」エラーになり得る。 |
| `dataset.use_header_cache` | `bool` | No | `true` | SEG-Y header を `*.headers.<endian>.npz` にキャッシュして高速化する。SEG-Y と同階層に sidecar が作られる（更新日時で再利用）。 |
| `dataset.verbose` | `bool` | No | `true` | Dataset 内部のログ/情報出力の有無。 |
| `dataset.progress` | `bool` | No | `dataset.verbose` | インデクシング時の tqdm 表示。未指定なら `verbose` と同じ。 |
| `dataset.primary_keys` | `list[str]` | No | `[ffid]` | gather 抽出の主キー（例: `ffid`）。空は禁止、重複禁止。 |
| `dataset.include_empty_gathers` | `bool` | No | `false` | P/S ピックが両方とも存在しないサンプルを許容するか。`false` の場合は空サンプルを reject してリサンプル。 |
| `dataset.secondary_key_fixed` | `bool` | No | `false` | 2次整列（secondary key）ルールを固定するか。 **学習側のみ**この値が反映される（推論側は常に固定）。 |
| `dataset.waveform_mode` | `str` | No | `eager` | `eager` / `mmap`。`mmap` はメモリ節約だが `train.num_workers=0` かつ `infer.num_workers=0` が必須。 |
| `dataset.train_endian` | `str` | No | `big` | 学習用 SEG-Y の読込エンディアン。`big` / `little`。 |
| `dataset.infer_endian` | `str` | No | `big` | 推論用 SEG-Y の読込エンディアン。`big` / `little`。 |

### 5.1 `primary_keys` と `secondary_key_fixed` の補足
- `primary_keys` はサンプル抽出時の gather 単位（例: `ffid`）を決める。
- `secondary_key_fixed=false` の場合、`primary_keys=['ffid']` なら secondary が `chno` と `offset` からランダムに選ばれ得る（多様性付与）。
- 推論データセットは **必ず** `secondary_key_fixed=True` として生成される（推論可視化の固定化のため）。

### 5.2 `include_empty_gathers` の注意
- `include_empty_gathers=true` かつ空サンプルの場合、`label_valid` が全 False になり、損失は画素マスクにより **0** になり得る。
- 学習の安定性・収束に影響する可能性があるため、通常は `false` 推奨。

### 5.3 `waveform_mode` の注意
- `eager`: すべての trace をメモリに読み込み。
- `mmap`: segyio mmap を使用してオンデマンド読み込み（メモリ節約）。
- `mmap` の場合は `train.num_workers=0` と `infer.num_workers=0` が必須（0 以外はエラー）。

---

## 6. `transform` セクション

PSN は `transform.target_len` を用いて **時間方向（W）**を crop/pad する。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `transform.target_len` | `int` | Yes | - | 出力する時間長（W）。`W > 元W` は右側ゼロパディング、`W < 元W` は crop。学習はランダム crop、推論は中央 crop。 |
| `transform.standardize_eps` | `float` | No | `1.0e-8` | Per-trace 標準化の分母安定化 epsilon。 |

補足:
- H（トレース本数）方向は dataset 側で `subset_traces` に固定され、transform は W のみを操作する想定。

---

## 7. `train` セクション（学習ループ + PSN 固有）

`train` セクションは **必須**。

### 7.1 学習ループ系（共通）
| key | 型 | 必須 | 意味 |
|---|---:|:---:|---|
| `train.device` | `str` | No | 学習デバイス指定。`auto` / `cpu` / `cuda` / `cuda:N`。`auto` は CUDA があれば GPU、なければ CPU。 |
| `train.batch_size` | `int` | Yes | 学習 DataLoader の batch size。 |
| `train.gradient_accumulation_steps` | `int` | No | gradient accumulation のステップ数。未指定なら `1`。 |
| `train.epochs` | `int` | Yes | epoch 数。 |
| `train.samples_per_epoch` | `int` | Yes | 1 epoch あたりに使用するサンプル数（`Subset(ds_train_full, range(samples_per_epoch))`）。 |
| `train.seed` | `int` | Yes | 学習の乱数 seed（epoch ごとに `seed+epoch` を使用）。 |
| `train.use_amp` | `bool` | Yes | AMP を使うか（CUDA 時のみ有効）。 |
| `train.max_norm` | `float` | Yes | gradient clipping の max norm。 |
| `train.num_workers` | `int` | Yes | 学習 DataLoader worker 数。0 の場合は main process が dataset RNG を直接更新。 |
| `train.print_freq` | `int` | No | 学習中のログ出力頻度（step 間隔）。未指定時は `10`。 |

### 7.2 PSN 固有
| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `train.lr` | `float` | Yes | AdamW の学習率。 |
| `train.subset_traces` | `int` | Yes | 学習データセットで切り出すトレース本数（H）。 |
| `train.psn_sigma` | `float` | No | `1.5` | `PhasePSNMap` のガウシアン sigma（サンプルbin単位）。大きいほどラベルが太る。 |

### 7.3 `train.time_len` について（重要）
`examples/config_train_psn.yaml` には `train.time_len` が存在するが、PSN パイプライン実装では参照されない。
- PSN で時間長（W）を決めるのは `transform.target_len`
- `train.time_len` は現状 **無効パラメータ**（残置の可能性が高い）
- 実効バッチサイズは `train.batch_size × train.gradient_accumulation_steps`。

---

## 8. `infer` セクション（評価/可視化ループ + PSN 固有）

`infer` セクションは **必須**。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `infer.batch_size` | `int` | Yes | 推論 DataLoader の batch size。 |
| `infer.max_batches` | `int` | Yes | 推論で処理する最大 batch 数。推論データセットは `batch_size * max_batches` 件で固定 subset される。 |
| `infer.seed` | `int` | Yes | 各 epoch の推論前に dataset RNG をこの seed で再初期化し、推論サンプルを固定化する。 |
| `infer.num_workers` | `int` | Yes | **必ず 0**（0 以外はエラー）。推論サンプル固定のため。 |
| `infer.subset_traces` | `int` | Yes | 推論データセットで切り出すトレース本数（H）。 |

---

## 9. `vis` セクション

| key | 型 | 必須 | 意味 |
|---|---:|:---:|---|
| `vis.out_subdir` | `str` | Yes | `paths.out_dir` 配下の可視化出力サブディレクトリ名。 |
| `vis.n` | `int` | Yes | 推論ループの先頭 `n` batch について `step_####.png` を保存。 |

出力パス規約:
- `out_dir/<vis.out_subdir>/epoch_####/step_####.png`

---

## 10. `model` セクション

PSN は `EncDec2D` を使用し、チャネル数に強い制約がある。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `model.backbone` | `str` | Yes | 例: `resnet18`（`EncDec2D` の backbone 名）。 |
| `model.pretrained` | `bool` | No | backbone の pretrained を使うか。未指定なら `false`。 |
| `model.in_chans` | `int` | Yes | **必ず 1**（waveform のみ）。1 以外はエラー。 |
| `model.out_chans` | `int` | Yes | **必ず 3**（P/S/Noise）。3 以外はエラー。 |
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

## 11. オプション: `fbgate` セクション（未指定時は OFF）

PSN dataset は `cfg.fbgate` を読む（未指定なら gating 無効）。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `fbgate.apply_on` | `str` | No | `off` | `any` / `super_only` / `off`（`on` は `any` と同義）。 |
| `fbgate.min_pick_ratio` | `float` | No | `0.0` | 最低 pick 比率（0.0 相当は実質無効扱い）。 |
| `fbgate.verbose` | `bool` | No | `false` | gate のログ出力。 |

---

## 12. 実行（例）

```bash
python examples/example_train_psn.py --config examples/config_train_psn.yaml
```

出力先は `paths.out_dir`。

---

## 13. よくある落とし穴（チェックリスト）

- `paths.*_files` の長さ不一致（segy と pick）がある
- pick `.npz` の `n_traces` が SEG-Y と一致しない
- `infer.num_workers` を 0 以外にしている（即エラー）
- `dataset.waveform_mode="mmap"` で `train.num_workers` / `infer.num_workers` が 0 以外（即エラー）
- `model.in_chans!=1` / `model.out_chans!=3`（即エラー）
- `include_empty_gathers=true` で有効画素が極端に減り、学習が進まない
