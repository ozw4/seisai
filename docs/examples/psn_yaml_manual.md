# PSN タスク: YAML 設定マニュアル（examples/config_train_psn.yaml）

本メモは `examples/config_train_psn.yaml`（PSN 学習パイプライン）で使用される YAML 設定の **キー・型・制約・実挙動**を、コード実装に合わせて整理したもの。
対象エントリポイントは `examples/example_train_psn.py` → `seisai_engine.pipelines.psn.train.main`。

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

train:
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

transform:
  target_len: 6016
  standardize_eps: 1.0e-8

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
  pretrained: false
  in_chans: 1
  out_chans: 3

tracking:
  enabled: true
  exp_name: baseline
  tracking_uri: file:./mlruns
  vis_best_only: true
  vis_max_files: 50
```

---

## 3. パス解決・listfile 展開ルール

### 3.1 相対パスの基準
以下のキーは、YAML ファイルのあるディレクトリ（`base_dir`）基準で絶対パスへ解決される。

- `paths.segy_files`
- `paths.phase_pick_files`
- `paths.infer_segy_files`
- `paths.infer_phase_pick_files`
- `paths.out_dir`（出力ディレクトリ）

また `tracking.tracking_uri` が `file:` URI の場合、`file:./mlruns` のような相対指定は `base_dir` 基準で `file:/abs/path` に解決される。

### 3.2 listfile（1行1パス）
`paths.*_files` は **list[str]** だけでなく **listfile へのパス（str）** として指定できる。

- listfile の中身は「1行1パス」
- 空行は無視
- `#` で始まる行はコメントとして無視
- 環境変数展開（`$VAR`）と `~` 展開を行う
- listfile 内の相対パスは **listfile 自体のディレクトリ**基準で解決
- 展開後に全ファイルの存在チェックを行う（存在しないと即エラー）

---

## 4. `paths` セクション

### 4.1 必須キー
| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `paths.segy_files` | `list[str]` または `str` | Yes | 学習用 SEG-Y ファイル群。`phase_pick_files` と **同じ長さ**が必要。 |
| `paths.phase_pick_files` | `list[str]` または `str` | Yes | 学習用 CSR 位相ピック `.npz` 群。SEG-Y と 1:1 対応。 |
| `paths.infer_segy_files` | `list[str]` または `str` | Yes | 推論（評価/可視化）用 SEG-Y ファイル群。 |
| `paths.infer_phase_pick_files` | `list[str]` または `str` | Yes | 推論用 CSR 位相ピック `.npz` 群。`infer_segy_files` と **同じ長さ**が必要。 |
| `paths.out_dir` | `str` | Yes | 出力先ディレクトリ。相対指定は YAML の場所基準で解決。 |

### 4.2 整合性チェック
- `len(segy_files) == len(phase_pick_files)` を要求
- `len(infer_segy_files) == len(infer_phase_pick_files)` を要求
- 全ファイルの存在をチェック（存在しないと即エラー）

---

## 5. 位相ピック（CSR `.npz`）入力仕様（要点）

`phase_pick_files` / `infer_phase_pick_files` は CSR 形式の `.npz`。

必須キー:
- `p_indptr: int, shape (n_traces+1,)`
- `p_data: int, shape (nnz_p,)`
- `s_indptr: int, shape (n_traces+1,)`
- `s_data: int, shape (nnz_s,)`

制約:
- `n_traces` は対応する SEG-Y のトレース数と一致する必要がある（不一致は即エラー）
- pick 値は raw time 軸の **サンプル index**（整数）
  - `pick <= 0` は invalid
  - `pick > 0` は valid

---

## 6. `dataset` セクション

`SegyGatherPhasePipelineDataset` の生成パラメータの一部。
`dataset` セクション自体は **必須**（ただし多くのキーは optional でデフォルト有り）。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `dataset.max_trials` | `int` | No | `2048` | サンプル生成のリトライ上限。小さすぎると「有効サンプルを引けず」エラーになり得る。 |
| `dataset.use_header_cache` | `bool` | No | `true` | SEG-Y header を `*.headers.npz` にキャッシュして高速化する。SEG-Y と同階層に sidecar が作られる（更新日時で再利用）。 |
| `dataset.verbose` | `bool` | No | `true` | Dataset 内部のログ/情報出力の有無。 |
| `dataset.primary_keys` | `list[str]` | No | `[ffid]` | gather 抽出の主キー（例: `ffid`）。空は禁止、重複禁止。 |
| `dataset.include_empty_gathers` | `bool` | No | `false` | P/S ピックが両方とも存在しないサンプルを許容するか。`false` の場合は空サンプルを reject してリサンプル。 |
| `dataset.secondary_key_fixed` | `bool` | No | `false` | 2次整列（secondary key）ルールを固定するか。 **学習側のみ**この値が反映される（推論側は常に固定）。 |

### 6.1 `primary_keys` と `secondary_key_fixed` の補足
- `primary_keys` はサンプル抽出時の gather 単位（例: `ffid`）を決める。
- `secondary_key_fixed=false` の場合、`primary_keys=['ffid']` なら secondary が `chno` と `offset` からランダムに選ばれ得る（多様性付与）。
- 推論データセットは **必ず** `secondary_key_fixed=True` として生成される（推論可視化の固定化のため）。

### 6.2 `include_empty_gathers` の注意
- `include_empty_gathers=true` かつ空サンプルの場合、`label_valid` が全 False になり、損失は画素マスクにより **0** になり得る。
  - 学習の安定性・収束に影響する可能性があるため、通常は `false` 推奨。

---

## 7. `transform` セクション

PSN は `transform.target_len` を用いて **時間方向（W）**を crop/pad する。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `transform.target_len` | `int` | Yes | - | 出力する時間長（W）。`W > 元W` は右側ゼロパディング、`W < 元W` は crop。学習はランダム crop、推論は中央 crop。 |
| `transform.standardize_eps` | `float` | No | `1.0e-8` | Per-trace 標準化の分母安定化 epsilon。 |

補足:
- H（トレース本数）方向は dataset 側で `subset_traces` に固定され、transform は W のみを操作する想定。

---

## 8. `train` セクション（学習ループ + PSN 固有）

`train` セクションは **必須**。

### 8.1 学習ループ系（共通）
| key | 型 | 必須 | 意味 |
|---|---:|:---:|---|
| `train.batch_size` | `int` | Yes | 学習 DataLoader の batch size。 |
| `train.epochs` | `int` | Yes | epoch 数。 |
| `train.samples_per_epoch` | `int` | Yes | 1 epoch あたりに使用するサンプル数（`Subset(ds_train_full, range(samples_per_epoch))`）。 |
| `train.seed` | `int` | Yes | 学習の乱数 seed（epoch ごとに `seed+epoch` を使用）。 |
| `train.use_amp` | `bool` | Yes | AMP を使うか（主に GPU 時）。 |
| `train.max_norm` | `float` | Yes | gradient clipping の max norm。 |
| `train.num_workers` | `int` | Yes | 学習 DataLoader worker 数。0 の場合は main process が dataset RNG を直接更新。 |
| `train.print_freq` | `int` | No | 学習中のログ出力頻度（step 間隔）。未指定時は `10`。 |

### 8.2 PSN 固有
| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `train.lr` | `float` | Yes | AdamW の学習率。 |
| `train.subset_traces` | `int` | Yes | 学習データセットで切り出すトレース本数（H）。 |
| `train.psn_sigma` | `float` | No | `1.5` | `PhasePSNMap` のガウシアン sigma（サンプルbin単位）。大きいほどラベルが太る。 |

### 8.3 `train.time_len` について（重要）
`examples/config_train_psn.yaml` には `train.time_len` が存在するが、PSN パイプライン実装では参照されない。
- PSN で時間長（W）を決めるのは `transform.target_len`
- `train.time_len` は現状 **無効パラメータ**（残置の可能性が高い）

---

## 9. `infer` セクション（評価/可視化ループ + PSN 固有）

`infer` セクションは **必須**。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `infer.batch_size` | `int` | Yes | 推論 DataLoader の batch size。 |
| `infer.max_batches` | `int` | Yes | 推論で処理する最大 batch 数。推論データセットは `batch_size * max_batches` 件で固定 subset される。 |
| `infer.seed` | `int` | Yes | 各 epoch の推論前に dataset RNG をこの seed で再初期化し、推論サンプルを固定化する。 |
| `infer.num_workers` | `int` | Yes | **必ず 0**（0 以外はエラー）。推論サンプル固定のため。 |
| `infer.subset_traces` | `int` | Yes | 推論データセットで切り出すトレース本数（H）。 |

---

## 10. `vis` セクション

| key | 型 | 必須 | 意味 |
|---|---:|:---:|---|
| `vis.out_subdir` | `str` | Yes | `paths.out_dir` 配下の可視化出力サブディレクトリ名。 |
| `vis.n` | `int` | Yes | 推論ループの先頭 `n` batch について `step_####.png` を保存。 |

出力パス規約:
- `out_dir/<vis.out_subdir>/epoch_####/step_####.png`

---

## 11. `model` セクション

PSN は `EncDec2D` を使用し、チャネル数に強い制約がある。

| key | 型 | 必須 | 意味 / 制約 |
|---|---:|:---:|---|
| `model.backbone` | `str` | Yes | 例: `resnet18`（`EncDec2D` の backbone 名）。 |
| `model.pretrained` | `bool` | Yes | backbone の pretrained を使うか。 |
| `model.in_chans` | `int` | Yes | **必ず 1**（waveform のみ）。1 以外はエラー。 |
| `model.out_chans` | `int` | Yes | **必ず 3**（P/S/Noise）。3 以外はエラー。 |

---

## 12. `ckpt` セクション（best-only 固定）

PSN の学習スクリプトは ckpt 設定に強い制約を課す。

| key | 型 | 必須 | 制約 |
|---|---:|:---:|---|
| `ckpt.save_best_only` | `bool` | Yes | **true 固定**（false はエラー）。 |
| `ckpt.metric` | `str` | Yes | **`infer_loss` 固定**（他はエラー）。 |
| `ckpt.mode` | `str` | Yes | **`min` 固定**（他はエラー）。 |

出力:
- `out_dir/ckpt/best.pt`

---

## 13. `tracking` セクション（任意）

`tracking` は未指定でも動作する（デフォルト無効）。指定した場合は MLflow 互換トラッキングが有効化される。

| key | 型 | 必須 | デフォルト | 意味 |
|---|---:|:---:|---:|---|
| `tracking.enabled` | `bool` | No | `false` | トラッキング有効化。 |
| `tracking.experiment_prefix` | `str` | No | `seisai` | experiment 名の prefix（`<prefix>/<pipeline>`）。 |
| `tracking.exp_name` | `str` | No | `baseline` | run 名に埋め込まれる識別子。 |
| `tracking.tracking_uri` | `str` | No | `file:./mlruns` | MLflow tracking URI。`file:` の相対パスは YAML 位置基準で絶対 `file:` に解決。 |
| `tracking.vis_best_only` | `bool` | No | `true` | best 更新時のみ vis をログ対象にする（実装では best 時のみ log_best）。 |
| `tracking.vis_max_files` | `int` | No | `50` | best 可視化のログ対象ファイル上限（超過分は out_dir には残るが tracking へは送らない）。 |

トラッキング有効時、`out_dir/tracking/` に以下が生成される:
- `config.resolved.yaml`
- `git.txt`, `env.txt`
- `data_manifest.json`
- `optimizer_groups.json`
- （必要に応じて）`overlong_values.json`

---

## 14. オプション: `fbgate` セクション（未指定時は OFF）

PSN dataset は `cfg.fbgate` を読む（未指定なら gating 無効）。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `fbgate.apply_on` | `str` | No | `off` | `any` / `super_only` / `off`（`on` は `any` と同義）。 |
| `fbgate.min_pick_ratio` | `float` | No | `0.0` | 最低 pick 比率（0.0 相当は実質無効扱い）。 |
| `fbgate.verbose` | `bool` | No | `false` | gate のログ出力。 |

---

## 15. 実行（例）

```bash
python examples/example_train_psn.py --config examples/config_train_psn.yaml
```

出力先は `paths.out_dir`。

---

## 16. よくある落とし穴（チェックリスト）

- `paths.*_files` の長さ不一致（segy と pick）がある
- pick `.npz` の `n_traces` が SEG-Y と一致しない
- `infer.num_workers` を 0 以外にしている（即エラー）
- `ckpt.*` が固定値から外れている（即エラー）
- `model.in_chans!=1` / `model.out_chans!=3`（即エラー）
- `include_empty_gathers=true` で有効画素が極端に減り、学習が進まない

