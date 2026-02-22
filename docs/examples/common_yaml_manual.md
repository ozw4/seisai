# YAML 共通説明（PSN / Pair / BlindTrace）

このドキュメントは `docs/examples/psn_yaml_manual.md` / `docs/examples/pair_yaml_manual.md` / `docs/examples/blindtrace_yaml_manual.md` に共通する **ルール・構造・制約** をまとめたものです。
各タスク固有のキー（例: Pair の `paths.input_segy_files`、PSN の `train.psn_sigma`、BlindTrace の `mask` など）は各タスクのマニュアル側に残しています。

---

## 1. パス解決・listfile 展開ルール

### 1.1 相対パスの基準（`base_dir`）
以下のキーは、YAML ファイルのあるディレクトリ（`base_dir`）基準で絶対パスへ解決されます。

- `paths.*_files`（キー名はタスクごと）
- `paths.out_dir`
- `train.init_ckpt`（指定時）

また `tracking.tracking_uri` が `file:` URI の場合、`file:./mlruns` のような相対指定は `base_dir` 基準で `file:/abs/path` に解決されます。

### 1.2 listfile（1行1パス）
`paths.*_files` は **list[str]** だけでなく **listfile へのパス（str）** として指定できます。

- listfile の中身は「1行1パス」
- 空行は無視
- `#` で始まる行はコメントとして無視
- 環境変数展開（`$VAR`）と `~` 展開を行う
- listfile 内の相対パスは **listfile 自体のディレクトリ**基準で解決
- 展開後に全ファイルの存在チェックを行う（存在しないと即エラー）
- listfile が空の場合はエラー

### 入力例A: listfile（`str` で指定）

(1) listfile の中身例（例: data/train_segy_list.txt）

#### 1行1パス。空行は無視。
```
/data/segy/train_0001.sgy
/data/segy/train_0002.sgy
```
#### listfileからの相対パス（このファイルのあるディレクトリ基準）
```
../segy/train_0003.sgy
```
#### ~ や環境変数もOK
```
~/datasets/segy/train_0004.sgy
$SEGY_ROOT/train_0005.sgy
```
#### (2) YAML 側の指定例
listfileへのパスを “文字列” で渡す
```
paths:
  segy_files: data/train_segy_list.txt
```
補足:
- data/train_segy_list.txt が相対パスなら、YAML のあるディレクトリ（base_dir）基準で解決されます。
- listfile 内の ../segy/... のような相対パスは、listfile 自体のディレクトリ基準で解決されます。

### 入力例B: list[str]（YAML に直接列挙）

```
paths:
  segy_files:
    - /data/segy/train_0001.sgy
    - /data/segy/train_0002.sgy
    - ../segy/train_0003.sgy     # これは YAML（base_dir）基準の相対パス
    - ~/datasets/segy/train_0004.sgy
    - $SEGY_ROOT/train_0005.sgy
```

## 2. `dataset` セクション

タスク固有の追加キーは各タスクマニュアルを参照しつつ、以下は共通で頻出です。

| key | 型 | 典型 default | 意味 / 注意 |
|---|---:|---:|---|
| `dataset.max_trials` | `int` | `2048` | サンプル生成に失敗した時の再試行上限。 |
| `dataset.use_header_cache` | `bool` | `true` | SEG-Y header を sidecar（`*.headers.<endian>.npz`）としてキャッシュ。 |
| `dataset.verbose` | `bool` | `true` | Dataset 内部ログ。 |
| `dataset.progress` | `bool` | `verbose` | tqdm 表示（未指定なら verbose に追従）。 |
| `dataset.primary_keys` | `list[str]` | `["ffid"]` など | gather 抽出の主キー。空/重複は不可。 |
| `dataset.secondary_key_fixed` | `bool` | `false` など | secondary key を固定するか（多くは「学習側のみ反映・推論側は固定」）。 |
| `dataset.waveform_mode` | `str` | `"eager"` | `"eager"` / `"mmap"`。`mmap` は **worker=0 制約**あり。 |

### 2.1 `waveform_mode: "mmap"` の制約
`dataset.waveform_mode: "mmap"` の場合は **I/O を安全にするため**、

- `train.num_workers = 0`
- `infer.num_workers = 0`

が必須です（違反は即エラー）。

---

## 3. `transform` セクション

### 3.1 `transform.time_len`
多くのタスクでは時間方向（W）を `time_len` へ揃えます。

- 学習: `RandomCropOrPad(target_len=time_len)`
- 推論: `DeterministicCropOrPad(target_len=time_len)`

> 例外: Pair は **推論側で時間 crop/pad を行わない**（Pair 側マニュアルを参照）。

### 3.2 標準化（Per-trace）
標準化の有無・eps の指定方法はタスクごとに異なります。
- PSN: `transform.standardize_eps`（PerTraceStandardize を必ず適用）
- BlindTrace: `transform.per_trace_standardize`（on/off）
- Pair: Dataset 側で input から z-score を計算して input/target 両方へ適用（現状は常に on）

---

## 4. `augment` セクション（任意・学習のみ）

`augment` は学習側の transform にのみ適用され、未指定なら全ての確率が 0 で実質無効。
適用順序は `hflip/space/time` → `RandomCropOrPad` → `freq/polarity` → `PerTraceStandardize`（有効時）。

### 4.1 共通
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `augment.hflip_prob` | `float` | No | `0.0` | 水平方向（トレース順）反転の確率。`[0,1]`。 |
| `augment.polarity_prob` | `float` | No | `0.0` | 極性反転の確率。`[0,1]`。 |

### 4.2 `augment.space`（空間伸縮）
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `augment.space.prob` | `float` | No | `0.0` | 空間方向の伸縮を適用する確率。`[0,1]`。 |
| `augment.space.factor_range` | `list[float]` | No | `[0.90, 1.10]` | 伸縮率の範囲。両方 `>0` かつ `min <= max`。 |

### 4.3 `augment.time`（時間伸縮）
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `augment.time.prob` | `float` | No | `0.0` | 時間方向の伸縮を適用する確率。`[0,1]`。 |
| `augment.time.factor_range` | `list[float]` | No | `[0.95, 1.05]` | 伸縮率の範囲。両方 `>0` かつ `min <= max`。 |

### 4.4 `augment.freq`（周波数フィルタ）
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `augment.freq.prob` | `float` | No | `0.0` | 周波数フィルタを適用する確率。`[0,1]`。 |
| `augment.freq.kinds` | `list[str]` | No | `[bandpass, lowpass, highpass]` | フィルタ種類。空は不可。許可値は `bandpass` / `lowpass` / `highpass`。 |
| `augment.freq.band` | `list[float]` | No | `[0.05, 0.45]` | バンド中心範囲。`[0,1]` 内で `min <= max`。 |
| `augment.freq.width` | `list[float]` | No | `[0.10, 0.35]` | バンド幅範囲。`[0,1]` 内で `min <= max`。 |
| `augment.freq.roll` | `float` | No | `0.02` | ロールオフ。`[0,1]`。 |
| `augment.freq.restandardize` | `bool` | No | `false` | フィルタ後に再標準化するか。 |

---

## 5. `train` / `eval` / `infer`

### 5.1 `train`（学習ループ）
頻出キー（タスク固有の追加キーは別途）:

| key | 型 | 意味 |
|---|---:|---|
| `train.device` | `str` | `auto` / `cpu` / `cuda` / `cuda:N` |
| `train.batch_size` | `int` | 学習 batch size |
| `train.gradient_accumulation_steps` | `int` | 未指定なら 1 |
| `train.epochs` | `int` | epoch 数 |
| `train.lr` | `float` | 学習率 |
| `train.samples_per_epoch` | `int` | 1 epoch に使うサンプル数（Subset の長さ） |
| `train.subset_traces` | `int` | gather window の H（トレース）方向の幅 |
| `train.seed` | `int` | 学習乱数 seed |
| `train.use_amp` | `bool` | AMP を使うか |
| `train.max_norm` | `float` | 勾配クリップ |
| `train.num_workers` | `int` | DataLoader worker 数（`mmap` なら 0 必須） |
| `train.print_freq` | `int` | 学習ログ出力頻度（省略時 10） |

### 5.2 `train.init_ckpt`（任意）
全タスク共通で `train.init_ckpt` を使えます。

- 型: `str` または `null`
- 相対パスは `base_dir` 基準で解決
- 互換性チェック:
  - `in_chans` が一致しないと即エラー
  - `out_chans` が不一致でも、モデルに `seg_head` がある場合は **seg_head だけ除外して読み込み**（backbone/decoder は読み込み、head は初期化扱い）

### 5.3 loss の指定方法（`train.losses` / `eval.losses`）
loss は次の形式のリストで指定します。

```yaml
train:
  loss_scope: all
  losses:
    - kind: l1
      weight: 1.0
      scope: all
      params: {}
```

- `loss_scope`: デフォルトの適用範囲（例: `all` / `masked_only`）
- `losses[*].scope`: 省略時は `loss_scope` が使われます
- `losses[*].weight`: 加重和の係数
- `losses[*].params`: loss ごとのパラメータ（必要なものだけ）

> 許可される `kind` はパイプライン側の実装に依存します。各タスクのサンプル YAML で使用されている `kind` を基準に選ぶのが安全です。

### 5.4 `infer`（推論/評価ループ）
| key | 型 | 意味 |
|---|---:|---|
| `infer.batch_size` | `int` | 推論 batch size（多くは 1） |
| `infer.max_batches` | `int` | 推論で回す batch 上限 |
| `infer.subset_traces` | `int` | 推論時の gather window H 幅 |
| `infer.seed` | `int` | 推論サンプル固定用 seed |
| `infer.num_workers` | `int` | **必ず 0**（固定サンプル維持のため、違反は即エラー） |

---

## 5.5 `optimizer` セクション（任意）

optimizer を指定しない場合は `torch.optim.AdamW`** が使われます。
`optimizer` セクションを指定した場合は **timm の optimizer factory（`timm.optim.create_optimizer_v2`）** 経由で生成され、`lion` などに差し替えできます。

### 5.5.1 基本ルール
- 学習率は **常に `train.lr`** を使用します。
- weight decay は **`train.weight_decay`** を使用します（タスクによって必須/任意が異なる場合あり）。
- `optimizer.kwargs` に **`lr` / `weight_decay` は書けません**（書くとエラー）。
  → それらは `train.*` 側で統一管理します。

> 補足: `filter_bias_and_bn` は timm 経由のときのみ有効です。
> 「bias / BN を weight decay から除外したい」場合は、`optimizer` セクションを明示して timm 経由にしてください（`name: adamw` でもOK）。

### 5.5.2 キー一覧
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `optimizer.name` | `str` | No | `adamw` | optimizer 名（timm が認識できる文字列）。例: `adamw`, `lion`, `sgd` など。大小文字は無視。 |
| `optimizer.filter_bias_and_bn` | `bool` | No | `false` | `true` の場合、bias / BN などを weight decay 対象から除外する param group を作る（timm 経由のみ）。 |
| `optimizer.kwargs` | `dict` | No | `{}` | optimizer 固有引数。`lr` / `weight_decay` は禁止。`betas` は YAML では list になりやすいので `[0.9, 0.99]` のように書く。 |

### 5.5.3 例

#### 例A: Lion（timm の実装を使用）
```yaml
optimizer:
  name: lion
  filter_bias_and_bn: false
  kwargs:
    betas: [0.9, 0.99]
例B: timm 経由で AdamW を使い、bias/Bn を weight decay から除外する
train:
  lr: 3.0e-4
  weight_decay: 1.0e-2

optimizer:
  name: adamw
  filter_bias_and_bn: true
  kwargs: {}
```
### 5.x.4 差し替え可能な optimizer の要件

  optimizer.name が timm 側で認識できる名前であること
  （sgd, adam, adamw, nadamw, lamb, lars, adafactor, adan, madgrad, muon…など。
    手元の timm バージョンでの一覧は timm.optim.list_optimizers() で確認できます）。

  optimizer.kwargs がその optimizer の __init__ が受け取れる引数であること（不一致は即エラー）。


## 6. `tile` セクション（Pair / BlindTrace で使用）

H 方向（トレース方向）をタイル分割して推論するための設定です。
主に **大きな gather を GPU メモリに載せるため**に使います。

| key | 型 | 意味 / 制約 |
|---|---:|---|
| `tile.tile_h` | `int` | タイルの高さ（トレース数）。**`tile.tile_h <= infer.subset_traces` 必須**。 |
| `tile.overlap_h` | `int` | オーバーラップ。`0 <= overlap_h < tile_h`。 |
| `tile.tiles_per_batch` | `int` | まとめて推論するタイル数。 |
| `tile.amp` | `bool` | tile 推論でも AMP を使うか。 |
| `tile.use_tqdm` | `bool` | tile 推論の進捗バー。 |

---

## 7. `vis` セクション（可視化）

全タスク共通で以下の 2 つを使います。

| key | 型 | 意味 |
|---|---:|---|
| `vis.out_subdir` | `str` | `out_dir` 配下に作るサブディレクトリ名（例: `vis`）。 |
| `vis.n` | `int` | 1 epoch あたりに保存する可視化サンプル数。 |

Pair / BlindTrace では追加で以下が使われます（タスク側で要求される場合あり）。

| key | 型 | 意味 |
|---|---:|---|
| `vis.cmap` | `str` | matplotlib colormap 名（例: `seismic`）。 |
| `vis.vmin` / `vis.vmax` | `float` | 画像レンジ。 |
| `vis.transpose_for_trace_time` | `bool` | trace×time の表示向きを入れ替えるか。 |
| `vis.per_trace_norm` | `bool` | 1 trace ごとに正規化して表示するか。 |
| `vis.per_trace_eps` | `float` | `per_trace_norm` の epsilon。 |
| `vis.figsize` | `list[float]` | figure サイズ（例: `[13.0, 8.0]`）。 |
| `vis.dpi` | `int` | 出力 dpi。 |

---

## 8. `scheduler` セクション（任意）
`scheduler` を未指定 / `null` にすると無効。指定時は learning rate scheduler が有効化されます。

### 8.1 共通キー
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `scheduler.type` | `str` | Yes | - | 使用するスケジューラ種別。下記のいずれか。 |
| `scheduler.interval` | `str` | No | type 依存 | `step` / `epoch`。type によって固定の場合あり。 |

### 8.2 type 別の必須キーと制約
- `warmup_cosine`: `interval` は `step` 固定。必須 `warmup_steps`（>=0）、任意 `eta_min`（float、既定 0.0）。`warmup_steps <= total_steps`。
- `step_lr`: `interval` 既定 `epoch`。必須 `step_size`（>0）、`gamma`（>0）。
- `multistep_lr`: `interval` 既定 `epoch`。必須 `milestones`（非空 list[int]）、`gamma`（>0）。
- `exponential_lr`: `interval` 既定 `epoch`。必須 `gamma`（>0）。
- `cosine_annealing`: `interval` 既定 `epoch`。任意 `t_max`（>0、未指定時は `epochs` または `total_steps`）、`eta_min`（float）。
- `cosine_warm_restarts`: `interval` 既定 `epoch`。必須 `t_0`（>0）、任意 `t_mult`（>0）、`eta_min`（float）。
- `one_cycle`: `interval` は `step` 固定。必須 `max_lr`、任意 `pct_start`（(0,1)）、`div_factor`（>0）、`final_div_factor`（>0）、`anneal_strategy`（`cos` / `linear`）。
- `reduce_on_plateau`: `interval` は `epoch` 固定。任意 `monitor`（既定 `infer_loss`）、`mode`（`min` / `max`）、`factor`（(0,1)）、`patience`（>=0）、`threshold`（>=0）、`min_lr`（>=0）。`monitor` は `infer_loss` / `infer/loss` / `train_loss` / `train/loss` 以外だとエラー。

---

## 9. `ckpt` セクション（best-only 固定）

| key | 型 | 必須 | 制約 |
|---|---:|:---:|---|
| `ckpt.save_best_only` | `bool` | Yes | **true 固定**（false はエラー）。 |
| `ckpt.metric` | `str` | Yes | **`infer_loss` 固定**（他はエラー）。 |
| `ckpt.mode` | `str` | Yes | **`min` 固定**（他はエラー）。 |

出力:
- `out_dir/ckpt/best.pt`

---

## 10. オプション: `ema` セクション

`ema` を指定すると EMA（Exponential Moving Average）を有効化できます。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `ema.enabled` | `bool` | No | `true` | `ema` セクションが存在する場合の有効/無効。 |
| `ema.decay` | `float` | No | `0.999` | EMA の減衰率。 |
| `ema.start_step` | `int` | No | `0` | EMA 更新を開始する global step。 |
| `ema.update_every` | `int` | No | `1` | EMA 更新間隔（step）。 |
| `ema.use_for_infer` | `bool` | No | `true` | 推論に EMA 重みを使うか。 |
| `ema.device` | `str` | No | （未指定） | `cpu` / `cuda` のみ。未指定ならモデルと同一デバイス。`cuda` は学習デバイスが CUDA のときのみ有効。 |

補足:
- `ema.use_for_infer=true` の場合、推論は EMA 重みで実行されます。
- `ema.device=cpu` は VRAM を節約できますが推論が遅くなります。
- EMA 状態は `best.pt` に保存されます。

---

## 11. `tracking` セクション（任意）

`tracking` は未指定でも動作します（デフォルト無効）。指定した場合は MLflow 互換トラッキングが有効化されます。

| key | 型 | 必須 | デフォルト | 意味 |
|---|---:|:---:|---:|---|
| `tracking.enabled` | `bool` | No | `false` | トラッキング有効化。 |
| `tracking.experiment_prefix` | `str` | No | `seisai` | experiment 名の prefix（`<prefix>/<pipeline>`）。 |
| `tracking.exp_name` | `str` | No | `baseline` | run 名に埋め込まれる識別子。 |
| `tracking.tracking_uri` | `str` | No | `file:./mlruns` | MLflow tracking URI。`file:` の相対パスは YAML 位置基準で絶対 `file:` に解決。 |
| `tracking.vis_best_only` | `bool` | No | `true` | best 更新時のみ vis をログ対象にする。 |
| `tracking.vis_max_files` | `int` | No | `50` | best 可視化のログ対象ファイル上限。 |

トラッキング有効時、`out_dir/tracking/` に以下が生成されます:
- `config.resolved.yaml`
- `git.txt`, `env.txt`
- `data_manifest.json`
- `optimizer_groups.json`
- （必要に応じて）`overlong_values.json`

---

## 12. 共通の落とし穴（チェックリスト）

- `paths.*_files` が空、または listfile が空
- `infer.num_workers` を 0 以外にしている（即エラー）
- `dataset.waveform_mode="mmap"` で `train.num_workers` / `infer.num_workers` が 0 以外（即エラー）
- `ckpt.*` が固定値から外れている（即エラー）
- `train.device` で CUDA を指定したが CUDA が利用できない
- `train.init_ckpt` の `in_chans` が一致しない（即エラー）
