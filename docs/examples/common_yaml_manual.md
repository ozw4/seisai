# YAML 共通説明（PSN / Pair / BlindTrace）

本メモは `docs/examples/psn_yaml_manual.md` / `docs/examples/pair_yaml_manual.md` / `docs/examples/blindtrace_yaml_manual.md` に共通する設定の説明をまとめたもの。
各タスク固有の設定や制約は、それぞれのマニュアルを参照。

---

## 1. パス解決・listfile 展開ルール

### 1.1 相対パスの基準
以下のキーは、YAML ファイルのあるディレクトリ（`base_dir`）基準で絶対パスへ解決される。

- `paths.*_files`（キー名はタスクごと）
- `paths.out_dir`（出力ディレクトリ）

また `tracking.tracking_uri` が `file:` URI の場合、`file:./mlruns` のような相対指定は `base_dir` 基準で `file:/abs/path` に解決される。

### 1.2 listfile（1行1パス）
`paths.*_files` は **list[str]** だけでなく **listfile へのパス（str）** として指定できる。

- listfile の中身は「1行1パス」
- 空行は無視
- `#` で始まる行はコメントとして無視
- 環境変数展開（`$VAR`）と `~` 展開を行う
- listfile 内の相対パスは **listfile 自体のディレクトリ**基準で解決
- 展開後に全ファイルの存在チェックを行う（存在しないと即エラー）
- listfile が空の場合はエラー

---

## 2. `augment` セクション（任意・学習のみ）

`augment` は学習側の transform にのみ適用され、未指定なら全ての確率が 0 で実質無効。
適用順序は `hflip/space/time` → `RandomCropOrPad` → `freq/polarity` → `PerTraceStandardize`（有効時）。

### 2.1 共通
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `augment.hflip_prob` | `float` | No | `0.0` | 水平方向（トレース順）反転の確率。`[0,1]`。 |
| `augment.polarity_prob` | `float` | No | `0.0` | 極性反転の確率。`[0,1]`。 |

### 2.2 `augment.space`（空間伸縮）
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `augment.space.prob` | `float` | No | `0.0` | 空間方向の伸縮を適用する確率。`[0,1]`。 |
| `augment.space.factor_range` | `list[float]` | No | `[0.90, 1.10]` | 伸縮率の範囲。両方 `>0` かつ `min <= max`。 |

### 2.3 `augment.time`（時間伸縮）
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `augment.time.prob` | `float` | No | `0.0` | 時間方向の伸縮を適用する確率。`[0,1]`。 |
| `augment.time.factor_range` | `list[float]` | No | `[0.95, 1.05]` | 伸縮率の範囲。両方 `>0` かつ `min <= max`。 |

### 2.4 `augment.freq`（周波数フィルタ）
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `augment.freq.prob` | `float` | No | `0.0` | 周波数フィルタを適用する確率。`[0,1]`。 |
| `augment.freq.kinds` | `list[str]` | No | `[bandpass, lowpass, highpass]` | フィルタ種類。空は不可。許可値は `bandpass` / `lowpass` / `highpass`。 |
| `augment.freq.band` | `list[float]` | No | `[0.05, 0.45]` | バンド中心範囲。`[0,1]` 内で `min <= max`。 |
| `augment.freq.width` | `list[float]` | No | `[0.10, 0.35]` | バンド幅範囲。`[0,1]` 内で `min <= max`。 |
| `augment.freq.roll` | `float` | No | `0.02` | ロールオフ。`[0,1]`。 |
| `augment.freq.restandardize` | `bool` | No | `false` | フィルタ後に再標準化するか。 |

---

## 3. `scheduler` セクション（任意）

`scheduler` を未指定 / `null` にすると無効。指定時は learning rate scheduler が有効化される。

### 3.1 共通キー
| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `scheduler.type` | `str` | Yes | - | 使用するスケジューラ種別。下記のいずれか。 |
| `scheduler.interval` | `str` | No | type 依存 | `step` / `epoch`。type によって固定の場合あり。 |

### 3.2 type 別の必須キーと制約
- `warmup_cosine`: `interval` は `step` 固定。必須 `warmup_steps`（>=0）、任意 `eta_min`（float、既定 0.0）。`warmup_steps <= total_steps`。
- `step_lr`: `interval` 既定 `epoch`。必須 `step_size`（>0）、`gamma`（>0）。
- `multistep_lr`: `interval` 既定 `epoch`。必須 `milestones`（非空 list[int]）、`gamma`（>0）。
- `exponential_lr`: `interval` 既定 `epoch`。必須 `gamma`（>0）。
- `cosine_annealing`: `interval` 既定 `epoch`。任意 `t_max`（>0、未指定時は `epochs` または `total_steps`）、`eta_min`（float）。
- `cosine_warm_restarts`: `interval` 既定 `epoch`。必須 `t_0`（>0）、任意 `t_mult`（>0）、`eta_min`（float）。
- `one_cycle`: `interval` は `step` 固定。必須 `max_lr`、任意 `pct_start`（(0,1)）、`div_factor`（>0）、`final_div_factor`（>0）、`anneal_strategy`（`cos` / `linear`）。
- `reduce_on_plateau`: `interval` は `epoch` 固定。任意 `monitor`（既定 `infer_loss`）、`mode`（`min` / `max`）、`factor`（(0,1)）、`patience`（>=0）、`threshold`（>=0）、`min_lr`（>=0）。`monitor` は `infer_loss` / `infer/loss` / `train_loss` / `train/loss` 以外だとエラー。

---

## 4. `ckpt` セクション（best-only 固定）

| key | 型 | 必須 | 制約 |
|---|---:|:---:|---|
| `ckpt.save_best_only` | `bool` | Yes | **true 固定**（false はエラー）。 |
| `ckpt.metric` | `str` | Yes | **`infer_loss` 固定**（他はエラー）。 |
| `ckpt.mode` | `str` | Yes | **`min` 固定**（他はエラー）。 |

出力:
- `out_dir/ckpt/best.pt`

---

## 5. オプション: `ema` セクション

`ema` を指定すると EMA（Exponential Moving Average）を有効化できる。

| key | 型 | 必須 | デフォルト | 意味 / 制約 |
|---|---:|:---:|---:|---|
| `ema.enabled` | `bool` | No | `true` | `ema` セクションが存在する場合の有効/無効。 |
| `ema.decay` | `float` | No | `0.999` | EMA の減衰率。 |
| `ema.start_step` | `int` | No | `0` | EMA 更新を開始する global step。 |
| `ema.update_every` | `int` | No | `1` | EMA 更新間隔（step）。 |
| `ema.use_for_infer` | `bool` | No | `true` | 推論に EMA 重みを使うか。 |
| `ema.device` | `str` | No | （未指定） | `cpu` / `cuda` のみ。未指定ならモデルと同一デバイス。`cuda` は学習デバイスが CUDA のときのみ有効。 |

補足:
- `ema.use_for_infer=true` の場合、推論は EMA 重みで実行される。
- `ema.device=cpu` は VRAM を節約できるが推論が遅くなる。
- EMA 状態は `best.pt` に保存される。

---

## 6. `tracking` セクション（任意）

`tracking` は未指定でも動作する（デフォルト無効）。指定した場合は MLflow 互換トラッキングが有効化される。

| key | 型 | 必須 | デフォルト | 意味 |
|---|---:|:---:|---:|---|
| `tracking.enabled` | `bool` | No | `false` | トラッキング有効化。 |
| `tracking.experiment_prefix` | `str` | No | `seisai` | experiment 名の prefix（`<prefix>/<pipeline>`）。 |
| `tracking.exp_name` | `str` | No | `baseline` | run 名に埋め込まれる識別子。 |
| `tracking.tracking_uri` | `str` | No | `file:./mlruns` | MLflow tracking URI。`file:` の相対パスは YAML 位置基準で絶対 `file:` に解決。 |
| `tracking.vis_best_only` | `bool` | No | `true` | best 更新時のみ vis をログ対象にする。 |
| `tracking.vis_max_files` | `int` | No | `50` | best 可視化のログ対象ファイル上限。 |

トラッキング有効時、`out_dir/tracking/` に以下が生成される:
- `config.resolved.yaml`
- `git.txt`, `env.txt`
- `data_manifest.json`
- `optimizer_groups.json`
- （必要に応じて）`overlong_values.json`

---

## 7. 共通の落とし穴（チェックリスト）

- `paths.*_files` が空、または listfile が空
- `infer.num_workers` を 0 以外にしている（即エラー）
- `dataset.waveform_mode="mmap"` で `train.num_workers` / `infer.num_workers` が 0 以外（即エラー）
- `ckpt.*` が固定値から外れている（即エラー）
- `train.device` で CUDA を指定したが CUDA が利用できない
