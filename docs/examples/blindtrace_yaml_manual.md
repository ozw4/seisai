# BlindTrace タスク: YAML 設定マニュアル（差分のみ）

共通の仕様（パス解決/listfile、dataset 共通、augment、scheduler、ckpt、ema、tracking、train/infer の共通キー、tile/vis の一般ルールなど）は `docs/examples/common_yaml_manual.md` を参照してください。

対象:
- エントリポイント: `cli/run_blindtrace_train.py` → `seisai_engine.pipelines.blindtrace.train.main`
- 参照 YAML: `examples/config_train_blindtrace.yaml`

---

## 1. BlindTrace の目的（何が違うか）
BlindTrace は入力 gather の一部トレースを **マスク（置換/加算ノイズ）**し、モデルで **元の波形を復元**する自己教師ありタスクです。

- 入力: マスク済み gather（`C×H×W`）
- 教師: マスク前の元波形（`1×H×W`）
- 損失: マスク領域のみ (`masked_only`) か、全領域 (`all`) を選択

---

## 2. `paths`（BlindTrace 固有ポイント）

### 2.1 必須キー
| key | 型 | 意味 |
|---|---:|---|
| `paths.segy_files` | `list[str]` または `str(listfile)` | 学習 SEG-Y |
| `paths.infer_segy_files` | `list[str]` または `str(listfile)` | 推論 SEG-Y |
| `paths.out_dir` | `str` | 出力先 |

### 2.2 FB pick（任意）と `fbgate` の扱い
BlindTrace では FB pick を任意で受け取れます（`paths.phase_pick_files` / `paths.infer_phase_pick_files`）。
ただし未指定の場合は **警告の上で `fbgate` を強制 off（min_pick_ratio=0）**になります。

---

## 3. `mask`（BlindTrace 固有）

| key | 型 | 意味 / 制約 |
|---|---:|---|
| `mask.ratio` | `float` | マスクするトレース比率（例: 0.5） |
| `mask.mode` | `str` | `replace` / `add` |
| `mask.noise_std` | `float` | `replace/add` に使うノイズ強度 |

---

## 4. `input`（BlindTrace 固有: 追加チャンネル）

| key | 型 | 意味 |
|---|---:|---|
| `input.use_offset_ch` | `bool` | offset チャンネルを追加するか |
| `input.offset_normalize` | `bool` | offset を正規化するか |
| `input.use_time_ch` | `bool` | time チャンネルを追加するか |

`use_offset_ch` / `use_time_ch` の組み合わせに応じて入力チャネル数が変化します（`in_chans = 1 + int(use_offset_ch) + int(use_time_ch)`）。

---

## 5. `train` / `eval`（BlindTrace 固有ポイント）

### 5.1 `loss_scope` と `masked_only`
- `loss_scope: masked_only` のとき、損失計算は `batch["mask_bool"]` を使って **マスク領域のみに限定**されます。
- 個別 loss に `scope` を書くと上書きできます。

### 5.2 代表的な loss
サンプル YAML では以下が使われています（params はサンプル YAML 参照）。

- `shift_robust_mse`
- `fx_mag_mse`
- `mse`

---

## 6. `transform`（BlindTrace 固有ポイント）
BlindTrace は `transform.per_trace_standardize` で標準化の on/off を切り替えます。

---

## 7. `tile` / `vis`（BlindTrace 固有ポイント）
`tile` と `vis` のキー説明・一般制約は共通マニュアルを参照しつつ、BlindTrace 固有の注意点だけまとめます。

- `tile` は **必須**（BlindTrace の設定では `tile` セクションが必須）
- `vis` は triptych 前提
- `model` の `in_chans/out_chans` は入力設定から計算され、合わないとエラーになる点に注意

---

## 8. 最小差分スニペット（例）

```yaml
paths:
  segy_files: path/to/segy_list.txt
  infer_segy_files: path/to/infer_segy_list.txt
  out_dir: ./_blindtrace_out

mask:
  ratio: 0.5
  mode: replace
  noise_std: 1.0

input:
  use_offset_ch: false
  offset_normalize: true
  use_time_ch: false

train:
  loss_scope: masked_only

transform:
  time_len: 5024
  per_trace_standardize: true

tile:  # 詳細は common_yaml_manual.md
  tile_h: 128
  overlap_h: 64
  tiles_per_batch: 16
  amp: true
  use_tqdm: false

vis:   # 詳細は common_yaml_manual.md
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
  backbone: resnet18
```
