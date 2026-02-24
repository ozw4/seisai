# Pair タスク: YAML 設定マニュアル（差分のみ）

共通の仕様（パス解決/listfile、dataset 共通、augment、scheduler、ckpt、ema、tracking、train/infer の共通キー、tile/vis の一般ルールなど）は `docs/examples/common_yaml_manual.md` を参照してください。

対象:
- エントリポイント: `cli/run_pair_train.py` → `seisai_engine.pipelines.pair.train.main`
- 参照 YAML: `examples/config_train_pair.yaml`

---

## 1. Pair の目的（何が違うか）
Pair は **input SEG-Y と target SEG-Y のペア**を 1:1 対応させ、同じ gather window から **波形回帰**（denoise / mapping）を行います。

- 入力: input gather → `(1,H,W)`
- 教師: target gather → `(1,H,W)`
- 推論: H 方向タイル（`tile`）で分割推論し、可視化は triptych（Input/Target/Pred）

---

## 2. `paths`（Pair 固有のキー名と整合性）

### 2.1 必須キー
| key | 型 | 意味 |
|---|---:|---|
| `paths.input_segy_files` | `list[str]` または `str(listfile)` | 学習 input SEG-Y |
| `paths.target_segy_files` | `list[str]` または `str(listfile)` | 学習 target SEG-Y |
| `paths.infer_input_segy_files` | `list[str]` または `str(listfile)` | 推論 input SEG-Y |
| `paths.infer_target_segy_files` | `list[str]` または `str(listfile)` | 推論 target SEG-Y |
| `paths.out_dir` | `str` | 出力先 |

整合性チェック（即エラー）:
- `len(input_segy_files) == len(target_segy_files)`
- `len(infer_input_segy_files) == len(infer_target_segy_files)`

### 2.2 SEG-Y ペア整合性（即エラー）
各ペアについて次が一致しないと即エラーです。

- `n_samples`（時間サンプル数）
- `n_traces`（トレース数）
- `dt`（サンプル間隔）

---

## 3. `dataset`（Pair 固有ポイント）

### 3.1 endian キーが input/target で分かれる
Pair は input/target で SEG-Y を別々に読むため、endian キーが分かれています。

- `dataset.train_input_endian`
- `dataset.train_target_endian`
- `dataset.infer_input_endian`
- `dataset.infer_target_endian`

（値は `big` / `little`）

### 3.2 `dataset.standardize_from_input` は現状実質固定
Pair の Dataset 生成は現状 **常に input から z-score を計算し、input/target 両方へ適用**する前提です（YAML の指定があっても挙動が変わらないケースがあります）。

---

## 4. `transform`（Pair 固有ポイント）
Pair の `transform.time_len` は **学習側のみ**で使われます。

- 学習: `RandomCropOrPad(time_len)` を適用
- 推論: **時間方向の crop/pad は行わない**（tile 推論の都合）

---

## 5. `tile` / `vis`（Pair 固有ポイント）
`tile` と `vis` のキー説明・一般制約は共通マニュアルを参照しつつ、Pair 固有の注意点だけまとめます。

- `tile` は **必須**（Pair の config では `tile` セクションが必須）
- `vis` は triptych 前提（Input / Target / Pred）
- 代表的には `model.out_chans: 1` を想定（回帰 1ch）

---

## 6. 最小差分スニペット（例）

```yaml
paths:
  input_segy_files:  path/to/input_list.txt
  target_segy_files: path/to/target_list.txt
  infer_input_segy_files:  path/to/infer_input_list.txt
  infer_target_segy_files: path/to/infer_target_list.txt
  out_dir: ./_pair_out

dataset:
  train_input_endian: big
  train_target_endian: big
  infer_input_endian: big
  infer_target_endian: big

transform:
  time_len: 6016

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
  figsize: [13.0, 8.0]
  dpi: 150

model:
  backbone: resnet18
  in_chans: 1
  out_chans: 1
```
