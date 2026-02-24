# PSN タスク: YAML 設定マニュアル（差分のみ）

共通の仕様（パス解決/listfile、dataset 共通、augment、scheduler、ckpt、ema、tracking、train/infer の共通キーなど）は `docs/examples/common_yaml_manual.md` を参照してください。

対象:
- エントリポイント: `cli/run_psn_train.py` → `seisai_engine.pipelines.psn.train.main`
- 参照 YAML: `examples/config_train_psn.yaml`

---

## 1. PSN の目的（何が違うか）
PSN（P/S/Noise）は、CSR 形式の位相ピック（P/S）から **(3,H,W) の確率ターゲット**を作り、`EncDec2D` で **P/S/Noise を画素単位推定**します。

- 入力: SEG-Y gather → `(1,H,W)`
- 教師: `(3,H,W)`（P, S, Noise の確率。各画素で `P+S+N=1`）

---

## 2. `paths`（PSN 固有: CSR 位相ピック）
PSN は SEG-Y と 1:1 対応する位相ピック（CSR `.npz`）を必要とします。

### 2.1 必須キー
| key | 型 | 意味 |
|---|---:|---|
| `paths.segy_files` | `list[str]` または `str(listfile)` | 学習 SEG-Y |
| `paths.phase_pick_files` | `list[str]` または `str(listfile)` | 学習 CSR 位相ピック（`.npz`） |
| `paths.infer_segy_files` | `list[str]` または `str(listfile)` | 推論 SEG-Y |
| `paths.infer_phase_pick_files` | `list[str]` または `str(listfile)` | 推論 CSR 位相ピック（`.npz`） |
| `paths.out_dir` | `str` | 出力先 |

整合性:
- `len(segy_files) == len(phase_pick_files)`
- `len(infer_segy_files) == len(infer_phase_pick_files)`

### 2.2 CSR `.npz` の想定フォーマット（要点）
ファイル内に少なくとも以下が必要です。

- `p_indptr: (n_traces+1,) int`
- `p_data: (nnz_p,) int`（サンプル index）
- `s_indptr: (n_traces+1,) int`
- `s_data: (nnz_s,) int`

---

## 3. `train`（PSN 固有ポイント）

### 3.1 `train.psn_sigma`
`psn_sigma` は、位相ピックから確率マップを生成する際の **にじませ幅（ガウス幅）**として使われます。

---

## 4. `transform`（PSN 固有ポイント）
PSN は Per-trace 標準化を必ず行います。

- `transform.standardize_eps`（既定 `1e-8`）

---

## 5. `model`（PSN 固有ポイント）
PSN は 3 クラスなので、原則:

- `model.in_chans: 1`
- `model.out_chans: 3`

---

## 6. 最小差分スニペット（例）

```yaml
paths:
  segy_files: path/to/segy_list.txt
  phase_pick_files: path/to/picks_list.txt
  infer_segy_files: path/to/infer_segy_list.txt
  infer_phase_pick_files: path/to/infer_picks_list.txt
  out_dir: ./_psn_out

train:
  psn_sigma: 1.5

transform:
  time_len: 6016
  standardize_eps: 1.0e-8

model:
  backbone: resnet18
  in_chans: 1
  out_chans: 3
```
