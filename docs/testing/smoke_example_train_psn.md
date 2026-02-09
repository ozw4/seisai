# example_train_psn.py スモークテスト手順（最低限）

目的: `examples/example_train_psn.py` が最低限の契約を満たして動くことを、人手で短時間に確認する。

---

## 事前準備

1. `examples/config_train_psn.yaml` の以下を自分の環境に合わせて設定する。
   - `paths.segy_files`
   - `paths.phase_pick_files`
   - `paths.infer_segy_files`
   - `paths.infer_phase_pick_files`
   - `paths.out_dir`（出力先）

2. 速く回すために、まずは以下を推奨（任意）。
   - `train.epochs = 1`
   - `train.samples_per_epoch = train.batch_size`（1バッチだけ確実に回す）
   - （重ければ）`train.time_len` や `train.subset_traces` を小さめにする

---

## 実行コマンド

```bash
python examples/example_train_psn.py --config examples/config_train_psn.yaml
```

---

## チェック項目（受け入れ条件）

### 1) 1epochだけ学習が回る

`epoch=0 loss=...` のようなログが出て、例外で停止しないこと。

### 2) 推論が回る

`epoch=0 infer_loss=...` のようなログが出て、例外で停止しないこと。

### 3) 可視化 png が1枚出る

以下のファイルが作られること。

- `<out_dir>/vis/epoch_0000/step_0000.png`

### 4) best checkpoint が出る

以下のファイルが作られること。

- `<out_dir>/ckpt/best.pt`
