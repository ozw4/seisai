# example_train_psn.py スモークテスト手順（最低限）

目的: `examples/example_train_psn.py` が最低限の契約を満たして動くことを、人手で短時間に確認する。

---

## 事前準備

1. `examples/example_train_psn.py` の以下を自分の環境に合わせて設定する。
   - `TRAIN_SEGY_PATH`
   - `TRAIN_PHASE_PICK_PATH`

2. 速く回すために、まずは以下を推奨（任意）。
   - `EPOCHS = 1`
   - `SAMPLES_PER_EPOCH = BATCH_SIZE`（1バッチだけ確実に回す）
   - （重ければ）`TRAIN_TIME_LEN` や `SUBSET_TRACES` を小さめにする

---

## 実行コマンド

```bash
python examples/example_train_psn.py
```

---

## チェック項目（受け入れ条件）

### 1) 1バッチだけ回して shape が出る

ログに以下の形式が出ること（値は環境依存）。

- `input=(B, 1, H, W)`
- `target=(B, 3, H, W)`
- `logits=(B, 3, H, W)`

例:

```text
[debug] epoch=0 input=(4, 1, 128, 4096) target=(4, 3, 128, 4096) logits=(4, 3, 128, 4096)
```

### 2) 1epochだけ学習が回る

`epoch=0 loss=...` のようなログが出て、例外で停止しないこと。

### 3) 可視化 png が1枚出る

以下のようなログが出て、ファイルが作られること。

```text
[saved] _psn_vis/psn_debug_epoch0000.png
```

### 4) metrics dict がログに出る

`[metrics] ...` の行が出ること（`nan` が混じっても「評価対象なし」の意味なのでOK）。

例:

```text
[metrics] p_mean=0.0000 p_median=0.0000 p_p_le_5=1.0000 ... s_mean=0.0000 ...
```

### 5) `pixel_mask.sum()==0` 相当でも NaN/例外が出ない（loss=0）

`run_epoch_debug()` 内で「空mask」を強制した loss を計算してログに出すため、
以下が満たされることを確認する:

- `loss_empty_mask` が `0.000000` である
- `loss_masked` が `nan` になっていない（有限値 or 0）

例:

```text
[debug] pixel_mask_sum=123456 loss_masked=0.693147 loss_empty_mask=0.000000
```

