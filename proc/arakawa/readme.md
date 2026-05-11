# Arakawa FBPICK Physical Export Runner 手順書

この手順書は、Arakawa データに対して以下を 1 コマンドで実行するための簡易ガイドです。

```text
Coarse infer
→ Physics / robust center 作成
→ physical_center_i を phase へ snap
→ grstat 形式の初動ファイルを書き出し
→ 任意でユーザー提供 grstat 初動との誤差評価
```

最終出力は、physics stage で作成された `physical_center_i` を基準にします。

---

## 1. 事前確認

repo root で作業します。

```bash
cd /workspace
```

runner が追加済みであることを確認します。

```bash
ls cli/run_arakawa_fbpick_physical_export.py
```

Arakawa 用 config があることを確認します。

```bash
ls proc/arakawa/configs/run_coarse_physics_export_minimal.yaml
ls proc/arakawa/configs/run_coarse_physics_export.yaml
```

通常ユーザーは、最小 config の方を使います。

```text
proc/arakawa/configs/run_coarse_physics_export_minimal.yaml
```

---

## 2. 最小 config を編集する

最小 config は、基本的に `sgy_file` だけを書き換えれば実行できます。

```yaml
paths:
  sgy_file: fdata_hset_ARA26_Vib.sgy
```

ファイル名だけを書いた場合、runner はデフォルトで以下のディレクトリから SEG-Y を探します。

```text
/home/dcuser/data/ActiveSeisField/Arakawa2026
```

別の場所にある SEG-Y を使う場合は、絶対パスで指定します。

```yaml
paths:
  sgy_file: /path/to/your/file.sgy
```

重み、coarse model 設定、physics 設定は既存の Arakawa config を default として使います。通常はユーザーが変更する必要はありません。

---

## 3. 実行する

```bash
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/configs/run_coarse_physics_export_minimal.yaml
```

処理が成功すると、以下のような出力が作られます。

```text
proc/arakawa/coarse/<TAG>.coarse.npz
proc/arakawa/robust/<TAG>.robust.npz
proc/arakawa/grstat/<TAG>.physical_center.snap_peak.ltcor2.crd
proc/arakawa/grstat/<TAG>.physical_center.snap_peak.ltcor2.npz
proc/arakawa/<TAG>.arakawa_physical_export_summary.json
```

`<TAG>` は SEG-Y の親ディレクトリ名とファイル stem から自動生成されます。

---

## 4. grstat 出力の意味

デフォルトでは以下の設定で grstat 初動ファイルを作成します。

```yaml
export:
  pick_key: physical_center_i
  phase_mode: peak
  max_shift_samples: 2
  duplicate_policy: error
```

意味は以下です。

```text
physical_center_i を初期 pick とする
→ 波形上の peak へ ±2 samples 以内で snap する
→ grstat 形式で .crd を出力する
```

`duplicate_policy: error` は、同じ `(FFID, CHNO)` が複数 trace に出た場合に処理を止める安全設定です。通常はこのままで使ってください。

---

## 5. ユーザー提供 grstat 初動と比較する場合

既存の grstat 初動ファイルと最終結果を比較したい場合は、config に `reference_grstat_path` を追加します。

```yaml
paths:
  sgy_file: fdata_hset_ARA26_Vib.sgy
  reference_grstat_path: /path/to/reference_first_break.crd
```

この場合、export 後に誤差評価も実行されます。

出力例:

```text
proc/arakawa/eval/<TAG>.physical_center.snap_peak.ltcor2.eval_summary.json
proc/arakawa/eval/<TAG>.physical_center.snap_peak.ltcor2.eval_summary.csv
```

主な評価指標:

```text
n_eval
prediction_valid_at_reference_rate
bias_samples_mean
mae_samples_mean
mae_samples_p50 / p90 / p95 / p99 / max
bias_ms_mean
mae_ms_mean
mae_ms_p50 / p90 / p95 / p99 / max
R1 / R2 / R4 / R8 / R16 / R32 / R64 / R127
```

trace ごとの詳細 CSV が必要な場合は、以下を有効にします。

```yaml
evaluation:
  write_per_trace_csv: true
```

---

## 6. 代表的な config 例

### SEG-Y だけ指定する最小例

```yaml
paths:
  sgy_file: fdata_hset_ARA26_Vib.sgy
```

### 絶対パスで SEG-Y を指定する例

```yaml
paths:
  sgy_file: /home/dcuser/data/ActiveSeisField/Arakawa2026/fdata_hset_ARA26_Vib.sgy
```

### grstat 参照ファイルも指定する例

```yaml
paths:
  sgy_file: /home/dcuser/data/ActiveSeisField/Arakawa2026/fdata_hset_ARA26_Vib.sgy
  reference_grstat_path: /workspace/proc/arakawa/reference/fdata_hset_ARA26_Vib_reference.crd
```

---

## 7. 実行後に見るファイル

まず見るべきファイルは grstat 出力です。

```bash
ls -lh proc/arakawa/grstat/*.crd
```

summary も確認します。

```bash
cat proc/arakawa/*arakawa_physical_export_summary.json
```

参照 grstat を指定した場合は、評価 summary を確認します。

```bash
cat proc/arakawa/eval/*.eval_summary.json
```

---

## 8. よくあるエラー

### SEG-Y が見つからない

`sgy_file` にファイル名だけを書いている場合、runner は Arakawa の default data root から探します。見つからない場合は絶対パスで指定してください。

```yaml
paths:
  sgy_file: /absolute/path/to/file.sgy
```

### checkpoint model_sig mismatch

coarse checkpoint と coarse config の model 設定が一致していません。Arakawa 用の既存 config を使っているか確認してください。

```text
proc/arakawa/configs/coarse_one.yaml
```

### duplicate FFID/CHNO error

同じ `(FFID, CHNO)` が複数 trace に存在します。通常は SEG-Y header または channel 定義を確認してください。理由が分かっていて処理を通したい場合のみ、以下のように変更できます。

```yaml
export:
  duplicate_policy: first
```

または、後の trace を採用する場合:

```yaml
export:
  duplicate_policy: last
```

通常は `error` のままが推奨です。

---

## 9. 作業の最小手順まとめ

```bash
cd /workspace

# 1. config の sgy_file を編集
vi proc/arakawa/configs/run_coarse_physics_export_minimal.yaml

# 2. 実行
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/configs/run_coarse_physics_export_minimal.yaml

# 3. 出力確認
ls -lh proc/arakawa/grstat/*.crd
cat proc/arakawa/*arakawa_physical_export_summary.json
```

参照 grstat と比較したい場合は、config に `reference_grstat_path` を追加して同じコマンドを実行します。
