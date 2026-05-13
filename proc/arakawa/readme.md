# Arakawa FBPICK Physical Export Runner 手順書

この手順書は、Arakawa データに対して以下を 1 コマンドで実行するための簡易ガイドです。

```text
Coarse infer
→ Physics / robust center 作成
→ physical_center_i を phase へ snap
→ grstat 形式の初動ファイルを書き出し
→ QC 可視化図を作成
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

詳細な出力先、評価、可視化設定を確認したい場合だけ、コメント付き full config を参照します。

```text
proc/arakawa/configs/run_coarse_physics_export.yaml
```

`proc/arakawa/configs/templates/*.yaml` は runner が読み込む template であり、ユーザーが直接実行・編集する config ではありません。旧 `coarse_one.yaml` / `physics_one.yaml` / `physics_qc_one_no_fb.yaml` / `fine_one.yaml` は deprecated です。

canonical layout は `proc/arakawa/README_LAYOUT.md` を参照してください。

---

## 2. 最小 config を編集する

最小 config は、基本的に `sgy_file` だけを書き換えれば実行できます。

```yaml
paths:
  sgy_file: fdata_hset_ARA26_Vib.sgy

visualization:
  enabled: true
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

重み、coarse model 設定、physics 設定は `proc/arakawa/configs/templates/` の canonical template を default として使います。通常はユーザーが変更する必要はありません。

---

## 3. 実行する

```bash
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/configs/run_coarse_physics_export_minimal.yaml
```

処理が成功すると、以下のような出力が作られます。

```text
proc/arakawa/outputs/coarse/<TAG>.coarse.npz
proc/arakawa/outputs/robust/<TAG>.robust.npz
proc/arakawa/outputs/grstat/<TAG>.physical_center.snap_peak.ltcor2.crd
proc/arakawa/outputs/grstat/<TAG>.physical_center.snap_peak.ltcor2.npz
proc/arakawa/outputs/qc/<TAG>/gather_*.png
proc/arakawa/outputs/qc/summary_global.csv
proc/arakawa/outputs/qc/summary_per_file.csv
proc/arakawa/outputs/<TAG>.arakawa_physical_export_summary.json
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

### 新 grstat format

現在の default 出力は新しい grstat format です。

```yaml
export:
  grstat_format: recno_channel_range
  values_per_line: 5
```

新 format は、各 `fb:` 行が以下の形になります。

```text
fb: recno start_ch end_ch fb値(start_ch) ... fb値(end_ch)
```

例:

```text
fb:          1       1       5    92.000    82.000    72.000    66.000    56.000
```

旧 format が必要な場合のみ、以下のように変更します。

```yaml
export:
  grstat_format: legacy
```

### 旧 grstat を新 format に変換する

既存の古い grstat 初動ファイルを新 format に直したい場合は、変換 CLI を使います。Arakawa の grstat 値が ms 単位で、サンプリング間隔が 2 ms の場合は `--dt-ms 2.0` を指定します。

```bash
python -m cli.convert_grstat_format \
  --input-crd /path/to/old_format.crd \
  --output-crd /path/to/new_format.crd \
  --dt-ms 2.0
```

旧 format へ戻したい場合は、以下を指定します。

```bash
python -m cli.convert_grstat_format \
  --input-crd /path/to/new_format.crd \
  --output-crd /path/to/legacy_format.crd \
  --dt-ms 2.0 \
  --output-format legacy
```

Python から使う場合は、`seisai_pick.pickio.io_grstat` にある IO を使います。

```python
from pathlib import Path

from seisai_pick.pickio.io_grstat import load_grstat_matrix, numpy2fbcrd

in_path = Path('/path/to/old_format.crd')
out_path = Path('/path/to/new_format.crd')
dt_ms = 2.0

parsed = load_grstat_matrix(
    in_path,
    dt_multiplier=dt_ms,
    strict_blocks=True,
    strict_channel_count=True,
)

numpy2fbcrd(
    dt=dt_ms,
    fbnum=parsed.samples,
    gather_range=parsed.record_numbers.tolist(),
    output_name=str(out_path),
    output_format='recno_channel_range',
    values_per_line=5,
    header_comment=f'converted from legacy grstat: {in_path.name}',
)
```

---

## 5. QC 可視化図

デフォルトでは、export 後に QC 可視化図も作成します。FB/教師データが無い場合は、runner が自動で dummy FB を作成するため、ユーザー側で追加ファイルを用意する必要はありません。

出力例:

```text
proc/arakawa/outputs/qc/<TAG>/gather_*.png
proc/arakawa/outputs/qc/summary_global.csv
proc/arakawa/outputs/qc/summary_per_file.csv
proc/arakawa/outputs/fb_dummy/<TAG>.fb_none.npy
```

デフォルトの可視化設定は以下です。

```yaml
visualization:
  enabled: true
  allow_no_fb: true
  max_gathers_per_file: 10
  gather_selection: even
  first_panel_only: true

  first_panel_flatten:
    enabled: true
    reference_key: physical_center_i
    half_samples: 256

  overlays:
    coarse_pmax: true
    trend_center: true
    physical_center: true
    fine_center: false
    window: false
    final_pick: false
    physical_model_status: true
```

この設定では、第1パネルのみを表示します。波形は `physical_center_i` を基準に flattening され、縦軸は `sample - physical_center_i` になります。表示範囲は上下 ±256 samples です。

表示する gather は先頭から連番ではなく、対象範囲から等間隔に選びます。たとえば FFID 1〜300 から 3 枚表示する場合は、おおむね 1, 150, 300 が選ばれます。

可視化を止めたい場合だけ、以下のようにします。

```yaml
visualization:
  enabled: false
```

---

## 6. ユーザー提供 grstat 初動と比較する場合

既存の grstat 初動ファイルと最終結果を比較したい場合は、config に `reference_grstat_path` を追加します。

```yaml
paths:
  sgy_file: fdata_hset_ARA26_Vib.sgy
  reference_grstat_path: /path/to/reference_first_break.crd
```

この場合、export 後に誤差評価も実行されます。

出力例:

```text
proc/arakawa/outputs/eval/<TAG>.physical_center.snap_peak.ltcor2.eval_summary.json
proc/arakawa/outputs/eval/<TAG>.physical_center.snap_peak.ltcor2.eval_summary.csv
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

### 評価だけ再実行する

参照 grstat を差し替えた場合や、trace ごとの詳細 CSV だけ後から出したい場合は、Coarse / Physics / export をやり直さず、評価だけ再実行できます。

config に書く場合:

```yaml
paths:
  sgy_file: fdata_hset_ARA26_Vib.sgy
  reference_grstat_path: /path/to/reference_first_break.crd

run:
  eval_only: true
```

CLI option で一時的に指定する場合:

```bash
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/configs/run_coarse_physics_export_minimal.yaml \
  --eval-only \
  --reference-grstat-path /path/to/reference_first_break.crd
```

trace ごとの詳細 CSV も出す場合:

```bash
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/configs/run_coarse_physics_export_minimal.yaml \
  --eval-only \
  --reference-grstat-path /path/to/reference_first_break.crd \
  --write-per-trace-csv
```

既存 export `.npz` が default path 以外にある場合は、`paths.export_npz` を指定します。

```yaml
paths:
  sgy_file: fdata_hset_ARA26_Vib.sgy
  reference_grstat_path: /path/to/reference_first_break.crd
  export_npz: /path/to/<TAG>.physical_center.snap_peak.ltcor2.npz

run:
  eval_only: true
```

---

## 7. 代表的な config 例

### SEG-Y だけ指定する最小例

```yaml
paths:
  sgy_file: fdata_hset_ARA26_Vib.sgy

visualization:
  enabled: true
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

## 8. 実行後に見るファイル

まず見るべきファイルは grstat 出力です。

```bash
ls -lh proc/arakawa/outputs/grstat/*.crd
```

summary も確認します。

```bash
cat proc/arakawa/*arakawa_physical_export_summary.json
```

可視化図も確認します。

```bash
ls -lh proc/arakawa/outputs/qc/*/gather_*.png
```

参照 grstat を指定した場合は、評価 summary を確認します。

```bash
cat proc/arakawa/outputs/eval/*.eval_summary.json
```

---

## 9. よくあるエラー

### SEG-Y が見つからない

`sgy_file` にファイル名だけを書いている場合、runner は Arakawa の default data root から探します。見つからない場合は絶対パスで指定してください。

```yaml
paths:
  sgy_file: /absolute/path/to/file.sgy
```

### checkpoint model_sig mismatch

coarse checkpoint と coarse config の model 設定が一致していません。Arakawa 用の既存 config を使っているか確認してください。

```text
proc/arakawa/configs/templates/coarse.yaml
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

## 10. 作業の最小手順まとめ

```bash
cd /workspace

# 1. config の sgy_file を編集
vi proc/arakawa/configs/run_coarse_physics_export_minimal.yaml

# 2. 実行
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/configs/run_coarse_physics_export_minimal.yaml

# 3. 出力確認
ls -lh proc/arakawa/outputs/grstat/*.crd
cat proc/arakawa/outputs/*arakawa_physical_export_summary.json
```

参照 grstat と比較したい場合は、config に `reference_grstat_path` を追加して同じコマンドを実行します。
