# SEG-Y → SEG-Y 推論 CLI マニュアル（psn / pair / blindtrace）

このドキュメントは、`cli/run_*_infer.py` で提供している **SEG-Y → SEG-Y 推論**（いわゆる *segy2segy*）の使い方をまとめたものです。

対象エントリポイント:

- `cli/run_psn_infer.py` → `seisai_engine.pipelines.psn.infer_segy2segy.main`
- `cli/run_pair_infer.py` → `seisai_engine.pipelines.pair.infer_segy2segy.main`
- `cli/run_blindtrace_infer.py` → `seisai_engine.pipelines.blindtrace.infer_segy2segy.main`

参考 YAML:

- `examples/config_infer_psn.yaml`
- `examples/config_infer_pair.yaml`
- `examples/config_infer_blindtrace.yaml`

> 学習 YAML（`config_train_*.yaml`）のマニュアルは `docs/examples/*_yaml_manual.md` を参照してください。

---

## 1. まず理解しておくこと（設計の前提）

### 1.1 入力の単位

この CLI は SEG-Y を **FFID (= shot gather)** 単位で順に読み込み、gather ごとに 2D 配列 `(H,W)` を作ってモデル推論します。

- `H`: gather 内のトレース数
- `W`: 1トレースあたりのサンプル数

### 1.2 出力 SEG-Y の書き込み方

出力は「入力 SEG-Y の header を最大限保ったまま、trace サンプルのみ差し替える」方式です。

- **出力の sample format は IEEE float32 (format=5) に固定**
- binary header / trace header は src → dst へコピー
- text header (`text[0]`) の **末尾 4 行 (C37–C40) に ML メタ情報を追記**
- `overwrite=false` のとき、出力ファイルが存在すると **即エラー**

さらに、同名の sidecar JSON を必ず書きます:

- `*.sgy.mlmeta.json`（例: `foo.pair_pred.sgy.mlmeta.json`）

---

## 2. 実行方法（共通）

### 2.1 必要な前提

リポジトリ直下で editable install 済みであること（README の Install セクション参照）。

### 2.2 実行コマンド

いずれの infer CLI も **オプションは `--config` のみ**です。

```bash
# PSN (P/S/Noise) 推論
python cli/run_psn_infer.py --config examples/config_infer_psn.yaml

# Pair 推論（denoise / recon などの 1ch 回帰）
python cli/run_pair_infer.py --config examples/config_infer_pair.yaml

# Blindtrace 推論（striped-mask cover 推論）
python cli/run_blindtrace_infer.py --config examples/config_infer_blindtrace.yaml
```

標準出力には、生成した出力 SEG-Y のパスが 1 行ずつ出力されます。

---

## 3. YAML の解釈ルール（共通）

### 3.1 相対パスの基準

`--config path/to/foo.yaml` の `foo.yaml` が置かれている **ディレクトリを base_dir** として、以下が解決されます。

- `paths.segy_files` の相対パス
- `paths.out_dir` の相対パス
- `infer.ckpt_path` の相対パス

> 実行時のカレントディレクトリではなく、**YAML の位置が基準**です。

### 3.2 config のマージ優先順位

推論 CLI は、最終的に使う設定を次の優先順位で組み立てます:

1. defaults（CLI 内に埋め込まれた既定値）
2. checkpoint 内の `cfg`（`best.pt` などに保存された設定）
3. infer 用 YAML（`--config` で渡した YAML）

つまり **`infer.yaml > ckpt cfg > defaults`** です。

### 3.3 CLI の unknown override（`KEY=VALUE`）

`--config` 以外の引数は、すべて **unknown override** として解釈されます。

- 書式: `KEY=VALUE`
- `KEY` は `.` 区切りのネストパス（例: `tile.tile_w`）
- `VALUE` は YAML として解釈されます（`yaml.safe_load`）

例:

```bash
python cli/run_psn_infer.py --config examples/config_infer_psn.yaml tile.tile_w=4096

# list / null なども YAML として解釈される
python cli/run_psn_infer.py --config examples/config_infer_psn.yaml infer.outputs=[P,S]
python cli/run_pair_infer.py --config examples/config_infer_pair.yaml infer.ffids=[1001,1002]
python cli/run_pair_infer.py --config examples/config_infer_pair.yaml infer.ffids=null
```

注意:

- override token が `--something` で始まるものは **エラー**（この CLI は `--config` 以外のフラグを受け取りません）
- `VALUE` を文字列にしたい場合、必要ならクォートしてください（例: `infer.device="cuda:0"`）

---

## 4. override の安全制限（安全なキーのみ許可）

誤操作で checkpoint の構造や dataset 前提を破壊しないように、unknown override で上書きできるキーはデフォルトで制限されています。

制限外のキーを override したい場合は、

- YAML に `infer.allow_unsafe_override: true` を書く
- または unknown override で `infer.allow_unsafe_override=true` を最初に渡す

のどちらかが必要です。

> 一度 `infer.allow_unsafe_override=true` になった後は、以降の unknown override は無制限になります。

### 4.1 safe override key 一覧

**PSN (`run_psn_infer.py`)**

- `paths.segy_files`, `paths.out_dir`
- `infer.ckpt_path`, `infer.device`, `infer.out_suffix`, `infer.overwrite`, `infer.sort_within`, `infer.ffids`, `infer.outputs`, `infer.standardize_eps`, `infer.note`, `infer.allow_unsafe_override`
- `tile.tile_h`, `tile.overlap_h`, `tile.tile_w`, `tile.overlap_w`, `tile.tiles_per_batch`, `tile.amp`, `tile.use_tqdm`
- `tta`

**Pair (`run_pair_infer.py`)**

- `paths.segy_files`, `paths.out_dir`
- `infer.ckpt_path`, `infer.device`, `infer.out_suffix`, `infer.overwrite`, `infer.sort_within`, `infer.ffids`, `infer.standardize_eps`, `infer.note`, `infer.allow_unsafe_override`
- `tile.tile_h`, `tile.overlap_h`, `tile.tile_w`, `tile.overlap_w`, `tile.tiles_per_batch`, `tile.amp`, `tile.use_tqdm`
- `tta`

**Blindtrace (`run_blindtrace_infer.py`)**

- `paths.segy_files`, `paths.out_dir`
- `infer.ckpt_path`, `infer.device`, `infer.out_suffix`, `infer.overwrite`, `infer.sort_within`, `infer.ffids`, `infer.note`, `infer.allow_unsafe_override`
- `tile.tile_h`, `tile.overlap_h`, `tile.tile_w`, `tile.overlap_w`, `tile.tiles_per_batch`, `tile.amp`, `tile.use_tqdm`
- `cover.mask_ratio`, `cover.band_width`, `cover.noise_std`, `cover.mask_noise_mode`, `cover.use_amp`, `cover.offsets`, `cover.passes_batch`

---

## 5. config リファレンス（共通キー）

### 5.1 `paths`

| key | 型 | 意味 |
|---|---:|---|
| `paths.segy_files` | `list[str]` | 入力 SEG-Y（複数可） |
| `paths.out_dir` | `str` | 出力ディレクトリ |

### 5.2 `infer`

| key | 型 | 意味 |
|---|---:|---|
| `infer.ckpt_path` | `str` | 推論に使う checkpoint（必須） |
| `infer.device` | `str` | `auto|cpu|cuda|cuda:N` |
| `infer.out_suffix` | `str` | 出力ファイル名の suffix |
| `infer.overwrite` | `bool` | 出力が存在する場合に上書きするか |
| `infer.sort_within` | `str` | `none|chno|offset`（gather 内トレース並び） |
| `infer.ffids` | `list[int] | null` | 処理する FFID の allowlist（注意点は後述） |
| `infer.allow_unsafe_override` | `bool` | unknown override の安全制限を解除 |
| `infer.note` | `str` | text header と sidecar に残す自由記述 |

補足:

- `infer.device=auto` は「CUDA があれば cuda、なければ cpu」です。

### 5.3 `tile`

`tile` は gather `(H,W)` に対して 2D タイル分割で推論するためのパラメータです。

| key | 型 | 意味 |
|---|---:|---|
| `tile.tile_h` | `int` | タイルの高さ（trace 方向） |
| `tile.overlap_h` | `int` | `tile_h` のオーバーラップ |
| `tile.tile_w` | `int` | タイルの幅（time/sample 方向） |
| `tile.overlap_w` | `int` | `tile_w` のオーバーラップ |
| `tile.tiles_per_batch` | `int` | タイル推論をまとめる数（VRAM/速度に影響） |
| `tile.amp` | `bool` | CUDA のとき AMP を使うか |
| `tile.use_tqdm` | `bool` | タイル推論の tqdm を出すか |

制約:

- `tile_*` は正、`overlap_*` は 0 以上
- `overlap_* < tile_*`

実際の推論では gather サイズに合わせて `tile_h=min(tile_h, H)`, `tile_w=min(tile_w, W)` に自動で丸められます。

---

## 6. pipeline 別の違い

### 6.1 PSN（P/S/Noise 分類）

追加キー:

- `infer.outputs: list[str] | null`
  - 書かない/`null` のとき: checkpoint の `output_ids` 全てを出力
  - 指定する場合: **checkpoint の `output_ids` の部分集合**（重複不可）
- `infer.standardize_eps: float | null`
  - per-trace 標準化の `eps`
  - `null` のときは `transform.standardize_eps`（cfg 内にあれば）→なければ既定 `1e-8`
- `tta: list | null`
  - 現状は「要求として受け取って sidecar に残す」用途（`tta_applied` は空）

出力:

- **各クラスを別 SEG-Y として出力**します。
- ファイル名: `"{src.stem}.psn_{output_id}{infer.out_suffix}"`
  - 例: `20200623002546.psn_P.sgy`, `20200623002546.psn_S.sgy`, `20200623002546.psn_N.sgy`
- 値は **softmax 後の確率**（`float32`）です。

### 6.2 Pair（1ch 回帰）

追加キー:

- `infer.standardize_eps: float | null`
  - 入力を per-trace 標準化して推論し、推論結果を **元スケールに戻して**出力します。
- `tta: list | null`
  - PSN 同様に、現状は sidecar に残す用途（`tta_applied` は空）

出力:

- ファイル名: `"{src.stem}.pair{infer.out_suffix}"`（既定 `*_pred.sgy`）
  - 例: `20200623002546.pair_pred.sgy`

### 6.3 Blindtrace（striped mask cover 推論）

追加キー:

- `cover.*`（後述）

動作イメージ:

- gather を 1 回で全トレース予測するのではなく、
  **等間隔の striped マスクで「隠す→その位置を予測」** を複数回回し、
  全トレースを 1 回以上 “隠した状態” で予測できるように合成します。

`cover`:

| key | 型 | 意味 |
|---|---:|---|
| `cover.mask_ratio` | `float` | 1 パスで隠すトレース割合（(0,1]） |
| `cover.band_width` | `int` | マスクの連続バンド幅（>=1） |
| `cover.noise_std` | `float` | マスク位置に入れるノイズの標準偏差（>=0） |
| `cover.mask_noise_mode` | `str` | `replace|add` |
| `cover.use_amp` | `bool` | cover の内側（striped のバッチ推論）で AMP を使うか |
| `cover.offsets` | `list[int]` | ストライプ開始オフセット（複数指定で平均 = TTA） |
| `cover.passes_batch` | `int` | ストライプのパスをまとめる数（メモリ/速度に影響） |

補足:

- `cover.use_amp` は cover 側の autocast 制御です。
- 実際のモデル呼び出しはタイル推論（`tile.*`）を通るため、
  `tile.amp` と `cover.use_amp` は別スイッチです。

出力:

- ファイル名: `"{src.stem}.blindtrace{infer.out_suffix}"`（既定 `*_pred.sgy`）

---

## 7. `infer.ffids` の注意点（落とし穴）

`infer.ffids` は「指定 FFID の gather だけ処理する」ためのフィルタですが、
現在の実装では **指定した FFID 群でファイル内の全トレースが埋まらない場合にエラー**になります。

- エラー例: `some traces were not filled (miss=...)`

つまり `infer.ffids` を使うときは、次のどちらかが必要です。

- 入力 SEG-Y 自体が “その FFID だけ” を含むように切り出されている
- もしくは `infer.ffids` に **ファイル内の全 FFID** を渡す（デバッグ用途）

---

## 8. 出力に残るメタ情報

### 8.1 SEG-Y text header (`text[0]`) 末尾 4 行

各 pipeline は C37–C40 を上書きし、次のような情報を残します:

- pipeline 名
- checkpoint 名 / epoch / global_step（PSN/pair）
- model_sig の hash / cfg hash
- tile 設定（H/W, overlap, AMP, tiles_per_batch）
- UTC 時刻 / seisai-engine version / note

### 8.2 sidecar JSON（`*.mlmeta.json`）

`text[0]` に入りきらない詳細も含めて JSON に保存します。

- PSN は `output_id`, `output_ids`, `softmax_axis` などを含む
- pair は `standardize_eps`, `tta_requested` などを含む
- blindtrace は `cover.*` を含む

---

## 9. よくあるエラーと対処

### 9.1 `checkpoint pipeline must be "..."`

checkpoint 内の `pipeline` フィールドと、実行した infer CLI が一致していません。

- `psn` 用 ckpt で `run_pair_infer.py` を回していないか

### 9.2 `output already exists: ...`

- `infer.overwrite=true` にする
- もしくは出力先を変える（`paths.out_dir` / `infer.out_suffix`）

### 9.3 `unsafe override key is not allowed by default: ...`

unknown override で safe でないキーを上書きしようとしています。

- YAML に `infer.allow_unsafe_override: true`
- または override の先頭に `infer.allow_unsafe_override=true`

### 9.4 `some traces were not filled`

`infer.ffids` の指定が原因で、ファイル内の全トレースが埋まりませんでした。
「7. `infer.ffids` の注意点」を参照してください。
