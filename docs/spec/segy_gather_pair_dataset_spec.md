# SegyGatherPairDataset 仕様案(SEGYペア写像学習)

## 目的
- 2つの SEGY(例: noise / clean)の **対応ペア**から、同一 gather(同一 trace subset)を取り出し、
  **同期 transform** を適用した上で、`input -> target` の写像学習用サンプルを生成する。
- `FirstBreakGate` や fb 関連処理は **一切行わない**。

---

## 前提条件
- `input_segy_files[i]` と `target_segy_files[i]` は対応ペアである。
- 各ペアの input/target は以下が一致していることを仮定する：
  - trace 数
  - サンプル数(nsamples)
  - dt(サンプリング間隔)
  - (原則)ヘッダ値一致(ただし全トレースの完全照合は必須ではない)

---

## 提供クラス
### `seisai_dataset/segy_gather_pair_dataset.py`
- クラス名：`SegyGatherPairDataset(torch.utils.data.Dataset)`

---

## コンストラクタ
```python
SegyGatherPairDataset(
    input_segy_files: list[str],
    target_segy_files: list[str],
    transform,
    plan: BuildPlan,
    *,
    ffid_byte=segyio.TraceField.FieldRecord,
    chno_byte=segyio.TraceField.TraceNumber,
    cmp_byte=segyio.TraceField.CDP,
    primary_keys: tuple[str, ...] | None = None,
    primary_key_weights: tuple[float, ...] | None = None,
    use_superwindow: bool = False,
    sw_halfspan: int = 0,
    sw_prob: float = 0.3,
    use_header_cache: bool = True,
    header_cache_dir: str | None = None,
    subset_traces: int = 128,
    secondary_key_fixed: bool = False,
    verbose: bool = False,
    max_trials: int = 2048,
)
```

### 引数の意味
- `input_segy_files`, `target_segy_files`
  - 同じ長さで、`zip(..., strict=True)` でペア化できること。
- `transform`
  - 2D array `(H, W0)` を受け取り `(H, W)` を返す(または `(x_view, meta)`)。
  - **H(trace方向)を変えない**こと(`(H, W0) -> (H, W)`)。H を変える transform は非対応。
  - `rng` 引数を受け取り、乱数は **必ずその rng からのみ**消費すること(同期のため)。
- `plan: BuildPlan`
  - ペア対応 plan(後述)で `input` と `target` を生成する。
- sampler/loader 系の引数は `SegyGatherPipelineDataset` と同等の意味で使用。

---

## 内部データ構造
### file info
- `file_infos: list[dict]` を持つ。各要素はペアに対応し、最低限以下を含む：
  - `input_info`: `build_file_info(input_path, ..., include_centroids=True)` の結果
  - `target_mmap`: target segy の mmap / 読み出しハンドル(`TraceSubsetLoader` が読める形式)
  - `target_path`
  - (軽量一致チェック用の)`target_dt_sec`, `target_nsamples`, `target_ntraces` など

> サンプリングに必要な keys/offsets/dt は **input_info** を主に使う。

---

## 初期化時チェック
- `len(input_segy_files) > 0`
- `len(input_segy_files) == len(target_segy_files)`
- 各ペアで以下を **軽量チェック**(取れる範囲で)：
  - nsamples 一致
  - dt 一致
  - trace 数一致
- 不一致の場合は `ValueError`。

---

## サンプリング仕様(混ぜ方)
- `__len__` は固定値(例: 1024)を返す(現行 `SegyGatherPipelineDataset` と同方式)。
- `__getitem__` は引数 index を使わず、内部 RNG によりランダムに：
  1. ファイルペア `pair_idx` を選ぶ
  2. その input_info を使って `TraceSubsetSampler.draw(...)` で `indices` を得る
  3. `max_trials` 回以内に有効サンプルを返す(失敗時は `RuntimeError`)

---

## transform 同期仕様(方式A)
- 1サンプルにつき seed を 1つ作る(dataset の `_rng` から生成)。
- 同 seed から独立に：
  - `rng_in = np.random.default_rng(seed)`
  - `rng_tg = np.random.default_rng(seed)`
- transform を2回呼ぶ：
  - `out_in = transform(x_in, rng=rng_in, return_meta=True)`
  - `out_tg = transform(x_tg, rng=rng_tg, return_meta=True)`
- `meta` は原則 `input` 側の `meta` を採用し、`target` 側は破棄(もしくは整合確認用途に任意保持)。

---

## `__getitem__` の処理フロー
1. `pair_idx` をランダム選択
2. `sample = sampler.draw(input_info, ...)` で `indices` 等を取得
3. `x_in = subsetloader.load(input_mmap, indices)`
4. `x_tg = subsetloader.load(target_mmap, indices)`
5. 同期 transform を適用して
   - `x_view_input`, `meta`
   - `x_view_target`
   を得る
6. `plan.run(sample_for_plan, rng=self._rng)` を実行
7. `sample_for_plan` に `input` と `target` が生成されていることを確認
8. 出力 dict を返す

---

## Plan 連携仕様(ペア対応)
### Dataset から Plan に渡すキー(例)
`sample_for_plan` には最低限以下を含める：

- `x_view_input: np.ndarray`  # (H, W)
- `x_view_target: np.ndarray` # (H, W)
- `meta: dict`
- `dt_sec: float`
- `offsets: np.ndarray`(必要なら。input_info から indices で切る)
- `trace_valid: np.ndarray`(pad を行う場合。shape=(H,), bool)
- `file_path_input: str`
- `file_path_target: str`
- `indices: np.ndarray`
- `key_name: str`
- `secondary_key: str`
- `primary_unique: str`

### Plan/Builder 側の期待
- **Identity Builder を拡張**し、参照元を指定できる：
  - `Identity(source_key='x_view_input')` → `input` を生成
  - `Identity(source_key='x_view_target')` → `target` を生成
- これにより Plan は「どっちを input/target にするか」を構成で決定できる。

---

## Dataset の返却値仕様
`__getitem__` は少なくとも以下を返す：

- `input: torch.Tensor`(例: (C,H,W))
- `target: torch.Tensor`(例: (C,H,W))
- `meta: dict`
- `dt_sec: torch.Tensor`
- `offsets: torch.Tensor`(任意)
- `trace_valid: torch.Tensor`(任意。pad を行う場合。shape=(H,), bool)
- `file_path_input: str`
- `file_path_target: str`
- `indices: np.ndarray`
- `key_name: str`
- `secondary_key: str`
- `primary_unique: str`
- `did_superwindow: bool`(`verbose` のときのみ返す/それ以外は None でも可)

---

## エラー/リトライ仕様
- `max_trials` 回試して有効サンプルが得られなければ `RuntimeError`
- fb/gate による棄却は無いので、棄却理由は主に
  - transform が不正値を返す
  - indices が空/不正(サンプラの仕様による)
  - I/O エラー(想定外として例外で即失敗)

---

## 期待される拡張ポイント
- `strict_header_check` オプション(ヘッダ一致をサンプリングチェック)
- target 側の offsets/dt/nsamples の完全整合チェック
- transform 同期の堅牢化(将来的に params-based API への移行)
