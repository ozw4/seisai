# SegyGatherPairDataset B系（Pair整合性）テスト 技術メモ

本メモは `test_segy_gather_pair_dataset_pair_consistency_headers_shape_dtype_and_sync_crop` を中心に、
**Pair整合性（壊れると学習が死ぬ系）**をどのように保証しているかを、テストの意図・確認点・保証内容として整理する。

---

## 対象テスト

- `test_segy_gather_pair_dataset_pair_consistency_headers_shape_dtype_and_sync_crop(tmp_path)`

補助関数:
- `_read_trace_headers(path, indices)`

---

## 前提・狙い

本テストは、`SegyGatherPairDataset` が返す 1サンプルについて、以下を同時に満たすことを保証する。

1. **trace対応が崩れていない**
   - input/target が同一 `indices` を参照し、ヘッダ由来の識別子（ffid/chno/offset）が完全一致する。
2. **shape/dtype契約が成立している**
   - `sample['input']` と `sample['target']` が `torch.float32` で `(C,H,W)`（ここでは `C=1`）を満たし、
     `H==subset_traces`、`W==time_len` である。
3. **transform が input/target に同期して適用されている**
   - `RandomCropOrPad(target_len=time_len)` の crop window が input/target で同一であり、
     `meta['start']` により検証可能である。
   - 結果として、元データから `indices` と `start` で切り出した値と **完全一致**する。

---

## データ生成と期待値の作り方

### 合成データ
- `n_traces=20`, `n_samples=64` の 2D array を生成。
- target: `target[i, :] = arange(n_samples) + 1000*i`
- input: `inp = 2.0 * target`

この構成により、同一窓で切り出した場合に `input == 2*target` が厳密に成立し、
transform同期やtrace対応の破綻が即検出できる。

### SEGY生成
- `write_unstructured_segy(input_path, inp, dt_us)`
- `write_unstructured_segy(target_path, target, dt_us)`

ここで重要なのは、headerを含む「同一構造」のSEGYを2本作ることである。
header検証（ffid/chno/offset一致）を成立させるのは `write_unstructured_segy` の責務。

---

## 補助関数 `_read_trace_headers` の保証

### 目的
- Datasetが返す `indices` に対応するトレースのヘッダを読み出し、
  input/target の一致を **外部検証**する（Dataset内部ロジックに依存しない検証）。

### 読み出すヘッダ
- ffid: `TraceField.FieldRecord`
- chno: `TraceField.TraceNumber`
- offset: `TraceField.offset`

### 入力制約
- `indices` は 1D
- 空配列不可
- 負値不可（padなど -1 を含むケースは別テストで扱う）

---

## テストの検証項目（何を確認し、何を保証するか）

### 1) `input/target` の存在と dtype 契約
**確認**
- `out` に `input` と `target` が存在
- `torch.Tensor` である
- dtype が `torch.float32`

**保証**
- plan が `input/target` を生成している（最低限）
- downstream（train loop）で dtype が原因の不具合（double混入等）が起こらない

---

### 2) shape 契約（(C,H,W) と H==subset_traces, W==time_len）
**確認**
- `as_chw()` 後に `ndim==3`
- `x_in.shape == x_tg.shape`
- `C==1`
- `H==subset_traces`
- `W==time_len`（RandomCropOrPadが効いている）

**保証**
- PairDatasetの builder/stack が正しく機能し、学習側の期待形状と一致する
- crop/pad の結果が `time_len` で安定し、モデル入出力契約が崩れない

---

### 3) indices 契約（同一トレース対応の根拠）
**確認**
- `out['indices']` が `np.ndarray` で `shape==(subset_traces,)`
- `0 <= indices < n_traces`（今回は pad を扱わない前提なので負値なし）
- `out['offsets']` が Tensor で `shape==(subset_traces,)`

**保証**
- Datasetが「どのトレースを選んだか」を外部に提示し、
  Pair整合性・再現性の検証が可能である（テストの観測点がある）
- offsets がトレース数に一致し、同一windowに付随するメタとして整合している

---

### 4) transform 同期（input/targetが同一窓を見ている）
**確認**
- `out['meta']` が dict で、`'start'` を含む
- `start` が `0 <= start <= n_samples - time_len` の範囲
- 期待値の構築:
  - `exp_tg_hw = target[indices, start:start+time_len]`
  - `exp_in_hw = inp[indices, start:start+time_len]`
- 実値:
  - `got_tg_hw = x_tg[0].numpy()`
  - `got_in_hw = x_in[0].numpy()`
- `np.array_equal` で **完全一致**を要求

**保証**
- inputとtargetに適用される transform が同一の crop window を共有している
- 「入力窓と教師窓がズレる」致命的なデータリーク/学習破綻を防止できる
- `meta['start']` を通じて、外部から transform の決定性・同期性を検証できる

---

### 5) header由来識別子の完全一致（trace対応の外部保証）
**確認**
- `_read_trace_headers` により、inputとtargetの
  - `ffid`
  - `chno`
  - `offset`
  が `indices` 上で完全一致することを `np.array_equal` で検証

**保証**
- PairDataset が input/target のトレース対応を崩していないことを、
  **SEGYヘッダという外部ソース**で検証している
- 「同じインデックスで slice しただけ」ではなく、
  ペアSEGYが持つ識別子レベルで一致していることを保証する

---

### 6) Dataset提供 offsets と header offsets の一致
**確認**
- `out['offsets']`（float32にキャスト）と `header offset` を一致比較

**保証**
- Datasetが返す offsets が、入力SEGYのヘッダから導出したものと一致し、
  PairDataset内で offsets がズレていない（参照元が一貫している）

---

### 7) 合成データ由来の最終整合性（input==2*target）
**確認**
- `torch.allclose(x_in, 2.0*x_tg, atol=0, rtol=0)` を要求

**保証**
- 上記の (shape/dtype/transform同期/trace対応) が全て成立していることの
  統合的な sanity check
- どれか一つでも崩れると、厳密一致が崩れてテストが落ちる

---

## このテストで明示的にカバーしないこと（範囲外）

- pad（indices=-1）を含むケース
  - 本テストは `indices >= 0` を前提に header読出しするため、padケースは別テストで扱うべき。
- gather全域取得や superwindow の挙動検証
- 多ch入力（C>1）や builder の複雑構成

---

## 次に追加推奨の派生テスト

1. **padを含むケース**
   - `subset_traces > n_traces` にして `indices` に -1 を含め、
     pad領域の `input/target` がゼロになっていることを検証（header検証は -1 を除外して行う）。
2. **valid=False の secondary_key 非決定性**
   - RNG固定で secondary_key の分岐を安定化しつつ、
     期待する並び（chno or offset）を検証。
3. **header cache の整合性**
   - `use_header_cache=True` で二回初期化した場合に、
     `indices`→header一致が再現されることを検証。

---
