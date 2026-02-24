# seisai-dataset

SEG-Y ギャザー（shot/recv/cmp など）を **PyTorch Dataset** として扱うためのツール群です。`seisai-transforms`（波形/ビュー変換）と `seisai-pick`（ピック/ラベル生成）を前提に、

- SEG-Y のヘッダ読み出しとキャッシュ
- ギャザー内トレースのランダムサブセット抽出（superwindow 対応）
- 変換後ビューへのメタ情報投影（time/offset/FB pick など）
- 品質ゲート（min-pick / FBLC）
- 学習用の入力/教師データ組み立て（BuildPlan）

をまとめて提供します。

## Requirements

- Python >= 3.10
- 依存: `numpy`, `scipy`, `torch`, `segyio`, `tqdm`, `seisai-pick`, `seisai-transforms`

`segyio` は環境によってネイティブ依存が入ることがあります（インストール方法は環境に合わせてください）。

## Install（開発用 / monorepo）

リポジトリルートから editable install する想定です。

```bash
pip install -e ./packages/seisai-dataset
```

他パッケージも合わせて使う場合は、必要に応じて `seisai-pick` / `seisai-transforms` も editable で入れてください。

## 入力データの前提

### 1) SEG-Y

- `segyio.open(..., ignore_geometry=True)` で読み出せる SEG-Y を想定しています。
- 主要ヘッダは既定で以下を使用します（必要なら差し替え可能）:
  - `ffid_byte = segyio.TraceField.FieldRecord`
  - `chno_byte = segyio.TraceField.TraceNumber`
  - `cmp_byte  = segyio.TraceField.CDP`

### 2) First-break（FB） picks: `.npy`

`SegyGatherPipelineDataset` は `fb_files` に `.npy` を受け取ります。

- 形状: `(n_traces,)`
- dtype: 整数
- 値: **サンプルインデックス**（`> 0` が有効、`<= 0` は無効扱い）

### 3) Phase picks（P/S）: CSR `.npz`

`SegyGatherPhasePipelineDataset` は、PhaseNet 風の可変長ピック列を CSR で受け取ります。

- 必須キー: `p_indptr`, `p_data`, `s_indptr`, `s_data`
- `indptr` は `(n_traces + 1,)`、`indptr[-1] == len(data)`
- `data` は `(nnz,)` の整数配列（`> 0` が有効、`<= 0` は無効扱い）

## 主要 Dataset

### `SegyGatherPipelineDataset`

FB `.npy` を使う標準パイプラインです。

**load → subset sample → transform →（min-pick/FBLC gate）→ BuildPlan**

返却 dict（代表）:

- `input`: `torch.Tensor`（`(C,H,W)` または plan に依存）
- `target`: `torch.Tensor`（学習用 plan の場合）
- `fb_idx`: `torch.Tensor (H,)`（ビュー投影前の FB、無効は `<=0`）
- `file_path`: str
- `did_superwindow`: bool
- `trace_valid`: `torch.Tensor (H,)`（pad トレース等の無効フラグ）

### `SegyGatherPhasePipelineDataset`

CSR 形式の P/S ピックを使うパイプラインです。

- P-first を従来互換の `fb_idx` として扱います
- plan 側に CSR pick を渡してラベル生成できます（例: `PhasePSNMap`）

### `SegyGatherPairDataset`

input/target で **対応する SEG-Y ペア**を使う Dataset です（transform を同期適用）。

- 例: noisy → clean、masked → original など

### `InferenceGatherWindowsDataset`

推論用に gather を決定論で window 列挙します。

- W（時間）方向は基本 crop しません（不足時だけ右 0pad）
- H（トレース）方向は `win_size_traces` / `stride_traces` で window 化

可変 W を右 0pad して `(B,C,H,Wmax)` にする `collate_pad_w_right` も同梱しています。

### `NoiseTraceSubsetDataset`

イベントを除外し、ノイズっぽい trace subset を返す Dataset です。

## Transform の契約

`SegyGather*Dataset` に渡す `transform` は、原則として以下の呼び出しに対応してください。

- `transform(x_hw: np.ndarray, rng: np.random.Generator, return_meta: bool=True)`
- 戻り値:
  - `np.ndarray (H,W)`
  - もしくは `(np.ndarray (H,W), meta: dict)`

`meta` は任意ですが、`seisai-transforms` の `ViewCompose` を使うと、
crop/pad/stretch/hflip などの情報が `meta` に集約されます。

## BuildPlan（入力/ラベル組み立て）

`BuildPlan` は `sample` dict を in-place に加工して、`sample['input']` / `sample['target']` を構築します。

最小例（FB Gaussian を教師にする）:

```python
import numpy as np
from seisai_dataset import (
    SegyGatherPipelineDataset,
    FirstBreakGate,
    FirstBreakGateConfig,
)
from seisai_dataset.builder.builder import (
    BuildPlan,
    FBGaussMap,
    MakeOffsetChannel,
    MakeTimeChannel,
    SelectStack,
)
from seisai_transforms import PerTraceStandardize, ViewCompose

transform = ViewCompose([PerTraceStandardize(eps=1e-8)])

fbgate = FirstBreakGate(
    FirstBreakGateConfig(
        percentile=95.0,
        thresh_ms=8.0,
        min_pairs=16,
        apply_on='any',
        min_pick_ratio=0.0,
    )
)

plan = BuildPlan(
    wave_ops=[
        MakeTimeChannel(dst='time_ch'),
        MakeOffsetChannel(dst='offset_ch', normalize=True),
    ],
    label_ops=[
        FBGaussMap(dst='fb_map', sigma=10),
    ],
    input_stack=SelectStack(
        keys=['x_view', 'offset_ch', 'time_ch'],
        dst='input',
        dtype=np.float32,
        to_torch=True,
    ),
    target_stack=SelectStack(
        keys=['fb_map'],
        dst='target',
        dtype=np.float32,
        to_torch=True,
    ),
)

ds = SegyGatherPipelineDataset(
    segy_files=["/path/to/gathers.sgy"],
    fb_files=["/path/to/fb.npy"],
    transform=transform,
    fbgate=fbgate,
    plan=plan,
    primary_keys=('ffid',),
    subset_traces=128,
    use_header_cache=True,
)

sample = ds[0]
x = sample['input']   # torch.Tensor (C,H,W)
y = sample['target']  # torch.Tensor (1,H,W)
meta = sample.get('meta')
```

より具体的な例は以下を参照してください。

- `examples/phase_dataset_quick_check.py`
- `examples/noise_dataset_quick_check.py`
- `src/example/example_segy_gather_pipeline_ds.py`

## ヘッダキャッシュ

`use_header_cache=True` の場合、SEG-Y のヘッダ（FFID/CHNO/CMP/offset/dt 等）を `.npz` にキャッシュします。

- 既定の保存先: `<segy_path>.headers.<endian>.npz`
- 変更: `header_cache_dir=...`

SEG-Y の更新時刻よりキャッシュが古い場合は自動で再構築します。

## 開発

テスト（パッケージ単体）:

```bash
pytest -q ./packages/seisai-dataset/tests
```
