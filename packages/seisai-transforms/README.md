# seisai-transforms

SeisAI 用の **信号処理 / データ拡張 / マスキング / view-space 投影** をまとめたパッケージです。

- 入力の基本形は **2D 配列 `(H, W)`**（H=traces, W=time samples）
- 学習時のデータ拡張（時間伸縮・周波数フィルタ・H反転など）
- 自己教師あり（inpainting 系）向けのマスク生成・推論補助
- 拡張後にラベル（FB index / phase picks / offsets / time grid）を整合させるための投影ユーティリティ

`seisai-dataset` / `seisai-engine` から呼ばれることを前提にしています。

---

## インストール

モノレポ内で editable install する想定です。

```bash
pip install -e packages/seisai-utils
pip install -e packages/seisai-transforms
```

依存: `numpy`, `torch`, `scipy`, `seisai-utils`（`pyproject.toml` 参照）。

---

## 形状・命名規約

- `x_hw`: `np.ndarray` の `(H, W)`
- `x_bchw`: `torch.Tensor` の `(B, C, H, W)`
- `fb_idx`: 初動の 1D インデックス列 `(H,)`（0-based だが **0 は無効扱い**、`-1` も無効）
- picks(CSR): `indptr (H+1,)` と `data (nnz,)` の CSR 形式（trace ごとに可変長の pick リスト）

---

## 主要コンセプト: ViewCompose と meta

学習時の拡張は「波形を変える」だけでなく、
**その拡張がラベル座標に与える影響**（hflip / 伸縮率 / crop start など）も保持する必要があります。

このパッケージでは、
- 各 transform が `return_meta=True` のときに **拡張メタ情報（dict）** を返し
- `ViewCompose` がそれらを順に適用しつつ **meta をマージ**します

```python
import numpy as np
from seisai_transforms import (
    RandomHFlip,
    RandomSpatialStretchSameH,
    RandomTimeStretch,
    RandomCropOrPad,
    ViewCompose,
)

x = np.random.randn(128, 2048).astype(np.float32)  # (H,W)

view = ViewCompose([
    RandomHFlip(prob=0.5),
    RandomSpatialStretchSameH(),
    RandomTimeStretch(),
    RandomCropOrPad(target_len=1536),
])

y, meta = view(x, rng=np.random.default_rng(0), return_meta=True)
```

### meta の代表キー（投影で参照されるもの）

- `hflip: bool` …… H 方向反転
- `factor_h: float` …… H 方向（trace axis）の中心固定リサンプル係数
- `factor: float` …… W 方向（time axis）の伸縮係数
- `start: int` …… W 方向の crop 開始位置（window の左端）
- `trace_tshift_view: (H,) int16` …… 一部トレースに対する view 空間での time shift

補足:
- `ViewCompose` は op ごとに meta を `dict.update()` で統合します（同名キーは後勝ち）。
- 推論でランダムが走ると困る場合、`seisai-dataset` 側では RNG 呼び出しを禁止するダミー RNG を使っています。
  推論パイプラインでは **prob=0** や **決定論 transform**（例: `DeterministicCropOrPad`）を利用してください。

---

## Augment（データ拡張）

`seisai_transforms.augment` には以下が含まれます。

- `RandomFreqFilter(FreqAugConfig)`
  - rFFT 上の smooth mask による low/high/band-pass 系のランダムフィルタ
- `RandomTimeStretch(TimeAugConfig)`
  - `scipy.signal.resample_poly` で時間軸を伸縮（出力の W は変化します）
  - meta: `{'factor': f}`
- `RandomSpatialStretchSameH(SpaceAugConfig)`
  - H 方向のみ中心固定で伸縮（出力の H は維持）
  - meta: `{'factor_h': f}`
- `RandomHFlip(prob)`
  - H 方向反転（trace order の reverse）
  - meta: `{'hflip': bool}`
- `RandomPolarityFlip(prob)`
  - 極性反転（`x -> -x`）
  - meta: `{'polarity_flip': bool}`
- `RandomCropOrPad(target_len)` / `DeterministicCropOrPad(target_len)`
  - W 方向の crop/pad
  - meta: `{'start': int}`
- `PerTraceStandardize(eps)`
  - W 軸で平均0・分散1（NumPy / Torch 両対応）
- `RandomSparseTraceTimeShift(...)`
  - 少数トレースだけ W 方向に数サンプルずらす（site effect 風の局所ずれを模擬）
  - meta: `{'trace_tshift_view': (H,)}`（キー名は `meta_key` で変更可）

---

## View-space 投影（ラベル整合）

データ拡張後にラベルを「同じ view 空間」へ揃えるための関数群です。

- `project_fb_idx_view(fb_idx, H, W, meta) -> (H,)`
  - hflip / factor_h / factor / start / trace_tshift_view を適用して FB index を view に投影
- `project_pick_csr_view(indptr, data, H, W, meta) -> (indptr_v, data_v)`
  - phase pick 等の CSR リストを view に投影（範囲外や `<=0` は drop）
- `project_offsets_view(offsets, H, meta) -> (H,)`
  - offsets を hflip + factor_h（線形補間）で view に投影
- `project_time_view(time_1d, H, W, meta) -> (W,)`
  - 元の time grid から view の time grid を生成（`dt_eff = dt / factor`、`start` を反映）

これらは `seisai-dataset` の推論用 window dataset などで使用します。

---

## Masking（自己教師あり / 欠損補完）

`seisai_transforms.masking` は、(H,T) に対する bool マスク生成と破壊（ノイズ注入）を提供します。

- `MaskGenerator.traces(ratio, width, mode, noise_std)`
  - trace 帯（H 方向バンド）マスク
- `MaskGenerator.checker_jitter(cfg, mode, noise_std)`
  - ジッター付き 2D ブロック（チェッカー）マスク

`mode`:
- `'replace'` …… masked 部分を `N(0, noise_std)` で置換（`noise_std=0` ならゼロ埋め）
- `'add'` …… masked 部分に `N(0, noise_std)` を加算

---

## マスク推論ユーティリティ（完全被覆ストライプ）

`cover_all_traces_predict_striped()` は、
等間隔ストライプ状に masked input を複数回作ってモデルを回し、
**全トレースが少なくとも1回はマスクされる**ように予測を合成します。

- 目的: blind-trace / inpainting 系の推論で、マスク位置の予測を全面的に回収する
- `offsets` を複数与えると開始位置をずらした TTA（平均化）になります

---

## Signal ops

`seisai_transforms.signal_ops` に、学習前処理や特徴量化に使う軽量ユーティリティがあります。

- scaling: per-trace standardize / AGC / robust AGC / softmax
- analytic: envelope
- smoothing: smooth

---

## テスト

```bash
pytest packages/seisai-transforms/tests -q
```

（ルートから実行する場合は `pytest -q` で workspace 全体のテストが走ります。）
