# Augment 開発者マニュアル

本ドキュメントは、SeisAI における **データ拡張（Augment）** の位置づけ・仕様・契約（守るべき要件）と、**新しい Augment を追加する際の実装ガイド**をまとめたものです。

---

## 1. 用語と前提

### 1.1 gather 配列の形状
- `x`: 2D NumPy 配列（dtype は `float32` を推奨）
- 形状は `(H, W)`
  - `H`: トレース数（高さ、trace axis）
  - `W`: サンプル数（時間方向、time axis）

### 1.2 Transform と Augment の関係
- Dataset は **SEG-Y から gather を抽出**したあと、`transform` を適用して学習用入力（view）を作ります。
- Augment は `transform` の一部として組み込まれます（`ViewCompose` などの合成で実装）。

---

## 2. Transform インターフェース（最重要契約）

### 2.1 Dataset から呼ばれる transform のシグネチャ
Dataset から transform は次の形で呼ばれます。

- 呼び出し: `transform(x, rng=rng, return_meta=True)`
- 返り値:
  - `y` または `(y, meta)`
  - `y` は **2D NumPy** であること
  - `meta` は **dict** であること（空 dict 可）

### 2.2 形状の契約
- **`H` は必ず保持**すること
  - `y.shape[0]` は入力 `x.shape[0]` と一致が必須
- `W` は変化してよい
  - 例: TimeStretch により `W` が変わる → その後の Crop/Pad で `time_len` に揃える、など

### 2.3 dtype / 数値の契約
- `y` は `np.float32` を維持すること（`np.pad` 等で `float64` にならないよう注意）
- `NaN` / `Inf` を生成しない（標準化・FFT・loss 計算で破綻しやすい）

---

## 3. ViewCompose と “op（Augment単体）” の契約

SeisAI では、複数の Augment を `ViewCompose` で順に適用し、メタ情報（`meta`）を統合します。

### 3.1 op の推奨シグネチャ
`ViewCompose` は各 op を次の形で呼びます。

- まず `op(x, rng, return_meta=True)` を試す
- それが `TypeError` になる場合のみ `op(x, rng)` を試す

したがって、新規 Augment は以下を満たすのが安全です。

- `__call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None, return_meta: bool = False)`
- 返り値: `y` または `(y, meta)`（`meta` は dict）

### 3.2 rng（乱数）の要件
- **与えられた `rng` を使用**して確率判定・パラメータサンプルを行うこと
- `np.random.random()` 等のグローバル乱数に依存しない
  - Pair タスクなどで input/target の変換が同期しなくなる原因になります

### 3.3 No-op（未適用時）の挙動
- `prob=0`、あるいは確率判定で未適用の場合、**入力をそのまま返す**
- 幾何に関わる Augment は、未適用でも **恒等変換を表す meta** を返すことを推奨します
  （例: `{'factor': 1.0}`, `{'factor_h': 1.0}`, `{'hflip': False}` など）

---

## 4. meta の仕様（投影・ゲート・可視化に影響）

Transform は `meta` を返し、Dataset は `meta` を使って first-break / offsets / time-grid などを view 空間へ投影します。
幾何（view 座標系）に影響する Augment を作る場合、**meta を正しく出すことが必須**です。

### 4.1 予約キー（本プロジェクトで解釈されるキー）
以下のキーは view 投影で利用されます。

| meta key | 型 | 意味 | 恒等（未適用） |
|---|---:|---|---|
| `hflip` | `bool` | トレース順を反転したか | `False` |
| `factor_h` | `float` (`>0`) | H方向（トレース方向）伸縮率（中心固定） | `1.0` |
| `factor` | `float` (`>0`) | 時間方向の伸縮率 | `1.0` |
| `start` | `int` (`>=0`) | time 窓の開始サンプル（Crop） | `0` |
| `trace_tshift_view` | `(H,) int` | トレースごとの time shift（view のサンプル単位） | `zeros(H)` |

> 注意: per-trace shift を投影に効かせたい場合、キー名は **`trace_tshift_view`** を使用してください。
> 別キーを使う場合は、投影側（view_projection）の拡張が必要になります。

### 4.2 投影の概念（理解用）
first-break index の投影（概念式）は以下です。

- H方向: `hflip` と `factor_h` により trace 軸を写像
- T方向: `fb_view = round(fb_raw * factor) - start`
- `trace_tshift_view` があれば `fb_view += trace_tshift_view[h]`

時間グリッド `time_view` は `factor` と `start` を使って、raw time grid を view に合わせて構成されます。
またゲート評価では `dt_eff = dt / factor` が使われます（時間伸縮を入れた Augment は `factor` 設定が必須）。

---

## 5. Augment の分類と配置（推奨）

### 5.1 幾何に関わる Augment（Geometry Augment）
- 例: HFlip / H方向伸縮 / TimeStretch / time crop（start） / per-trace time shift
- 要件:
  - `H` を変えない
  - 投影に必要な `meta` を返す（上表の予約キー）
- 配置:
  - **Crop/Pad より前**に適用するのが推奨
    （Crop が `start` を定義し、TimeStretch が `factor` を定義するため）

### 5.2 波形の値だけを変える Augment（Signal Augment）
- 例: 周波数フィルタ、極性反転、ノイズ付加、ランダムゲイン、マスク（time/freq dropout）
- 要件:
  - 幾何を変えないので `meta` は空 dict でよい
  - `NaN/Inf` を作らない
- 配置:
  - **Crop/Pad の後**に置くと、入力サイズが揃って扱いやすい

---

## 6. 追加できない（または拡張が必要な）Augment の例

次の操作は、投影・ラベル整合・ゲート評価に影響が大きく、追加前に設計検討が必要です。

- トレースの削除 / 追加（`H` が変わる）
- 任意 permutation / shuffle（`hflip` 以外の並び替え）
- 時間軸の非線形ワープ（`factor` だけで表現できない）
- トレースごとの異なる time stretch（`factor` がスカラー前提のため）

これらを導入する場合は、少なくとも次が必要になります。

- meta 表現の追加（例: `trace_src_map` のような dst->src mapping）
- `view_projection` 側の拡張（fb / offsets / time の投影を新 meta に追従させる）
- テスト（ラベル整合の検証を含む）

---

## 7. 新しい Augment を追加する手順（チェックリスト）

### 7.1 実装と公開 API
1) `packages/seisai-transforms/src/seisai_transforms/augment.py` にクラスを追加
2) 必要なら `packages/seisai-transforms/src/seisai_transforms/config.py` に設定 dataclass を追加
3) `packages/seisai-transforms/src/seisai_transforms/__init__.py` に export を追加（engine 側で import できるようにする）

### 7.2 YAML で設定できるようにする（任意）
4) `packages/seisai-engine/src/seisai_engine/pipelines/common/augment.py` に設定読み取り・validation を追加
5) `docs/examples/common_yaml_manual.md` にキー説明を追加

### 7.3 テスト（最低限）
- transforms の unit test:
  - 形状保持（H維持）
  - dtype（float32）
  - `prob=0` が No-op
  - `return_meta=True` の meta が dict
- e2e:
  - augment 有効時に学習ループが最後まで走る

---

## 8. Augment 実装テンプレ（コピペ可）

以下は「Signal Augment（幾何を変えない）」のテンプレです。

```python
import numpy as np

class RandomAdditiveNoise:
    def __init__(self, prob: float = 0.0, sigma_range: tuple[float, float] = (0.0, 0.02)) -> None:
        self.prob = float(prob)
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))
        if self.sigma_range[0] < 0.0 or self.sigma_range[1] < 0.0:
            raise ValueError('sigma_range must be >= 0')
        if self.sigma_range[0] > self.sigma_range[1]:
            raise ValueError('sigma_range must satisfy min <= max')

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None, return_meta: bool = False):
        if self.prob <= 0.0:
            return (x_hw, {}) if return_meta else x_hw

        r = rng or np.random.default_rng()
        if float(r.random()) >= self.prob:
            return (x_hw, {}) if return_meta else x_hw

        sigma = float(r.uniform(self.sigma_range[0], self.sigma_range[1]))
        if sigma <= 0.0:
            return (x_hw, {}) if return_meta else x_hw

        noise = r.normal(loc=0.0, scale=sigma, size=x_hw.shape).astype(np.float32, copy=False)
        y = (x_hw.astype(np.float32, copy=False) + noise).astype(np.float32, copy=False)
        return (y, {}) if return_meta else y
```

幾何に関わる Augment を追加する場合は、**meta（予約キー）を必ず返す**ようにしてください。
たとえば TimeStretch 相当なら `{'factor': f}`、H方向伸縮なら `{'factor_h': f}` などです。

---

## 9. よくある落とし穴

- `np.pad` の戻り dtype が `float64` になる
  → `astype(np.float32, copy=False)` を明示する
- `prob=0` のときに meta を返さず、後段が `factor` を参照して破綻
  → 幾何 Augment は恒等 meta を返す（`factor=1.0` 等）
- `meta` のキー衝突
  → `ViewCompose` は `meta.update()` で統合するため、同名キーは後勝ち
- rng を無視してグローバル乱数を使う
  → Pair の同期や再現性が壊れる
- `H` を変える、トレースを並び替える
  → 投影とラベル整合が崩れる（導入するなら投影の拡張が必要）

---

## 10. 仕様拡張の指針（高度な Augment を入れたい場合）

「任意の trace permutation」「trace dropout（削除）」「非線形 time warp」などを導入したい場合は、
まず **“投影が何を前提にしているか”** を明確にし、meta で表現できる最小の追加情報を設計してください。

推奨アプローチ:
- dst-trace -> src-trace の写像（`trace_src_map`）を meta に持たせる
- time 方向の写像をスカラー `factor` で表現できない場合は、1D のサンプル写像（例: `time_map`）を導入する
- `project_fb_idx_view / project_offsets_view / project_time_view` を新 meta に追従させる
- その上で unit test と e2e を追加し、ラベル整合を担保する
