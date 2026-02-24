# seisai-utils

SeisAI モノレポで共通利用する **設定ロード / 再現性（乱数）/ 分散補助 / メトリクスログ / 型・shape検証 / 可視化 / SEGY書き出し** をまとめたユーティリティ集です。

- どのパッケージからも参照される前提のため、依存はできるだけ軽くしています
- 一方で、モジュールごとに `numpy` / `torch` / `segyio` 等を使うため、実運用では（他パッケージと同じく）それらが入っている環境を想定します

> 注: `seisai_utils/__init__.py` は空なので、基本は `from seisai_utils.<module> import ...` で import してください。

---

## Requirements

- Python >= 3.10
- `PyYAML`, `matplotlib`（`pyproject.toml` の必須依存）
- 追加で必要になるもの（利用モジュール次第）:
  - `numpy`, `torch`（`rng` / `convert` / `validator` / `viz*` / `logging` など）
  - `segyio`（`segy_write` を使う場合）

---

## Install（開発用 / monorepo）

```bash
pip install -e packages/seisai-utils
```

---

## 何が入っているか（モジュール別）

### 1) YAML 設定ロード（base 継承 + パス解決）

- `seisai_utils.config_yaml.load_yaml(path) -> dict`
  - `base:` キーで **複数 YAML の継承**（deep merge）
  - いくつかの `paths.*` キーを **YAML のあるディレクトリ基準で絶対パス化**
    - 対象キー: `segy_files`, `phase_pick_files`, `infer_segy_files`, `infer_phase_pick_files`,
      `input_segy_files`, `target_segy_files`, `infer_input_segy_files`, `infer_target_segy_files`
  - `base` の循環参照は例外で即失敗

- `seisai_utils.config.load_config(path) -> dict`
  - 実体は `load_yaml` の薄いラッパ

#### 例: base 継承 + paths の解決

```yaml
# configs/train.yaml
base: ./base.yaml

paths:
  segy_files:
    - data/train.sgy
  out_dir: ./out

train:
  seed: 42
```

```python
from seisai_utils.config import load_config, require_dict, require_int

cfg = load_config('configs/train.yaml')
paths = require_dict(cfg, 'paths')
seed = require_int(cfg['train'], 'seed')

print(paths['segy_files'])
```

### 2) 型付き config 取得（require / optional）

- `require_value(d, key, types, ...)` / `optional_value(d, key, default, types, ...)`
- よく使うショートカット:
  - `require_dict`, `require_list_str`, `require_int`, `require_float`, `require_bool`
  - `optional_int`, `optional_float`, `optional_bool`, `optional_str`, `optional_tuple2_float`

例:

```python
from seisai_utils.config import require_float, optional_bool

lr = require_float(cfg['train'], 'lr')
use_amp = optional_bool(cfg['train'], 'amp', default=True)
```

---

### 3) 乱数（再現性）

- `seisai_utils.rng.set_seed(seed=42)`
  - `PYTHONHASHSEED` / `random` / `numpy.default_rng`（本レポ用）/ `torch` を一括で固定
  - `torch.backends.cudnn.deterministic=True` / `benchmark=False` を設定
- `seisai_utils.rng.get_np_rng() -> np.random.Generator`
  - レポ内で共有する NumPy RNG（`np.random.default_rng`）
- `seisai_utils.rng.worker_init_fn(worker_id)`
  - `DataLoader(worker)` でのワーカ初期化用

例:

```python
from torch.utils.data import DataLoader
from seisai_utils.rng import set_seed, worker_init_fn

set_seed(123)
loader = DataLoader(ds, num_workers=4, worker_init_fn=worker_init_fn)
```

---

### 4) 分散学習補助（torch.distributed）

- `seisai_utils.dist.init_distributed_mode(args)`
  - `RANK/WORLD_SIZE/LOCAL_RANK` または `SLURM_PROCID` を見て初期化
  - `args` に `rank/world_size/local_rank/dist_backend/distributed` をセット
  - rank0 以外の `print` を抑制（`setup_for_distributed`）
- 便利関数: `get_rank`, `get_world_size`, `is_main_process`, `save_on_master`

---

### 5) 学習ログ（SmoothedValue / MetricLogger）

- `seisai_utils.logging.SmoothedValue`
  - 直近 window の `median/avg` と全体 `global_avg` を保持
  - 分散時は `synchronize_between_processes()` が count/total を all-reduce
- `seisai_utils.logging.MetricLogger`
  - `update(loss=..., lr=...)` でメトリクスを追加
  - `log_every(iterable, print_freq, header)` で定期的にログを出す generator

例:

```python
from seisai_utils.logging import MetricLogger

logger = MetricLogger()
for batch in logger.log_every(loader, print_freq=50, header='train'):
    loss = step(batch)
    logger.update(loss=loss)
```

---

### 6) NumPy/Torch 変換（convert）

- `to_torch(x, like=None) -> torch.Tensor`
  - `numpy.ndarray` は `torch.from_numpy`（CPUゼロコピー）
  - `like` があれば dtype/device を合わせる
- `to_numpy(*xs) -> np.ndarray | None | tuple[...]`
  - tensor は `detach().cpu().numpy()`
- `first_or_self(v)`
  - list なら先頭、空なら `None`
- `to_bool_mask_torch(valid, like=ref) -> torch.bool tensor`
  - bool / int / float のマスクを `ref.device` 上の bool mask に変換

---

### 7) Checkpoint ロード（ckpt）

- `seisai_utils.ckpt.load_checkpoint(model, path)`
  - checkpoint dict の中から `model_ema` / `state_dict` / `model` を優先探索
  - 見つからなければロードした object 自体を state_dict とみなす
  - `strict=False` でロードし、missing/unexpected の個数を表示

---

### 8) SEGY 書き出し（入力 SEGY をコピーして trace データだけ差し替え）

- `seisai_utils.segy_write.write_segy_like_input(...) -> Path`
  - `src_path` を **丸ごとコピー**して、trace サンプルだけ上書き
  - text/binary/trace header と sample format を維持
  - `data_hw` は `(n_traces, n_samples)`
  - `trace_indices` を指定すれば一部の trace のみ差し替え

例:

```python
import numpy as np
from seisai_utils.segy_write import write_segy_like_input

pred = np.zeros((100, 6016), dtype=np.float32)

dst = write_segy_like_input(
    src_path='in.sgy',
    out_dir='out',
    out_suffix='_pred.sgy',
    data_hw=pred,
    overwrite=False,
)
print(dst)
```

---

### 9) 入力検証（validator）

NumPy / Torch の **型・ndim・空配列**などを簡単に検証する関数群です。

- `validate_array(x, allowed_ndims=(1,2,3,4), backend='auto', ...)`
- `require_float_array(x, backend='auto')` / `require_boolint_array(x, backend='auto')`
- `require_non_negative(x, ...)` / `require_all_finite(x, ...)`
- `require_same_shape_and_backend(a, b, *others, ...)`

---

## 可視化（viz / viz_phase / viz_pair / viz_wiggle）

### 1) (H,W) imshow 系（viz）

- `imshow_hw(ax, data_hw, transpose_for_trace_time=True, ...)`
  - `transpose_for_trace_time=True` のとき **x=Trace(H), y=Time(W)** になる向きで表示
- `save_imshow_row(png_path, panels, ...)`
  - 横一列の複数パネル保存（triptych 用）
- `save_triptych_bchw(x_in_bchw, x_tg_bchw, x_pr_bchw, out_path, ...)`
  - (B,C,H,W) の Input/Target/Pred を 3 枚で保存

### 2) PSN デバッグ（viz_phase）

- `save_psn_debug_png(out_path, x_bchw, target_b3hw, logits_b3hw, ...)`
  - 2x3 パネルで wave / target P,S / pred P,S,Noise を保存
- `make_title_from_batch_meta(batch, b=0) -> str | None`
  - collate 済み batch / meta からタイトルを best-effort 生成

### 3) Pair の triptych（viz_pair）

- `save_pair_triptych_png(...)` / `save_pair_triptych_step_png(...)`
- `make_pair_suptitle(batch, b=0)`

### 4) Wiggle プロット（viz_wiggle）

- `plot_wiggle(data, cfg=WiggleConfig(...)) -> Axes`
  - 正側塗りつぶし（`fill_positive=True`）
  - per-trace / global 正規化、gain/clip、trace spacing
  - pick overlay（`PickOverlay`）を scatter で重畳可能

---

## テスト

```bash
pytest packages/seisai-utils/tests -q
```

`viz_phase` のテストは `matplotlib` の backend を `Agg` に切り替えて実行します（GUI不要）。
