from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LoaderConfig:
    pad_traces_to: int = 128  # トレース本数の下側ゼロパディング上限


@dataclass(frozen=True)
class TraceSubsetSamplerConfig:
    # キー選択
    primary_keys: tuple[str, ...] | None = None  # 例: ('ffid','chno','cmp')
    primary_key_weights: tuple[float, ...] | None = None  # 重み(同順)
    # superwindow
    use_superwindow: bool = False
    sw_halfspan: int = 0  # 片側キー数(K = 1 + 2*sw_halfspan)
    sw_prob: float = 0.3  # superwindow を適用する確率(<1.0なら確率適用)
    # secondary 整列ルールを固定したいときに secondary_key_fixed=True
    secondary_key_fixed: bool = False
    # 連続サブセット本数(不足時は後段でパディング)
    subset_traces: int = 128


@dataclass(frozen=True)
class FirstBreakGateConfig:
    percentile: float = 95.0
    thresh_ms: float = 8.0
    min_pairs: int = 16
    # ここを拡張:
    apply_on: Literal['any', 'super_only', 'off'] = 'any'
    min_pick_ratio: float | None = 0.0  # 0.0 or None で無効
    verbose: bool = False
