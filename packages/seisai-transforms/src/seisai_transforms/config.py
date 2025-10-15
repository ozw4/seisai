from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class FreqAugConfig:
    prob: float = 0.0
    kinds: Tuple[str, ...] = ("bandpass", "lowpass", "highpass")
    band: Tuple[float, float] = (0.05, 0.45)    # 0..1 (rFFT正規化)
    width: Tuple[float, float] = (0.10, 0.35)   # 0..1 幅
    roll: float = 0.02                           # 0..1 スムーズロール
    restandardize: bool = True

@dataclass(frozen=True)
class TimeAugConfig:
    prob: float = 0.0
    factor_range: Tuple[float, float] = (0.95, 1.05)  # 伸縮率
    target_len: int | None = None                     # Noneなら入力Wを維持

@dataclass(frozen=True)
class SpaceAugConfig:
    prob: float = 0.0
    factor_range: Tuple[float, float] = (0.90, 1.10)  # トレース方向伸縮率