from dataclasses import dataclass


@dataclass(frozen=True)
class FreqAugConfig:
    prob: float = 0.0
    kinds: tuple[str, ...] = ('bandpass', 'lowpass', 'highpass')
    band: tuple[float, float] = (0.05, 0.45)  # 0..1 (rFFT正規化)
    width: tuple[float, float] = (0.10, 0.35)  # 0..1 幅
    roll: float = 0.02  # 0..1 スムーズロール
    restandardize: bool = True


@dataclass(frozen=True)
class TimeAugConfig:
    prob: float = 0.0
    factor_range: tuple[float, float] = (0.95, 1.05)  # 伸縮率


@dataclass(frozen=True)
class SpaceAugConfig:
    prob: float = 0.0
    factor_range: tuple[float, float] = (0.90, 1.10)  # トレース方向伸縮率
