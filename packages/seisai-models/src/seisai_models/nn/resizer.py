# resizer.py
# 出力サイズポリシーを切り替え可能な薄いユニット。
# - policy: "match_input" | "native" | "fixed" | "scale"
# - data_kind: "continuous" | "discrete" で補間法を自動選択
#     * continuous: 連続値(回帰/確率/ロジット等)→ bilinear(align_corners=False)
#     * discrete  : 離散ラベル(クラスID等)       → nearest
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

Interpolation = Literal['nearest', 'bilinear', 'bicubic']
Policy = Literal['match_input', 'native', 'fixed', 'scale']
DataKind = Literal['continuous', 'discrete']


def _to_hw(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return int(x), int(x)
    assert isinstance(x, tuple)
    assert len(x) == 2
    return int(x[0]), int(x[1])


def _to_scale(x: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(x, (int, float)):
        v = float(x)
        assert v > 0.0
        return v, v
    assert isinstance(x, tuple)
    assert len(x) == 2
    sx, sy = float(x[0]), float(x[1])
    assert sx > 0.0
    assert sy > 0.0
    return sx, sy


class Resizer(nn.Module):
    """出力サイズを制御する薄い後処理ユニット。
    - 入力は 4D (B, C, H, W) を前提。
    - policy に応じて目標サイズを決定し、F.interpolate でリサイズ。
    - data_kind:
        * "continuous": デフォルト補間 'bilinear'(align_corners=False, 勾配可)
        * "discrete"  : デフォルト補間 'nearest'(離散ラベル保持).
    """

    def __init__(
        self,
        *,
        policy: Policy = 'match_input',
        data_kind: DataKind = 'continuous',
        interp: Interpolation | None = None,
        align_corners: bool = False,
        antialias: bool = False,
        # policy 固有パラメータ
        fixed_size: tuple[int, int] | None = None,  # (H_out, W_out)
        scale: float | tuple[float, float] | None = None,  # 例: 0.5 or (0.5, 0.25)
    ) -> None:
        super().__init__()
        assert policy in ('match_input', 'native', 'fixed', 'scale')

        # 後方互換なし: 旧名が来たら即失敗
        if isinstance(data_kind, str) and data_kind in ('logits', 'labels'):
            msg = "data_kind は 'continuous' / 'discrete' を使用してください('logits'/'labels' は廃止)"
            raise AssertionError(
                msg
            )
        assert data_kind in ('continuous', 'discrete')
        self.policy: Policy = policy
        self.data_kind: DataKind = data_kind

        # 既定の補間モードを data_kind で決定(手動指定があればそれを優先)
        if interp is None:
            self.interp: Interpolation = (
                'bilinear' if data_kind == 'continuous' else 'nearest'
            )
        else:
            assert interp in ('nearest', 'bilinear', 'bicubic')
            self.interp = interp

        # bilinear/bicubic の align_corners は False 推奨
        self.align_corners: bool = (
            bool(align_corners) if self.interp in ('bilinear', 'bicubic') else False
        )
        self.antialias: bool = (
            bool(antialias) if self.interp in ('bilinear', 'bicubic') else False
        )

        # 固有パラメータ検証
        if self.policy == 'fixed':
            assert fixed_size is not None
            H, W = _to_hw(fixed_size)
            assert H > 0
            assert W > 0
            self.fixed_size = (H, W)
        else:
            self.fixed_size = None

        if self.policy == 'scale':
            assert scale is not None
            self.scale_hw = _to_scale(scale)  # (sH, sW)
        else:
            self.scale_hw = None

    @staticmethod
    def _hw(t: torch.Tensor) -> tuple[int, int]:
        assert t.dim() == 4, 'Resizerは4Dテンソル (B,C,H,W) を前提とする'
        return int(t.shape[-2]), int(t.shape[-1])

    def _target_size(
        self,
        y: torch.Tensor,
        *,
        ref: torch.Tensor | None = None,
    ) -> tuple[int, int] | None:
        """目標サイズを返す。None のときはリサイズしない(native)。."""
        if self.policy == 'native':
            return None

        if self.policy == 'match_input':
            assert ref is not None, "policy='match_input' は ref を要する"
            return self._hw(ref)

        H, W = self._hw(y)

        if self.policy == 'fixed':
            return self.fixed_size

        if self.policy == 'scale':
            sh, sw = self.scale_hw  # float
            # 端数は四捨五入で整数化。0 は許容しない。
            th = max(1, round(H * sh))
            tw = max(1, round(W * sw))
            return (th, tw)

        msg = f'unknown policy: {self.policy}'
        raise AssertionError(msg)

    def forward(
        self, y: torch.Tensor, *, ref: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert isinstance(y, torch.Tensor)
        Htgt = self._target_size(y, ref=ref)
        if Htgt is None:
            return y  # native

        mode = self.interp
        kwargs = {}
        if mode in ('bilinear', 'bicubic'):
            kwargs['align_corners'] = self.align_corners
            kwargs['antialias'] = self.antialias
        return F.interpolate(y, size=Htgt, mode=mode, **kwargs)


# 便利関数(continuous/discreteで補間法を固定)
def resize_continuous(
    y: torch.Tensor,
    *,
    policy: Policy = 'match_input',
    ref: torch.Tensor | None = None,
    fixed_size: tuple[int, int] | None = None,
    scale: float | tuple[float, float] | None = None,
    antialias: bool = False,
) -> torch.Tensor:
    r = Resizer(
        policy=policy,
        data_kind='continuous',
        fixed_size=fixed_size,
        scale=scale,
        antialias=antialias,
    )
    return r(y, ref=ref)


def resize_discrete(
    y: torch.Tensor,
    *,
    policy: Policy = 'match_input',
    ref: torch.Tensor | None = None,
    fixed_size: tuple[int, int] | None = None,
    scale: float | tuple[float, float] | None = None,
) -> torch.Tensor:
    r = Resizer(policy=policy, data_kind='discrete', fixed_size=fixed_size, scale=scale)
    return r(y, ref=ref)
