from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SegmentationHead2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
		)

	def forward(self, x):
		return self.conv(x)


class SegClsHead2d(nn.Module):
	"""判別(分類)タスク向け 2D セグメンテーションヘッド。
	入力:  x (B, C_in, H, W)
	出力:  y (B, num_classes, H_out, W_out)
	  - 二値: num_classes=1 + activation='sigmoid'
	  - 多クラス: num_classes>=2 + activation='softmax' もしくはロジット返却
	"""

	def __init__(
		self,
		in_channels: int,
		num_classes: int,
		*,
		mid_channels: int | None = None,
		use_bn: bool = True,
		dropout: float = 0.0,
		activation: str | None = None,  # None | 'sigmoid' | 'softmax'
		upsample_to: tuple[int, int] | None = None,  # 出力サイズを固定したい場合
		align_corners: bool = False,
	) -> None:
		super().__init__()
		if in_channels <= 0 or num_classes <= 0:
			raise ValueError('in_channels と num_classes は正の整数が必要')
		if dropout < 0.0:
			raise ValueError('dropout は 0 以上が必要')
		if activation not in (None, 'sigmoid', 'softmax'):
			raise ValueError("activation は None|'sigmoid'|'softmax' のいずれか")

		c_mid = mid_channels if mid_channels is not None else in_channels
		bias1 = not use_bn

		self.proj = nn.Conv2d(in_channels, c_mid, kernel_size=3, padding=1, bias=bias1)
		self.bn = nn.BatchNorm2d(c_mid) if use_bn else nn.Identity()
		self.act = nn.ReLU(inplace=True)
		self.drop = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()
		self.classifier = nn.Conv2d(c_mid, num_classes, kernel_size=1, bias=True)

		self.activation = activation
		self.upsample_to = upsample_to
		self.align_corners = bool(align_corners)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.dim() != 4:
			raise ValueError(
				f'x は (B,C,H,W) である必要があります: got {tuple(x.shape)}'
			)
		h = self.proj(x)
		h = self.bn(h)
		h = self.act(h)
		h = self.drop(h)
		y = self.classifier(h)

		if self.upsample_to is not None:
			y = F.interpolate(
				y,
				size=self.upsample_to,
				mode='bilinear',
				align_corners=self.align_corners,
			)

		if self.activation == 'sigmoid':  # 二値
			return torch.sigmoid(y)
		if self.activation == 'softmax':  # 多クラス
			return torch.softmax(y, dim=1)
		return y  # ロジット
