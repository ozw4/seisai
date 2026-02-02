from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

# 既存の operator / stack / plan は流用
from .builder import (
	FBGaussMap,
	IdentitySignal,
	MakeOffsetChannel,
	MakeTimeChannel,
	MaskedSignal,
	SelectStack,
)


# ---------------- 基本ユニット ----------------
class BuildUnit:
	"""一方(input か target どちらか)の組立ユニット。"""

	def __init__(self, ops: list[Callable], stack: SelectStack):
		self.ops = ops
		self.stack = stack

	def run(self, sample: dict[str, Any], rng=None) -> None:
		# 依存順にオペレータを流して最後にスタック
		for op in self.ops:
			op(sample, rng)
		self.stack(sample, rng)


# ---------------- レジストリ ----------------
@dataclass(frozen=True)
class RegItem:
	# factory(ctx) -> (callable op, produced_key)
	factory: Callable[[dict[str, Any]], tuple[Callable, str]]


def _need(ctx: dict[str, Any], name: str):
	if name not in ctx or ctx[name] is None:
		raise ValueError(f"ctx['{name}'] is required for this tag.")
	return ctx[name]


def make_registry(ctx: dict[str, Any] | None = None) -> dict[str, RegItem]:
	"""タグ→レシピのレジストリ。ctx には依存物を入れる(例: masker, fb_sigma, offset_normalize)。"""
	{} if ctx is None else dict(ctx)
	reg: dict[str, RegItem] = {}

	# 波形系
	reg['amplitude'] = RegItem(
		factory=lambda _c: (
			IdentitySignal(src='x_view', dst='x_amp', copy=False),
			'x_amp',
		)
	)
	reg['masked'] = RegItem(
		factory=lambda c: (
			MaskedSignal(_need(c, 'masker'), src='x_view', dst='x_masked'),
			'x_masked',
		)
	)
	reg['time_ch'] = RegItem(
		factory=lambda _c: (MakeTimeChannel(dst='time_ch'), 'time_ch')
	)
	reg['offset_ch'] = RegItem(
		factory=lambda c: (
			MakeOffsetChannel(
				dst='offset_ch', normalize=bool(c.get('offset_normalize', True))
			),
			'offset_ch',
		)
	)

	# ラベル系
	reg['fb_map'] = RegItem(
		factory=lambda c: (
			FBGaussMap(dst='fb_map', sigma=float(c.get('fb_sigma', 1.5))),
			'fb_map',
		)
	)

	return reg


# ---------------- ビルダー本体 ----------------
def builder(
	*,
	tags: Iterable[str],
	ctx: dict[str, Any] | None = None,
	dst: str,
	dtype=np.float32,
	to_torch: bool = True,
	registry: dict[str, RegItem] | None = None,
) -> BuildUnit:
	"""単独(input か target 片側)用のビルダー。
	- tags: 生成したいチャネルのタグ列(例: ["masked","time_ch"])
	- ctx : タグが必要とする依存物(input/target で別 dict を渡す)
	- dst : 最終的に格納するキー名("input"や"target"など)
	"""
	ctx = {} if ctx is None else dict(ctx)
	reg = registry or make_registry(ctx)
	tags = list(tags)

	# 未登録タグチェック
	unknown = [t for t in tags if t not in reg]
	if unknown:
		raise KeyError(f'Unknown tag(s): {unknown}. Available: {sorted(reg.keys())}')

	ops: list[Callable] = []
	produced_ops: dict[str, Callable] = {}
	keys: list[str] = []

	# タグ順にオペレータを構築(同じキーは一度だけ作る)
	for tag in tags:
		op, key = reg[tag].factory(ctx)
		if key not in produced_ops:
			produced_ops[key] = op
			ops.append(op)
		keys.append(key)

	stack = SelectStack(keys=keys, dst=dst, dtype=dtype, to_torch=to_torch)
	return BuildUnit(ops=ops, stack=stack)
