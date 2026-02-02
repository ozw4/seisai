from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import segyio

from .file_info import build_file_info

SortWithinGather = Literal['none', 'chno', 'offset']


@dataclass(frozen=True)
class FfidGather:
	"""FFID(=shot gather)単位で取り出した gather とメタ情報。"""

	file_idx: int
	file_path: str
	ffid: int
	trace_indices: np.ndarray  # (H,) int64 0-based trace index in file
	x_hw: np.ndarray  # (H,W) float32
	dt_sec: float


def _sort_indices_within_gather(
	*,
	info: dict,
	indices: np.ndarray,
	sort_within: SortWithinGather,
) -> np.ndarray:
	idx = np.asarray(indices, dtype=np.int64)
	if idx.size == 0:
		return idx

	if sort_within == 'none':
		return idx
	if sort_within == 'chno':
		key = np.asarray(info['chno_values'], dtype=np.int64)[idx]
	elif sort_within == 'offset':
		key = np.asarray(info['offsets'], dtype=np.float64)[idx]
	else:
		raise ValueError(f'unknown sort_within: {sort_within}')

	o = np.argsort(key, kind='mergesort')
	return idx[o]


class FFIDGatherIterator:
	"""SEG-Y を FFID ごとに順に取り出す iterator。

	- trace の並びは、デフォルトで chno で安定ソート(mergesort)
	- x_hw は float32 で返す(mmap -> gather slice)
	- caller は close() を呼ぶ(segyio ハンドルを保持する)
	"""

	def __init__(
		self,
		segy_files: Sequence[str],
		*,
		sort_within: SortWithinGather = 'chno',
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		cmp_byte=segyio.TraceField.CDP,
		use_header_cache: bool = True,
		header_cache_dir: str | None = None,
	) -> None:
		if len(segy_files) == 0:
			raise ValueError('segy_files must be non-empty')
		self.segy_files = [str(p) for p in segy_files]
		self.sort_within: SortWithinGather = sort_within
		self.ffid_byte = ffid_byte
		self.chno_byte = chno_byte
		self.cmp_byte = cmp_byte
		self.use_header_cache = bool(use_header_cache)
		self.header_cache_dir = header_cache_dir

		self._file_infos: list[dict] = []
		for p in self.segy_files:
			if not Path(p).is_file():
				raise FileNotFoundError(f'SEG-Y not found: {p}')
			info = build_file_info(
				p,
				ffid_byte=self.ffid_byte,
				chno_byte=self.chno_byte,
				cmp_byte=self.cmp_byte,
				header_cache_dir=self.header_cache_dir,
				use_header_cache=self.use_header_cache,
				include_centroids=False,
			)
			if info['ffid_key_to_indices'] is None:
				raise ValueError(f'ffid_key_to_indices is None for: {p}')
			self._file_infos.append(info)

	def close(self) -> None:
		for info in self._file_infos:
			f = info.get('segy_obj')
			if f is not None:
				f.close()

	def __enter__(self) -> FFIDGatherIterator:
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def __iter__(self) -> Iterator[FfidGather]:
		return self.iter_gathers()

	def iter_gathers(
		self,
		*,
		ffids: Sequence[int] | None = None,
		file_indices: Sequence[int] | None = None,
	) -> Iterator[FfidGather]:
		"""FFID gather を順に yield。

		Args:
			ffids: 指定した FFID のみを処理(None なら全 FFID)
			file_indices: 指定したファイル index のみを処理(None なら全ファイル)

		"""
		ffid_set = None
		if ffids is not None:
			ffid_set = {int(x) for x in ffids}

		if file_indices is None:
			fis = range(len(self._file_infos))
		else:
			fis = [int(i) for i in file_indices]
			if len(fis) == 0:
				raise ValueError('file_indices must be non-empty')
			if min(fis) < 0 or max(fis) >= len(self._file_infos):
				raise ValueError('file_indices out of range')

		for fi in fis:
			info = self._file_infos[int(fi)]
			key_to_idx = info['ffid_key_to_indices']
			assert key_to_idx is not None

			keys = info['ffid_unique_keys']
			if keys is None:
				keys = list(key_to_idx.keys())

			for k in keys:
				ffid = int(k)
				if ffid_set is not None and ffid not in ffid_set:
					continue

				idx0 = key_to_idx[ffid]
				idx = _sort_indices_within_gather(
					info=info,
					indices=idx0,
					sort_within=self.sort_within,
				)
				x_hw = info['mmap'][idx].astype(np.float32)
				yield FfidGather(
					file_idx=int(fi),
					file_path=str(info['path']),
					ffid=int(ffid),
					trace_indices=idx.astype(np.int64, copy=False),
					x_hw=x_hw,
					dt_sec=float(info['dt_sec']),
				)


__all__ = [
	'FFIDGatherIterator',
	'FfidGather',
]
