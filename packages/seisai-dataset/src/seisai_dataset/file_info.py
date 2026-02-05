from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import segyio

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FileInfo:
	path: str
	mmap: np.ndarray
	segy_obj: segyio.SegyFile
	dt_sec: float
	n_traces: int
	n_samples: int
	ffid_values: np.ndarray
	chno_values: np.ndarray
	cmp_values: np.ndarray | None
	ffid_key_to_indices: dict[int, np.ndarray] | None
	chno_key_to_indices: dict[int, np.ndarray] | None
	cmp_key_to_indices: dict[int, np.ndarray] | None
	ffid_unique_keys: list[int] | None
	chno_unique_keys: list[int] | None
	cmp_unique_keys: list[int] | None
	offsets: np.ndarray
	ffid_centroids: dict[int, tuple[float, float]] | None
	chno_centroids: dict[int, tuple[float, float]] | None
	fb: np.ndarray | None = None
	# Optional CSR phase picks (PhaseNet-style training). Stored as raw arrays to keep
	# file_info pickle/lightweight and avoid carrying variable-length lists in outputs.
	p_indptr: np.ndarray | None = None
	p_data: np.ndarray | None = None
	s_indptr: np.ndarray | None = None
	s_data: np.ndarray | None = None

	def __getitem__(self, key: str):
		return getattr(self, key)

	def __setitem__(self, key: str, value) -> None:
		setattr(self, key, value)

	def get(self, key: str, default=None):
		return getattr(self, key, default)


@dataclass(slots=True)
class PairFileInfo:
	input_info: FileInfo
	target_path: str
	target_mmap: np.ndarray
	target_segy_obj: segyio.SegyFile
	target_n_samples: int
	target_n_traces: int
	target_dt_sec: float


def load_headers_with_cache(
	segy_path: str,
	ffid_byte,
	chno_byte,
	cmp_byte=None,
	cache_dir: str | None = None,
	rebuild: bool = False,
):
	"""Load SEG-Y header fields with a small npz cache.
	Logs via `logging` (no print/warnings).
	"""
	segy_p = Path(segy_path)
	cache_p = (
		Path(cache_dir) / (segy_p.name + '.headers.npz')
		if cache_dir
		else segy_p.with_suffix(segy_p.suffix + '.headers.npz')
	)

	# use cache if exists and newer than segy
	try:
		if (
			(not rebuild)
			and cache_p.exists()
			and cache_p.stat().st_mtime >= segy_p.stat().st_mtime
		):
			z = np.load(cache_p, allow_pickle=False)
			meta = {
				'ffid_values': z['ffid_values'],
				'chno_values': z['chno_values'],
				'cmp_values': (z['cmp_values'] if 'cmp_values' in z.files else None),
				'offsets': z['offsets'],
				'dt_us': int(z['dt_us']),
				'n_traces': int(z['n_traces']),
				'n_samples': int(z['n_samples']),
			}
			logger.info('Loaded header cache: %s', cache_p)
			return meta
	except Exception:
		# broken cache etc. → rebuild
		logger.warning(
			'Failed to load header cache (will rebuild): %s', cache_p, exc_info=True
		)

	# rebuild from segy
	with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
		ffid_values = np.asarray(f.attributes(ffid_byte)[:], dtype=np.int32)
		chno_values = np.asarray(f.attributes(chno_byte)[:], dtype=np.int32)
		cmp_values = None
		if cmp_byte is not None:
			try:
				cmp_values = np.asarray(f.attributes(cmp_byte)[:], dtype=np.int32)
			except Exception:
				logger.warning(
					'Failed to read CMP values: %s', segy_path, exc_info=True
				)
				cmp_values = None

		try:
			offsets = np.asarray(
				f.attributes(segyio.TraceField.offset)[:], dtype=np.float32
			)
			if len(offsets) != f.tracecount:
				logger.warning(
					'Offset length mismatch in %s (got %d, want %d). Zero-filling.',
					segy_path,
					len(offsets),
					f.tracecount,
				)
				offsets = np.zeros(f.tracecount, dtype=np.float32)
		except Exception:
			logger.warning(
				'Failed to read offsets from %s (zero-filling).',
				segy_path,
				exc_info=True,
			)
			offsets = np.zeros(f.tracecount, dtype=np.float32)

		dt_us = int(f.bin[segyio.BinField.Interval])
		meta = {
			'ffid_values': ffid_values,
			'chno_values': chno_values,
			'cmp_values': (
				cmp_values if cmp_values is not None else np.array([], dtype=np.int32)
			),
			'offsets': offsets,
			'dt_us': dt_us,
			'n_traces': f.tracecount,
			'n_samples': f.samples.size if f.samples is not None else 0,
		}

	# save cache (atomic-ish replace)
	try:
		tmp = cache_p.with_name(cache_p.stem + '.tmp' + cache_p.suffix)
		np.savez_compressed(tmp, **meta)
		tmp.replace(cache_p)
		logger.info('Saved header cache: %s', cache_p)
	except Exception:
		logger.warning('Failed to save header cache: %s', cache_p, exc_info=True)

	# normalize return
	meta['cmp_values'] = (
		None
		if (isinstance(meta['cmp_values'], np.ndarray) and meta['cmp_values'].size == 0)
		else meta['cmp_values']
	)
	return meta


def build_index_map(arr: np.ndarray | None) -> dict[int, np.ndarray] | None:
	"""1次元配列 arr から {値 -> その値を持つトレースのインデックス配列(int32)} を作る。
	- arr が None のときは None を返す
	- 安定性のため mergesort を使用
	- 戻り値の各インデックス配列は dtype=int32(GPU転送やシリアライズで軽量)
	"""
	if arr is None:
		return None
	a = np.asarray(arr)
	if a.ndim != 1:
		a = a.reshape(-1)  # 厳密に1Dへ
	uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
	order = np.argsort(inv, kind='mergesort')
	splits = np.cumsum(counts)[:-1]
	groups = np.split(order, splits)
	return {int(k): g.astype(np.int32) for k, g in zip(uniq, groups, strict=False)}


def _build_centroids(
	key_to_indices: dict[int, np.ndarray] | None,
	x: np.ndarray | None,
	y: np.ndarray | None,
) -> dict[int, tuple[float, float]] | None:
	"""キーごとの (mean(x), mean(y)) を返す。入力が欠けていれば None を返す。"""
	if key_to_indices is None or x is None or y is None:
		return None
	out: dict[int, tuple[float, float]] = {}
	for k, idx in key_to_indices.items():
		if idx.size == 0:
			continue
		xm = float(np.mean(x[idx]))
		ym = float(np.mean(y[idx]))
		out[int(k)] = (xm, ym)
	return out if out else None


def build_file_info(
	segy_path: str,
	*,
	ffid_byte,
	chno_byte,
	cmp_byte=None,
	header_cache_dir: str | None = None,
	use_header_cache: bool = True,
	include_centroids: bool = False,
) -> dict:
	"""SEG-Y 1ファイルから dataset が要求する file_info dict を構築する共通関数。

	- ヘッダは load_headers_with_cache() を用いて取得(cache_dir の有無で自動切替)
	- インデックスマップ({値→行インデックス配列})を安定ソートで構築
	- mmap を開いた segyio オブジェクトを保持(caller が close() を呼ぶこと)
	- include_centroids=True の場合、座標が読めたときのみキーごとのセントロイドを付与
	(座標が欠落/不一致なら警告を出して None を格納)

	Returns (dict):
	path, mmap, segy_obj, dt_sec, n_traces, n_samples,
	ffid_values/chno_values/cmp_values,
	ffid_key_to_indices/chno_key_to_indices/cmp_key_to_indices,
	ffid_unique_keys/chno_unique_keys/cmp_unique_keys,
	offsets, ffid_centroids/chno_centroids(任意)
	"""
	segy_path = str(segy_path)
	if not Path(segy_path).exists():
		raise FileNotFoundError(f'SEG-Y not found: {segy_path}')

	meta = load_headers_with_cache(
		segy_path,
		ffid_byte,
		chno_byte,
		cmp_byte,
		cache_dir=(header_cache_dir if use_header_cache else None),
		rebuild=False,
	)
	ffid_values = meta['ffid_values']
	chno_values = meta['chno_values']
	cmp_values = meta['cmp_values']  # None あり得る
	offsets = np.asarray(meta['offsets'], dtype=np.float32)
	dt_us = int(meta['dt_us'])
	n_traces = int(meta['n_traces'])
	n_samples = int(meta['n_samples'])
	dt_sec = dt_us * 1e-6

	# mmap 用に開いて保持(caller が close する)
	f = segyio.open(segy_path, 'r', ignore_geometry=True)
	mmap = f.trace.raw[:]

	# index maps(安定ソート)と unique_keys
	ffid_key_to_indices = build_index_map(ffid_values)
	chno_key_to_indices = build_index_map(chno_values)
	cmp_key_to_indices = (
		build_index_map(cmp_values) if (cmp_values is not None) else None
	)

	ffid_unique_keys = list(ffid_key_to_indices.keys()) if ffid_key_to_indices else None
	chno_unique_keys = list(chno_key_to_indices.keys()) if chno_key_to_indices else None
	cmp_unique_keys = list(cmp_key_to_indices.keys()) if cmp_key_to_indices else None

	ffid_centroids = None
	chno_centroids = None

	if include_centroids:
		# 座標読み出し(失敗/不一致時は警告して None)
		try:
			srcx = np.asarray(
				f.attributes(segyio.TraceField.SourceX)[:], dtype=np.float64
			)
			srcy = np.asarray(
				f.attributes(segyio.TraceField.SourceY)[:], dtype=np.float64
			)
			grx = np.asarray(
				f.attributes(segyio.TraceField.GroupX)[:], dtype=np.float64
			)
			gry = np.asarray(
				f.attributes(segyio.TraceField.GroupY)[:], dtype=np.float64
			)

			scal = np.asarray(
				f.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=np.float64
			)
			# SEG-Y の SourceGroupScalar: 負値は分母(1/|scal|)
			scal_arr = np.asarray(scal, dtype=np.float64)
			scal_eff = np.ones_like(scal_arr, dtype=np.float64)

			pos = scal_arr > 0.0
			neg = scal_arr < 0.0

			scal_eff[pos] = scal_arr[pos]
			scal_eff[neg] = 1.0 / np.abs(scal_arr[neg])

			if scal_eff.size not in (1, srcx.size):
				msg = 'SourceGroupScalar size mismatch'
				raise ValueError(msg)
			if scal_eff.size == 1:
				s = float(scal_eff.reshape(()))
				srcx *= s
				srcy *= s
				grx *= s
				gry *= s
			else:
				srcx *= scal_eff
				srcy *= scal_eff
				grx *= scal_eff
				gry *= scal_eff

			ffid_centroids = _build_centroids(ffid_key_to_indices, srcx, srcy)
			chno_centroids = _build_centroids(chno_key_to_indices, grx, gry)
		except Exception as e:
			logger.warning(
				'centroids disabled (coordinate read failed): %s', e, exc_info=True
			)
			ffid_centroids = None
			chno_centroids = None

	return {
		'path': segy_path,
		'mmap': mmap,
		'segy_obj': f,
		'dt_sec': dt_sec,
		'n_traces': n_traces,
		'n_samples': n_samples,
		'ffid_values': ffid_values,
		'chno_values': chno_values,
		'cmp_values': cmp_values
		if isinstance(cmp_values, np.ndarray) and cmp_values.size > 0
		else None,
		'ffid_key_to_indices': ffid_key_to_indices,
		'chno_key_to_indices': chno_key_to_indices,
		'cmp_key_to_indices': cmp_key_to_indices,
		'ffid_unique_keys': ffid_unique_keys,
		'chno_unique_keys': chno_unique_keys,
		'cmp_unique_keys': cmp_unique_keys,
		'offsets': offsets,
		'ffid_centroids': ffid_centroids,
		'chno_centroids': chno_centroids,
	}


def build_file_info_dataclass(
	segy_path: str,
	*,
	ffid_byte,
	chno_byte,
	cmp_byte=None,
	header_cache_dir: str | None = None,
	use_header_cache: bool = True,
	include_centroids: bool = False,
) -> FileInfo:
	info = build_file_info(
		segy_path,
		ffid_byte=ffid_byte,
		chno_byte=chno_byte,
		cmp_byte=cmp_byte,
		header_cache_dir=header_cache_dir,
		use_header_cache=use_header_cache,
		include_centroids=include_centroids,
	)
	return FileInfo(**info)
