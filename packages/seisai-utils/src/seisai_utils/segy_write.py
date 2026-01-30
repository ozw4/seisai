from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import segyio


def _cast_like_dtype(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
	dt = np.dtype(dtype)

	if np.issubdtype(dt, np.floating):
		return np.asarray(x, dtype=dt)

	if np.issubdtype(dt, np.integer):
		info = np.iinfo(dt)
		y = np.rint(np.asarray(x, dtype=np.float64))
		y = np.clip(y, info.min, info.max)
		return y.astype(dt, copy=False)

	raise TypeError(f'unsupported dtype: {dt}')


def write_segy_like_input(
	*,
	src_path: str | Path,
	out_dir: str | Path,
	out_suffix: str,
	data_hw: np.ndarray,
	overwrite: bool = False,
	trace_indices: list[int] | None = None,
) -> Path:
	"""入力SEGYをコピーし、traceサンプルだけ差し替えたSEGYを出力する。

	- text/binary/trace header は変更しない（srcを丸ごとコピー→traceのみ上書き）
	- sample format は「コピーされたdstの形式」に従う（＝維持）
	- data_hw は (n_traces, n_samples) または trace_indices 指定時 (len(indices), n_samples)

	Args:
		src_path: 入力SEGY
		out_dir: 出力ディレクトリ
		out_suffix: 出力ファイル名の末尾（例: "_pred.sgy"）
		data_hw: 上書きしたい trace データ（H=trace, W=sample）
		overwrite: 出力ファイルが存在する場合に上書きするか
		trace_indices: 上書き対象のtrace index（0-based）。Noneなら全traceを上書き。

	Returns:
		dst_path: 出力SEGYパス

	"""
	src_path = Path(src_path)
	if not src_path.is_file():
		raise FileNotFoundError(f'src segy not found: {src_path}')

	if not isinstance(out_suffix, str) or len(out_suffix) == 0:
		raise ValueError('out_suffix must be non-empty str')

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	dst_path = out_dir / f'{src_path.stem}{out_suffix}'
	if dst_path.exists() and not overwrite:
		raise FileExistsError(f'output already exists: {dst_path}')

	data_hw = np.asarray(data_hw)
	if data_hw.ndim != 2:
		raise ValueError(f'data_hw must be (H,W), got {data_hw.shape}')

	# ヘッダ含め完全コピー
	shutil.copyfile(src_path, dst_path)

	in_f = segyio.open(str(src_path), 'r', ignore_geometry=True)
	out_f = segyio.open(str(dst_path), 'r+', ignore_geometry=True)

	n_traces = len(in_f.trace)
	if n_traces <= 0:
		raise ValueError('input segy has no traces')

	# 出力側のトレース dtype（sample format維持）
	out_sample_dtype = np.asarray(out_f.trace[0]).dtype

	# 入力のサンプル数（trace長）を基準に整合チェック
	tr0_in = np.asarray(in_f.trace.raw[0])
	n_samples = int(tr0_in.shape[0])
	if n_samples <= 0:
		raise ValueError('input segy has invalid trace length')

	if int(data_hw.shape[1]) != n_samples:
		raise ValueError(
			f'data_hw W must equal n_samples={n_samples}, got {int(data_hw.shape[1])}'
		)

	if trace_indices is None:
		if int(data_hw.shape[0]) != n_traces:
			raise ValueError(
				f'data_hw H must equal n_traces={n_traces}, got {int(data_hw.shape[0])}'
			)
		indices = list(range(n_traces))
	else:
		if not isinstance(trace_indices, list) or not all(
			isinstance(i, int) for i in trace_indices
		):
			raise TypeError('trace_indices must be list[int]')
		if len(trace_indices) == 0:
			raise ValueError('trace_indices must be non-empty')
		if len(set(trace_indices)) != len(trace_indices):
			raise ValueError('trace_indices must be unique')
		if min(trace_indices) < 0 or max(trace_indices) >= n_traces:
			raise ValueError(f'trace_indices out of range: valid=[0,{n_traces - 1}]')
		if int(data_hw.shape[0]) != len(trace_indices):
			raise ValueError(
				f'data_hw H must equal len(trace_indices)={len(trace_indices)}, got {int(data_hw.shape[0])}'
			)
		indices = trace_indices

	# 書き込み（ヘッダは触らない。traceサンプルだけ差し替え）
	for j, tr_idx in enumerate(indices):
		tr = _cast_like_dtype(data_hw[j], out_sample_dtype)
		out_f.trace[int(tr_idx)] = tr

	in_f.close()
	out_f.close()

	return dst_path


__all__ = [
	'write_segy_like_input',
]
