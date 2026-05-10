"""Thin entrypoint for fbpick physics QC summary CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from seisai_engine.pipelines.fbpick.common.qc_gathers import (
	iter_qc_gathers,
	sort_gather_indices_for_qc,
)
from seisai_engine.pipelines.fbpick.common.path_naming import (
	build_fbpick_tag,
	build_final_npz_name,
)
from seisai_engine.pipelines.fbpick.physics.physical_center import (
	PHYSICAL_MODEL_FAILURE_LABELS,
	PHYSICAL_MODEL_STATUS_LABELS,
)

__all__ = ['main', 'run_pipeline']

DT_SEC_ATOL = 1e-9


OPTIONAL_SUMMARY_METRIC_COLUMNS = [
	'fine_center_R32',
	'fine_center_R64',
	'fine_center_R127',
	'fine_center_abs_err_median',
	'fine_center_abs_err_p90',
	'fine_center_abs_err_p95',
	'fine_center_delta_p90_vs_robust',
	'fine_center_delta_p95_vs_robust',
	'physical_center_R127',
	'physical_center_abs_err_p90',
	'physical_center_abs_err_p95',
	'gt_in_actual_window_rate',
	'final_pick_valid_rate',
	'final_pick_R32',
	'final_pick_R64',
	'final_pick_R127',
	'final_pick_abs_err_median',
	'final_pick_abs_err_p90',
	'final_pick_abs_err_p95',
]


PER_FILE_COLUMNS = [
	'scope',
	'segy_path',
	'fb_path',
	'coarse_npz_path',
	'robust_npz_path',
	'fine_ready',
	'robust_ready',
	'fine_center_ready',
	'actual_window_ready',
	'n_traces',
	'n_valid_gt',
	'n_invalid_gt',
	'n_samples_orig',
	'dt_sec',
	'R32',
	'R64',
	'R127',
	'coarse_abs_err_median',
	'coarse_abs_err_p90',
	'coarse_abs_err_p95',
	'robust_abs_err_median',
	'robust_abs_err_p90',
	'robust_abs_err_p95',
	'delta_p90',
	'delta_p95',
	*OPTIONAL_SUMMARY_METRIC_COLUMNS,
	'physical_model_status_counts',
	'physical_model_failure_reason_counts',
]

GLOBAL_COLUMNS = [
	'scope',
	'fine_ready',
	'robust_ready',
	'fine_center_ready',
	'actual_window_ready',
	'n_files',
	'n_traces',
	'n_valid_gt',
	'n_invalid_gt',
	'R32',
	'R64',
	'R127',
	'coarse_abs_err_median',
	'coarse_abs_err_p90',
	'coarse_abs_err_p95',
	'robust_abs_err_median',
	'robust_abs_err_p90',
	'robust_abs_err_p95',
	'delta_p90',
	'delta_p95',
	*OPTIONAL_SUMMARY_METRIC_COLUMNS,
	'physical_model_status_counts',
	'physical_model_failure_reason_counts',
]


def _load_runtime() -> SimpleNamespace:
	import segyio
	from seisai_dataset.file_info import build_file_info
	from seisai_engine.pipelines.common import load_cfg_with_base_dir, resolve_cfg_paths
	from seisai_engine.pipelines.fbpick.common import (
		load_coarse_npz,
		load_fbpick_final_npz,
		load_robust_npz,
	)
	from seisai_engine.viewer.fbpick import (
		save_fbpick_physics_qc_cdf_png,
		save_fbpick_physics_qc_gather_png,
	)
	from seisai_utils.listfiles import expand_cfg_listfiles

	return SimpleNamespace(
		build_file_info=build_file_info,
		expand_cfg_listfiles=expand_cfg_listfiles,
		load_cfg_with_base_dir=load_cfg_with_base_dir,
		load_coarse_npz=load_coarse_npz,
		load_fbpick_final_npz=load_fbpick_final_npz,
		load_robust_npz=load_robust_npz,
		save_fbpick_physics_qc_cdf_png=save_fbpick_physics_qc_cdf_png,
		save_fbpick_physics_qc_gather_png=save_fbpick_physics_qc_gather_png,
		resolve_cfg_paths=resolve_cfg_paths,
		segyio=segyio,
	)


def _require_dict(cfg: dict[str, Any], key: str) -> dict[str, Any]:
	value = cfg.get(key)
	if not isinstance(value, dict):
		msg = f'{key} must be dict'
		raise TypeError(msg)
	return value


def _require_str(cfg: dict[str, Any], key: str) -> str:
	value = cfg.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{key} must be non-empty str'
		raise TypeError(msg)
	return value


def _require_list_str(cfg: dict[str, Any], key: str) -> list[str]:
	value = cfg.get(key)
	if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
		msg = f'{key} must be list[str]'
		raise TypeError(msg)
	return list(value)


def _optional_str(cfg: dict[str, Any], key: str) -> str | None:
	value = cfg.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{key} must be non-empty str or null'
		raise TypeError(msg)
	return value


def _optional_list_str(cfg: dict[str, Any], key: str) -> list[str] | None:
	value = cfg.get(key)
	if value is None:
		return None
	if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
		msg = f'{key} must be list[str] or null'
		raise TypeError(msg)
	if not value:
		msg = f'{key} must be non-empty when specified'
		raise ValueError(msg)
	return list(value)


def _prepare_cfg(
	cfg: dict[str, Any],
	*,
	base_dir: Path,
	runtime: SimpleNamespace,
) -> dict[str, Any]:
	paths = cfg.get('paths')
	optional_listfile_keys: list[str] = []
	if isinstance(paths, dict) and paths.get('final_npz_files') is not None:
		optional_listfile_keys.append('paths.final_npz_files')
	runtime.expand_cfg_listfiles(
		cfg,
		keys=[
			'paths.segy_files',
			'paths.fb_files',
			*optional_listfile_keys,
		],
	)
	path_keys = [
		'paths.segy_files',
		'paths.fb_files',
		'paths.coarse_npz_dir',
		'paths.robust_npz_dir',
		'paths.out_dir',
	]
	if isinstance(paths, dict):
		if paths.get('final_npz_dir') is not None:
			path_keys.append('paths.final_npz_dir')
		if paths.get('final_npz_files') is not None:
			path_keys.append('paths.final_npz_files')
	runtime.resolve_cfg_paths(
		cfg,
		base_dir,
		keys=path_keys,
	)
	return cfg


def _build_tag(segy_path: str | Path) -> str:
	return build_fbpick_tag(segy_path)


def _build_coarse_npz_path(
	*, segy_path: str | Path, coarse_npz_dir: str | Path
) -> Path:
	return Path(coarse_npz_dir) / (_build_tag(segy_path) + '.coarse.npz')


def _build_robust_npz_path(
	*, segy_path: str | Path, robust_npz_dir: str | Path
) -> Path:
	return Path(robust_npz_dir) / (_build_tag(segy_path) + '.robust.npz')


def _build_final_npz_path(
	*, segy_path: str | Path, final_npz_dir: str | Path
) -> Path:
	return Path(final_npz_dir) / build_final_npz_name(segy_path)


def _build_legacy_final_npz_path(
	*, segy_path: str | Path, final_npz_dir: str | Path
) -> Path:
	return Path(final_npz_dir) / f'{Path(segy_path).stem}.fbpick_final.npz'


def _validate_paths(
	cfg: dict[str, Any],
) -> tuple[
	list[str],
	list[str],
	Path,
	Path,
	Path | None,
	list[str] | None,
	Path,
]:
	paths = _require_dict(cfg, 'paths')
	segy_files = _require_list_str(paths, 'segy_files')
	fb_files = _require_list_str(paths, 'fb_files')
	if len(segy_files) != len(fb_files):
		msg = 'paths.segy_files and paths.fb_files must have the same length'
		raise ValueError(msg)
	if not segy_files:
		msg = 'paths.segy_files must contain at least one entry'
		raise ValueError(msg)

	coarse_npz_dir = Path(_require_str(paths, 'coarse_npz_dir'))
	robust_npz_dir = Path(_require_str(paths, 'robust_npz_dir'))
	final_npz_dir_raw = _optional_str(paths, 'final_npz_dir')
	final_npz_dir = Path(final_npz_dir_raw) if final_npz_dir_raw is not None else None
	final_npz_files = _optional_list_str(paths, 'final_npz_files')
	if final_npz_files is not None and len(final_npz_files) != len(segy_files):
		msg = 'paths.final_npz_files must match paths.segy_files length'
		raise ValueError(msg)
	out_dir = Path(_require_str(paths, 'out_dir'))
	return (
		segy_files,
		fb_files,
		coarse_npz_dir,
		robust_npz_dir,
		final_npz_dir,
		final_npz_files,
		out_dir,
	)


def _load_dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
	dataset = _require_dict(cfg, 'dataset')
	primary_keys = dataset.get('primary_keys', ['ffid'])
	if not isinstance(primary_keys, list) or not all(
		isinstance(item, str) for item in primary_keys
	):
		msg = 'dataset.primary_keys must be list[str]'
		raise TypeError(msg)
	if not primary_keys:
		msg = 'dataset.primary_keys must not be empty'
		raise ValueError(msg)

	infer_endian = dataset.get('infer_endian', 'big')
	if not isinstance(infer_endian, str):
		msg = 'dataset.infer_endian must be str'
		raise TypeError(msg)

	use_header_cache = dataset.get('use_header_cache', True)
	if not isinstance(use_header_cache, bool):
		msg = 'dataset.use_header_cache must be bool'
		raise TypeError(msg)

	return {
		'primary_keys': primary_keys,
		'infer_endian': infer_endian,
		'use_header_cache': use_header_cache,
	}


def _load_vis_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
	vis = cfg.get('vis', {})
	if vis is None:
		vis = {}
	if not isinstance(vis, dict):
		msg = 'vis must be dict'
		raise TypeError(msg)

	max_gathers_per_file = vis.get('max_gathers_per_file', 8)
	if isinstance(max_gathers_per_file, bool) or not isinstance(
		max_gathers_per_file, int
	):
		msg = 'vis.max_gathers_per_file must be int'
		raise TypeError(msg)
	if int(max_gathers_per_file) < 0:
		msg = 'vis.max_gathers_per_file must be >= 0'
		raise ValueError(msg)

	save_cdf = vis.get('save_cdf', False)
	if not isinstance(save_cdf, bool):
		msg = 'vis.save_cdf must be bool'
		raise TypeError(msg)

	save_summary_csv = vis.get('save_summary_csv', True)
	if not isinstance(save_summary_csv, bool):
		msg = 'vis.save_summary_csv must be bool'
		raise TypeError(msg)

	waveform_norm = vis.get('waveform_norm', 'global')
	if not isinstance(waveform_norm, str):
		msg = 'vis.waveform_norm must be str'
		raise TypeError(msg)
	if waveform_norm not in ('global', 'per_trace'):
		msg = 'vis.waveform_norm must be one of: global, per_trace'
		raise ValueError(msg)

	clip_percentile_raw = vis.get('clip_percentile', 99.0)
	if isinstance(clip_percentile_raw, bool) or not isinstance(
		clip_percentile_raw, (int, float)
	):
		msg = 'vis.clip_percentile must be float'
		raise TypeError(msg)
	clip_percentile = float(clip_percentile_raw)
	if (
		not np.isfinite(clip_percentile)
		or clip_percentile <= 0.0
		or clip_percentile > 100.0
	):
		msg = 'vis.clip_percentile must lie in (0, 100]'
		raise ValueError(msg)

	skip_gather_keys_raw = vis.get('skip_gather_keys', {})
	if not isinstance(skip_gather_keys_raw, dict) or not all(
		isinstance(key, str) for key in skip_gather_keys_raw
	):
		msg = 'vis.skip_gather_keys must be dict[str, list[int]]'
		raise TypeError(msg)
	skip_gather_keys: dict[str, set[int]] = {}
	for primary_key, values in skip_gather_keys_raw.items():
		if not isinstance(values, list) or not all(
			isinstance(item, int) and not isinstance(item, bool) for item in values
		):
			msg = 'vis.skip_gather_keys must be dict[str, list[int]]'
			raise TypeError(msg)
		skip_gather_keys[primary_key] = {int(item) for item in values}

	max_traces_per_gather_raw = vis.get('max_traces_per_gather', 10000)
	if max_traces_per_gather_raw is None:
		max_traces_per_gather = None
	elif isinstance(max_traces_per_gather_raw, bool) or not isinstance(
		max_traces_per_gather_raw, int
	):
		msg = 'vis.max_traces_per_gather must be int > 0 or null'
		raise TypeError(msg)
	elif int(max_traces_per_gather_raw) <= 0:
		msg = 'vis.max_traces_per_gather must be int > 0 or null'
		raise ValueError(msg)
	else:
		max_traces_per_gather = int(max_traces_per_gather_raw)

	return {
		'max_gathers_per_file': int(max_gathers_per_file),
		'save_cdf': bool(save_cdf),
		'save_summary_csv': bool(save_summary_csv),
		'waveform_norm': waveform_norm,
		'clip_percentile': clip_percentile,
		'skip_gather_keys': skip_gather_keys,
		'max_traces_per_gather': max_traces_per_gather,
	}


def _sort_gather_indices(
	info: dict[str, Any],
	*,
	primary_key: str,
	indices: np.ndarray,
) -> np.ndarray:
	return sort_gather_indices_for_qc(
		info,
		primary_key=primary_key,
		indices=indices,
	)


def _iter_vis_gathers(
	info: dict[str, Any],
	*,
	primary_keys: list[str],
	max_gathers: int,
	skip_gather_keys: dict[str, set[int]],
	max_traces_per_gather: int | None,
	segy_path: str | Path | None = None,
):
	yield from iter_qc_gathers(
		info,
		primary_keys=primary_keys,
		max_gathers=max_gathers,
		skip_gather_keys=skip_gather_keys,
		max_traces_per_gather=max_traces_per_gather,
		segy_path=segy_path,
	)


def _write_per_file_outputs(
	*,
	out_dir: Path,
	segy_path: str,
	row: dict[str, Any],
	coarse_abs_err: np.ndarray,
	robust_abs_err: np.ndarray,
	vis_cfg: dict[str, Any],
	runtime: SimpleNamespace,
) -> None:
	tag = _build_tag(segy_path)
	out_subdir = Path(out_dir) / tag
	if bool(vis_cfg['save_summary_csv']):
		_write_csv(out_subdir / 'summary.csv', PER_FILE_COLUMNS, [row])
	if bool(vis_cfg['save_cdf']):
		runtime.save_fbpick_physics_qc_cdf_png(
			out_subdir / 'cdf.png',
			coarse_abs_err=coarse_abs_err,
			robust_abs_err=robust_abs_err,
			title=f'{tag} absolute error CDF',
		)


def _save_vis_pngs(
	*,
	info: dict[str, Any],
	segy_path: str,
	out_dir: Path,
	gt_pick_i: np.ndarray,
	coarse_pick_i: np.ndarray,
	robust_pick_i: np.ndarray,
	dataset_cfg: dict[str, Any],
	vis_cfg: dict[str, Any],
	runtime: SimpleNamespace,
	coarse_pmax: np.ndarray | None = None,
	trend_center_i: np.ndarray | None = None,
	physical_center_i: np.ndarray | None = None,
	fine_center_i: np.ndarray | None = None,
	window_start_i: np.ndarray | None = None,
	window_end_i: np.ndarray | None = None,
	final_pick_i: np.ndarray | None = None,
	physical_model_status: np.ndarray | None = None,
) -> list[Path]:
	max_gathers = int(vis_cfg['max_gathers_per_file'])
	if max_gathers <= 0:
		return []

	n_traces = int(gt_pick_i.shape[0])

	def _slice_optional_trace_array(
		values: np.ndarray | None,
		*,
		key: str,
		trace_indices: np.ndarray,
	) -> np.ndarray | None:
		if values is None:
			return None
		arr = np.asarray(values)
		if arr.ndim != 1 or int(arr.shape[0]) != n_traces:
			msg = f'{key} must be 1D with length {n_traces}'
			raise ValueError(msg)
		return arr[trace_indices]

	tag = _build_tag(segy_path)
	out_subdir = Path(out_dir) / tag
	out_paths: list[Path] = []
	for gather_idx, (primary_key, gather_key, trace_indices) in enumerate(
		_iter_vis_gathers(
			info,
			primary_keys=list(dataset_cfg['primary_keys']),
			max_gathers=max_gathers,
			skip_gather_keys=dict(vis_cfg['skip_gather_keys']),
			max_traces_per_gather=vis_cfg['max_traces_per_gather'],
			segy_path=segy_path,
		)
	):
		x_hw = np.stack(
			[np.asarray(info['mmap'][int(i)], dtype=np.float32) for i in trace_indices],
			axis=0,
		)
		out_png = out_subdir / f'gather_{gather_idx:04d}.png'
		title = f'{Path(segy_path).name} {primary_key}={gather_key}'
		out_paths.append(
			runtime.save_fbpick_physics_qc_gather_png(
				out_png,
				raw_wave_hw=x_hw,
				gt_pick_i=gt_pick_i[trace_indices],
				coarse_pick_i=coarse_pick_i[trace_indices],
				robust_pick_i=robust_pick_i[trace_indices],
				coarse_pmax=_slice_optional_trace_array(
					coarse_pmax,
					key='coarse_pmax',
					trace_indices=trace_indices,
				),
				trend_center_i=_slice_optional_trace_array(
					trend_center_i,
					key='trend_center_i',
					trace_indices=trace_indices,
				),
				physical_center_i=_slice_optional_trace_array(
					physical_center_i,
					key='physical_center_i',
					trace_indices=trace_indices,
				),
				fine_center_i=_slice_optional_trace_array(
					fine_center_i,
					key='fine_center_i',
					trace_indices=trace_indices,
				),
				window_start_i=_slice_optional_trace_array(
					window_start_i,
					key='window_start_i',
					trace_indices=trace_indices,
				),
				window_end_i=_slice_optional_trace_array(
					window_end_i,
					key='window_end_i',
					trace_indices=trace_indices,
				),
				final_pick_i=_slice_optional_trace_array(
					final_pick_i,
					key='final_pick_i',
					trace_indices=trace_indices,
				),
				physical_model_status=_slice_optional_trace_array(
					physical_model_status,
					key='physical_model_status',
					trace_indices=trace_indices,
				),
				title=title,
				waveform_norm=str(vis_cfg['waveform_norm']),
				clip_percentile=float(vis_cfg['clip_percentile']),
			)
		)
	if not out_paths:
		print(f'No gather PNGs written for {segy_path}: all candidates were skipped')
	return out_paths


def _load_gt_fb(fb_path: str | Path, *, n_traces: int) -> np.ndarray:
	fb = np.asarray(np.load(fb_path), dtype=np.int64)
	if fb.ndim != 1:
		msg = f'fb must be 1D: {fb_path}'
		raise ValueError(msg)
	if int(fb.shape[0]) != int(n_traces):
		msg = f'fb length {int(fb.shape[0])} != n_traces {int(n_traces)} for {fb_path}'
		raise ValueError(msg)
	return fb


def _require_scalar_int(payload: dict[str, np.ndarray], key: str) -> int:
	return int(np.asarray(payload[key]).item())


def _require_scalar_float(payload: dict[str, np.ndarray], key: str) -> float:
	return float(np.asarray(payload[key]).item())


def _validate_payload_against_info(
	payload: dict[str, np.ndarray],
	*,
	kind: str,
	info: dict[str, Any],
) -> None:
	n_traces = _require_scalar_int(payload, 'n_traces')
	n_samples_orig = _require_scalar_int(payload, 'n_samples_orig')
	dt_sec = _require_scalar_float(payload, 'dt_sec')

	if n_traces != int(info['n_traces']):
		msg = f'{kind} n_traces {n_traces} != info.n_traces {int(info["n_traces"])}'
		raise ValueError(msg)
	if n_samples_orig != int(info['n_samples']):
		msg = (
			f'{kind} n_samples_orig {n_samples_orig} != '
			f'info.n_samples {int(info["n_samples"])}'
		)
		raise ValueError(msg)
	if not np.isclose(dt_sec, float(info['dt_sec']), rtol=0.0, atol=DT_SEC_ATOL):
		msg = f'{kind} dt_sec {dt_sec} != info.dt_sec {float(info["dt_sec"])}'
		raise ValueError(msg)

	expected_trace_indices = np.arange(n_traces, dtype=np.int64)
	trace_indices = np.asarray(payload['trace_indices'], dtype=np.int64)
	if not np.array_equal(trace_indices, expected_trace_indices):
		msg = f'{kind} trace_indices must equal np.arange(n_traces)'
		raise ValueError(msg)

	for key in ('ffid_values', 'chno_values'):
		if not np.array_equal(np.asarray(payload[key]), np.asarray(info[key])):
			msg = f'{kind} {key} does not match SEG-Y headers'
			raise ValueError(msg)
	if not np.allclose(
		np.asarray(payload['offsets_m'], dtype=np.float32),
		np.asarray(info['offsets'], dtype=np.float32),
		rtol=0.0,
		atol=1e-6,
	):
		msg = f'{kind} offsets_m does not match SEG-Y headers'
		raise ValueError(msg)


def _validate_coarse_robust_alignment(
	coarse: dict[str, np.ndarray],
	robust: dict[str, np.ndarray],
) -> None:
	for key in ('n_traces', 'n_samples_orig'):
		if _require_scalar_int(coarse, key) != _require_scalar_int(robust, key):
			msg = f'coarse {key} != robust {key}'
			raise ValueError(msg)
	if not np.isclose(
		_require_scalar_float(coarse, 'dt_sec'),
		_require_scalar_float(robust, 'dt_sec'),
		rtol=0.0,
		atol=DT_SEC_ATOL,
	):
		msg = 'coarse dt_sec != robust dt_sec'
		raise ValueError(msg)
	for key in ('trace_indices', 'ffid_values', 'chno_values'):
		if not np.array_equal(np.asarray(coarse[key]), np.asarray(robust[key])):
			msg = f'coarse {key} != robust {key}'
			raise ValueError(msg)
	if not np.allclose(
		np.asarray(coarse['offsets_m'], dtype=np.float32),
		np.asarray(robust['offsets_m'], dtype=np.float32),
		rtol=0.0,
		atol=1e-6,
	):
		msg = 'coarse offsets_m != robust offsets_m'
		raise ValueError(msg)


def _validate_final_alignment(
	robust: dict[str, np.ndarray],
	final: dict[str, np.ndarray],
) -> None:
	for key in ('n_traces', 'n_samples_orig'):
		if _require_scalar_int(robust, key) != _require_scalar_int(final, key):
			msg = f'robust {key} != final {key}'
			raise ValueError(msg)
	if not np.isclose(
		_require_scalar_float(robust, 'dt_sec'),
		_require_scalar_float(final, 'dt_sec'),
		rtol=0.0,
		atol=DT_SEC_ATOL,
	):
		msg = 'robust dt_sec != final dt_sec'
		raise ValueError(msg)
	if not np.array_equal(
		np.asarray(robust['trace_indices']),
		np.asarray(final['trace_indices']),
	):
		msg = 'robust trace_indices != final trace_indices'
		raise ValueError(msg)


def _resolve_final_npz_path(
	*,
	segy_path: str | Path,
	file_index: int,
	final_npz_dir: Path | None,
	final_npz_files: list[str] | None,
) -> Path | None:
	if final_npz_files is not None:
		return Path(final_npz_files[int(file_index)])
	if final_npz_dir is not None:
		canonical = _build_final_npz_path(
			segy_path=segy_path,
			final_npz_dir=final_npz_dir,
		)
		if canonical.is_file():
			return canonical
		legacy = _build_legacy_final_npz_path(
			segy_path=segy_path,
			final_npz_dir=final_npz_dir,
		)
		if legacy.is_file():
			return legacy
		msg = (
			'final npz file not found; tried:\n'
			f'- {canonical}\n'
			f'- {legacy}'
		)
		raise FileNotFoundError(msg)
	return None


def _rate(mask: np.ndarray) -> float:
	if int(mask.shape[0]) == 0:
		return float('nan')
	return float(np.mean(mask.astype(np.float64)))


def _percentile(values: np.ndarray, q: float) -> float:
	if int(values.shape[0]) == 0:
		return float('nan')
	return float(np.percentile(values.astype(np.float64), float(q)))


def _require_int_trace_vector(
	name: str,
	value: np.ndarray,
	*,
	length: int,
) -> np.ndarray:
	arr = np.asarray(value, dtype=np.int64)
	if arr.ndim != 1 or int(arr.shape[0]) != int(length):
		msg = f'{name} must be 1D with length {int(length)}'
		raise ValueError(msg)
	return arr


def _optional_int_trace_vector(
	name: str,
	value: np.ndarray | None,
	*,
	length: int,
) -> np.ndarray | None:
	if value is None:
		return None
	return _require_int_trace_vector(name, value, length=length)


def _center_r127_mask(center_i: np.ndarray, gt_pick_i: np.ndarray) -> np.ndarray:
	center = np.asarray(center_i, dtype=np.int64)
	gt = np.asarray(gt_pick_i, dtype=np.int64)
	return (gt >= center - 128) & (gt <= center + 127)


def _ready_from_valid_mask(mask: np.ndarray) -> bool:
	return bool(int(mask.shape[0]) > 0 and np.all(mask))


def _empty_optional_summary_arrays() -> dict[str, np.ndarray | None]:
	return {
		'fine_center_abs_err': None,
		'fine_center_robust_abs_err': None,
		'fine_center_r127': None,
		'physical_center_abs_err': None,
		'physical_center_r127': None,
		'gt_in_actual_window': None,
		'final_pick_valid': None,
		'final_pick_r32': None,
		'final_pick_r64': None,
		'final_pick_r127': None,
		'final_pick_abs_err': None,
	}


def _empty_optional_metrics() -> dict[str, float]:
	return {key: float('nan') for key in OPTIONAL_SUMMARY_METRIC_COLUMNS}


def _concat_optional_arrays(arrays: list[np.ndarray]) -> np.ndarray | None:
	if not arrays:
		return None
	return np.concatenate(arrays, axis=0)


def _format_value(value: object) -> object:
	if isinstance(value, float):
		if np.isnan(value):
			return ''
		return f'{value:.10g}'
	return value


def _optional_uint8_vector(
	name: str,
	value: np.ndarray | None,
	*,
	length: int,
) -> np.ndarray | None:
	if value is None:
		return None
	arr = np.asarray(value, dtype=np.uint8)
	if arr.ndim != 1 or int(arr.shape[0]) != int(length):
		msg = f'{name} must be 1D with length {int(length)}'
		raise ValueError(msg)
	return arr


def _format_uint8_counts(
	values: np.ndarray | None,
	*,
	labels: dict[int, str],
) -> str:
	if values is None:
		return ''
	arr = np.asarray(values, dtype=np.uint8)
	if arr.ndim != 1:
		msg = 'values must be 1D'
		raise ValueError(msg)
	if int(arr.shape[0]) == 0:
		return ''
	unique_values, counts = np.unique(arr.astype(np.int64), return_counts=True)
	parts: list[str] = []
	for value, count in zip(unique_values, counts, strict=True):
		value_int = int(value)
		label = labels.get(value_int, f'unknown_{value_int}')
		parts.append(f'{label}={int(count)}')
	return '; '.join(parts)


def _summarize_errors(
	*,
	coarse_pick_i: np.ndarray,
	robust_pick_i: np.ndarray,
	gt_pick_i: np.ndarray,
	n_traces: int,
	n_samples_orig: int,
	physical_center_i: np.ndarray | None = None,
	fine_center_i: np.ndarray | None = None,
	window_start_i: np.ndarray | None = None,
	window_end_i: np.ndarray | None = None,
	final_pick_i: np.ndarray | None = None,
) -> tuple[
	dict[str, Any],
	np.ndarray,
	np.ndarray,
	np.ndarray,
	dict[str, np.ndarray | None],
]:
	coarse_pick = _require_int_trace_vector(
		'coarse_pick_i',
		coarse_pick_i,
		length=n_traces,
	)
	robust_pick = _require_int_trace_vector(
		'robust_pick_i',
		robust_pick_i,
		length=n_traces,
	)
	gt_pick = _require_int_trace_vector('gt_pick_i', gt_pick_i, length=n_traces)
	physical_center = _optional_int_trace_vector(
		'physical_center_i',
		physical_center_i,
		length=n_traces,
	)
	fine_center = _optional_int_trace_vector(
		'fine_center_i',
		fine_center_i,
		length=n_traces,
	)
	final_values = (window_start_i, window_end_i, final_pick_i)
	if any(value is not None for value in final_values) and not all(
		value is not None for value in final_values
	):
		msg = 'window_start_i, window_end_i, and final_pick_i must be provided together'
		raise ValueError(msg)
	window_start = _optional_int_trace_vector(
		'window_start_i',
		window_start_i,
		length=n_traces,
	)
	window_end = _optional_int_trace_vector(
		'window_end_i',
		window_end_i,
		length=n_traces,
	)
	final_pick = _optional_int_trace_vector(
		'final_pick_i',
		final_pick_i,
		length=n_traces,
	)
	valid = (gt_pick > 0) & (gt_pick < int(n_samples_orig))
	valid_gt = gt_pick[valid]
	coarse_abs_err = np.abs(coarse_pick[valid] - valid_gt)
	robust_abs_err = np.abs(robust_pick[valid] - valid_gt)
	r127 = _center_r127_mask(robust_pick[valid], valid_gt)
	coarse_p90 = _percentile(coarse_abs_err, 90.0)
	coarse_p95 = _percentile(coarse_abs_err, 95.0)
	robust_p90 = _percentile(robust_abs_err, 90.0)
	robust_p95 = _percentile(robust_abs_err, 95.0)
	summary_arrays = _empty_optional_summary_arrays()
	robust_ready = _ready_from_valid_mask(r127)
	metrics = {
		'fine_ready': robust_ready,
		'robust_ready': robust_ready,
		'fine_center_ready': float('nan'),
		'actual_window_ready': float('nan'),
		'n_traces': int(n_traces),
		'n_valid_gt': int(valid.sum()),
		'n_invalid_gt': int(n_traces) - int(valid.sum()),
		'R32': _rate(robust_abs_err <= 32),
		'R64': _rate(robust_abs_err <= 64),
		'R127': _rate(r127),
		'coarse_abs_err_median': _percentile(coarse_abs_err, 50.0),
		'coarse_abs_err_p90': coarse_p90,
		'coarse_abs_err_p95': coarse_p95,
		'robust_abs_err_median': _percentile(robust_abs_err, 50.0),
		'robust_abs_err_p90': robust_p90,
		'robust_abs_err_p95': robust_p95,
		'delta_p90': robust_p90 - coarse_p90,
		'delta_p95': robust_p95 - coarse_p95,
		**_empty_optional_metrics(),
	}
	if fine_center is not None:
		fine_abs_err = np.abs(fine_center[valid] - valid_gt)
		fine_r127 = _center_r127_mask(fine_center[valid], valid_gt)
		fine_p90 = _percentile(fine_abs_err, 90.0)
		fine_p95 = _percentile(fine_abs_err, 95.0)
		fine_center_ready = _ready_from_valid_mask(fine_r127)
		metrics.update(
			{
				'fine_ready': fine_center_ready,
				'fine_center_ready': fine_center_ready,
				'fine_center_R32': _rate(fine_abs_err <= 32),
				'fine_center_R64': _rate(fine_abs_err <= 64),
				'fine_center_R127': _rate(fine_r127),
				'fine_center_abs_err_median': _percentile(fine_abs_err, 50.0),
				'fine_center_abs_err_p90': fine_p90,
				'fine_center_abs_err_p95': fine_p95,
				'fine_center_delta_p90_vs_robust': fine_p90 - robust_p90,
				'fine_center_delta_p95_vs_robust': fine_p95 - robust_p95,
			}
		)
		summary_arrays['fine_center_abs_err'] = fine_abs_err
		summary_arrays['fine_center_robust_abs_err'] = robust_abs_err
		summary_arrays['fine_center_r127'] = fine_r127
	if physical_center is not None:
		physical_abs_err = np.abs(physical_center[valid] - valid_gt)
		physical_r127 = _center_r127_mask(physical_center[valid], valid_gt)
		metrics.update(
			{
				'physical_center_R127': _rate(physical_r127),
				'physical_center_abs_err_p90': _percentile(
					physical_abs_err,
					90.0,
				),
				'physical_center_abs_err_p95': _percentile(
					physical_abs_err,
					95.0,
				),
			}
		)
		summary_arrays['physical_center_abs_err'] = physical_abs_err
		summary_arrays['physical_center_r127'] = physical_r127
	if window_start is not None and window_end is not None and final_pick is not None:
		gt_in_actual_window = (window_start[valid] <= valid_gt) & (
			valid_gt <= window_end[valid]
		)
		final_pick_valid = (final_pick > 0) & (final_pick < int(n_samples_orig))
		final_pick_valid_gt = final_pick_valid[valid]
		final_abs_err_scored = np.abs(final_pick[valid] - valid_gt)
		final_pick_r32 = final_pick_valid_gt & (final_abs_err_scored <= 32)
		final_pick_r64 = final_pick_valid_gt & (final_abs_err_scored <= 64)
		final_pick_r127 = final_pick_valid_gt & (final_abs_err_scored <= 127)
		final_abs_err = final_abs_err_scored[final_pick_valid_gt]
		metrics.update(
			{
				'actual_window_ready': _ready_from_valid_mask(gt_in_actual_window),
				'gt_in_actual_window_rate': _rate(gt_in_actual_window),
				'final_pick_valid_rate': _rate(final_pick_valid_gt),
				'final_pick_R32': _rate(final_pick_r32),
				'final_pick_R64': _rate(final_pick_r64),
				'final_pick_R127': _rate(final_pick_r127),
				'final_pick_abs_err_median': _percentile(final_abs_err, 50.0),
				'final_pick_abs_err_p90': _percentile(final_abs_err, 90.0),
				'final_pick_abs_err_p95': _percentile(final_abs_err, 95.0),
			}
		)
		summary_arrays['gt_in_actual_window'] = gt_in_actual_window
		summary_arrays['final_pick_valid'] = final_pick_valid_gt
		summary_arrays['final_pick_r32'] = final_pick_r32
		summary_arrays['final_pick_r64'] = final_pick_r64
		summary_arrays['final_pick_r127'] = final_pick_r127
		summary_arrays['final_pick_abs_err'] = final_abs_err
	return metrics, coarse_abs_err, robust_abs_err, r127, summary_arrays


def _build_info(
	*,
	segy_path: str,
	dataset_cfg: dict[str, Any],
	runtime: SimpleNamespace,
) -> dict[str, Any]:
	info = runtime.build_file_info(
		segy_path,
		ffid_byte=runtime.segyio.TraceField.FieldRecord,
		chno_byte=runtime.segyio.TraceField.TraceNumber,
		cmp_byte=runtime.segyio.TraceField.CDP,
		use_header_cache=bool(dataset_cfg['use_header_cache']),
		include_centroids=False,
		waveform_mode='mmap',
		segy_endian=str(dataset_cfg['infer_endian']),
	)
	return info


def _close_info(info: dict[str, Any]) -> None:
	segy_obj = info.get('segy_obj')
	if segy_obj is not None:
		segy_obj.close()


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=columns)
		writer.writeheader()
		for row in rows:
			writer.writerow({key: _format_value(row.get(key, '')) for key in columns})


def run_pipeline(config_path: str | Path) -> Path:
	runtime = _load_runtime()
	cfg, base_dir = runtime.load_cfg_with_base_dir(Path(config_path))
	prepared = _prepare_cfg(cfg, base_dir=base_dir, runtime=runtime)
	dataset_cfg = _load_dataset_cfg(prepared)
	vis_cfg = _load_vis_cfg(prepared)
	(
		segy_files,
		fb_files,
		coarse_npz_dir,
		robust_npz_dir,
		final_npz_dir,
		final_npz_files,
		out_dir,
	) = _validate_paths(prepared)

	per_file_rows: list[dict[str, Any]] = []
	all_coarse_abs_err: list[np.ndarray] = []
	all_robust_abs_err: list[np.ndarray] = []
	all_r127: list[np.ndarray] = []
	all_fine_ready_r127: list[np.ndarray] = []
	all_fine_center_abs_err: list[np.ndarray] = []
	all_fine_center_robust_abs_err: list[np.ndarray] = []
	all_fine_center_r127: list[np.ndarray] = []
	all_physical_center_abs_err: list[np.ndarray] = []
	all_physical_center_r127: list[np.ndarray] = []
	all_gt_in_actual_window: list[np.ndarray] = []
	all_final_pick_valid: list[np.ndarray] = []
	all_final_pick_r32: list[np.ndarray] = []
	all_final_pick_r64: list[np.ndarray] = []
	all_final_pick_r127: list[np.ndarray] = []
	all_final_pick_abs_err: list[np.ndarray] = []
	all_physical_model_status: list[np.ndarray] = []
	all_physical_model_failure_reason: list[np.ndarray] = []
	total_traces = 0
	total_valid_gt = 0
	total_invalid_gt = 0

	for file_index, (segy_path, fb_path) in enumerate(
		zip(segy_files, fb_files, strict=True)
	):
		coarse_npz_path = _build_coarse_npz_path(
			segy_path=segy_path,
			coarse_npz_dir=coarse_npz_dir,
		)
		robust_npz_path = _build_robust_npz_path(
			segy_path=segy_path,
			robust_npz_dir=robust_npz_dir,
		)
		final_npz_path = _resolve_final_npz_path(
			segy_path=segy_path,
			file_index=file_index,
			final_npz_dir=final_npz_dir,
			final_npz_files=final_npz_files,
		)

		info = _build_info(
			segy_path=segy_path, dataset_cfg=dataset_cfg, runtime=runtime
		)
		try:
			coarse = runtime.load_coarse_npz(coarse_npz_path)
			robust = runtime.load_robust_npz(robust_npz_path)
			_validate_payload_against_info(coarse, kind='coarse', info=info)
			_validate_payload_against_info(robust, kind='robust', info=info)
			_validate_coarse_robust_alignment(coarse, robust)
			final = None
			if final_npz_path is not None:
				final = runtime.load_fbpick_final_npz(final_npz_path)
				_validate_payload_against_info(final, kind='final', info=info)
				_validate_final_alignment(robust, final)

			n_traces = _require_scalar_int(robust, 'n_traces')
			n_samples_orig = _require_scalar_int(robust, 'n_samples_orig')
			physical_model_status = _optional_uint8_vector(
				'physical_model_status',
				robust.get('physical_model_status'),
				length=n_traces,
			)
			physical_model_failure_reason = _optional_uint8_vector(
				'physical_model_failure_reason',
				robust.get('physical_model_failure_reason'),
				length=n_traces,
			)
			gt_pick_i = _load_gt_fb(fb_path, n_traces=n_traces)
			coarse_pick_i = np.asarray(coarse['coarse_pick_i'], dtype=np.int64)
			robust_pick_i = np.asarray(robust['robust_pick_i'], dtype=np.int64)
			coarse_pmax = (
				np.asarray(coarse['coarse_pmax'], dtype=np.float32)
				if 'coarse_pmax' in coarse
				else None
			)
			(
				metrics,
				coarse_abs_err,
				robust_abs_err,
				r127,
				optional_summary_arrays,
			) = _summarize_errors(
				coarse_pick_i=coarse_pick_i,
				robust_pick_i=robust_pick_i,
				gt_pick_i=gt_pick_i,
				n_traces=n_traces,
				n_samples_orig=n_samples_orig,
				physical_center_i=robust.get('physical_center_i'),
				fine_center_i=robust.get('fine_center_i'),
				window_start_i=None if final is None else final.get('window_start_i'),
				window_end_i=None if final is None else final.get('window_end_i'),
				final_pick_i=None if final is None else final.get('final_pick_i'),
			)
			_save_vis_pngs(
				info=info,
				segy_path=segy_path,
				out_dir=out_dir,
				gt_pick_i=gt_pick_i,
				coarse_pick_i=coarse_pick_i,
				robust_pick_i=robust_pick_i,
				coarse_pmax=coarse_pmax,
				trend_center_i=robust.get('trend_center_i'),
				physical_center_i=robust.get('physical_center_i'),
				fine_center_i=robust.get('fine_center_i'),
				window_start_i=None if final is None else final.get('window_start_i'),
				window_end_i=None if final is None else final.get('window_end_i'),
				final_pick_i=None if final is None else final.get('final_pick_i'),
				physical_model_status=physical_model_status,
				dataset_cfg=dataset_cfg,
				vis_cfg=vis_cfg,
				runtime=runtime,
			)
		finally:
			_close_info(info)

		row = {
			'scope': 'file',
			'segy_path': segy_path,
			'fb_path': fb_path,
			'coarse_npz_path': str(coarse_npz_path),
			'robust_npz_path': str(robust_npz_path),
			'n_samples_orig': n_samples_orig,
			'dt_sec': _require_scalar_float(robust, 'dt_sec'),
			**metrics,
			'physical_model_status_counts': _format_uint8_counts(
				physical_model_status,
				labels=PHYSICAL_MODEL_STATUS_LABELS,
			),
			'physical_model_failure_reason_counts': _format_uint8_counts(
				physical_model_failure_reason,
				labels=PHYSICAL_MODEL_FAILURE_LABELS,
			),
		}
		per_file_rows.append(row)
		_write_per_file_outputs(
			out_dir=out_dir,
			segy_path=segy_path,
			row=row,
			coarse_abs_err=coarse_abs_err,
			robust_abs_err=robust_abs_err,
			vis_cfg=vis_cfg,
			runtime=runtime,
		)
		all_coarse_abs_err.append(coarse_abs_err)
		all_robust_abs_err.append(robust_abs_err)
		all_r127.append(r127)
		if optional_summary_arrays['fine_center_r127'] is not None:
			all_fine_ready_r127.append(optional_summary_arrays['fine_center_r127'])
		else:
			all_fine_ready_r127.append(r127)
		if optional_summary_arrays['fine_center_abs_err'] is not None:
			all_fine_center_abs_err.append(
				optional_summary_arrays['fine_center_abs_err']
			)
		if optional_summary_arrays['fine_center_robust_abs_err'] is not None:
			all_fine_center_robust_abs_err.append(
				optional_summary_arrays['fine_center_robust_abs_err']
			)
		if optional_summary_arrays['fine_center_r127'] is not None:
			all_fine_center_r127.append(optional_summary_arrays['fine_center_r127'])
		if optional_summary_arrays['physical_center_abs_err'] is not None:
			all_physical_center_abs_err.append(
				optional_summary_arrays['physical_center_abs_err']
			)
		if optional_summary_arrays['physical_center_r127'] is not None:
			all_physical_center_r127.append(
				optional_summary_arrays['physical_center_r127']
			)
		if optional_summary_arrays['gt_in_actual_window'] is not None:
			all_gt_in_actual_window.append(
				optional_summary_arrays['gt_in_actual_window']
			)
		if optional_summary_arrays['final_pick_valid'] is not None:
			all_final_pick_valid.append(optional_summary_arrays['final_pick_valid'])
		if optional_summary_arrays['final_pick_r32'] is not None:
			all_final_pick_r32.append(optional_summary_arrays['final_pick_r32'])
		if optional_summary_arrays['final_pick_r64'] is not None:
			all_final_pick_r64.append(optional_summary_arrays['final_pick_r64'])
		if optional_summary_arrays['final_pick_r127'] is not None:
			all_final_pick_r127.append(optional_summary_arrays['final_pick_r127'])
		if optional_summary_arrays['final_pick_abs_err'] is not None:
			all_final_pick_abs_err.append(
				optional_summary_arrays['final_pick_abs_err']
			)
		if physical_model_status is not None:
			all_physical_model_status.append(physical_model_status)
		if physical_model_failure_reason is not None:
			all_physical_model_failure_reason.append(physical_model_failure_reason)
		total_traces += int(metrics['n_traces'])
		total_valid_gt += int(metrics['n_valid_gt'])
		total_invalid_gt += int(metrics['n_invalid_gt'])

	global_coarse_abs_err = np.concatenate(all_coarse_abs_err, axis=0)
	global_robust_abs_err = np.concatenate(all_robust_abs_err, axis=0)
	global_r127 = np.concatenate(all_r127, axis=0)
	global_fine_ready_r127 = np.concatenate(all_fine_ready_r127, axis=0)
	global_coarse_p90 = _percentile(global_coarse_abs_err, 90.0)
	global_coarse_p95 = _percentile(global_coarse_abs_err, 95.0)
	global_robust_p90 = _percentile(global_robust_abs_err, 90.0)
	global_robust_p95 = _percentile(global_robust_abs_err, 95.0)
	global_fine_center_abs_err = _concat_optional_arrays(all_fine_center_abs_err)
	global_fine_center_robust_abs_err = _concat_optional_arrays(
		all_fine_center_robust_abs_err
	)
	global_fine_center_r127 = _concat_optional_arrays(all_fine_center_r127)
	global_physical_center_abs_err = _concat_optional_arrays(
		all_physical_center_abs_err
	)
	global_physical_center_r127 = _concat_optional_arrays(all_physical_center_r127)
	global_gt_in_actual_window = _concat_optional_arrays(all_gt_in_actual_window)
	global_final_pick_valid = _concat_optional_arrays(all_final_pick_valid)
	global_final_pick_r32 = _concat_optional_arrays(all_final_pick_r32)
	global_final_pick_r64 = _concat_optional_arrays(all_final_pick_r64)
	global_final_pick_r127 = _concat_optional_arrays(all_final_pick_r127)
	global_final_pick_abs_err = _concat_optional_arrays(all_final_pick_abs_err)
	global_physical_model_status = (
		np.concatenate(all_physical_model_status, axis=0)
		if all_physical_model_status
		else None
	)
	global_physical_model_failure_reason = (
		np.concatenate(all_physical_model_failure_reason, axis=0)
		if all_physical_model_failure_reason
		else None
	)
	global_robust_ready = _ready_from_valid_mask(global_r127)
	global_fine_ready = _ready_from_valid_mask(global_fine_ready_r127)
	global_row = {
		'scope': 'global',
		'fine_ready': global_fine_ready,
		'robust_ready': global_robust_ready,
		'fine_center_ready': float('nan'),
		'actual_window_ready': float('nan'),
		'n_files': len(per_file_rows),
		'n_traces': total_traces,
		'n_valid_gt': total_valid_gt,
		'n_invalid_gt': total_invalid_gt,
		'R32': _rate(global_robust_abs_err <= 32),
		'R64': _rate(global_robust_abs_err <= 64),
		'R127': _rate(global_r127),
		'coarse_abs_err_median': _percentile(global_coarse_abs_err, 50.0),
		'coarse_abs_err_p90': global_coarse_p90,
		'coarse_abs_err_p95': global_coarse_p95,
		'robust_abs_err_median': _percentile(global_robust_abs_err, 50.0),
		'robust_abs_err_p90': global_robust_p90,
		'robust_abs_err_p95': global_robust_p95,
		'delta_p90': global_robust_p90 - global_coarse_p90,
		'delta_p95': global_robust_p95 - global_coarse_p95,
		**_empty_optional_metrics(),
		'physical_model_status_counts': _format_uint8_counts(
			global_physical_model_status,
			labels=PHYSICAL_MODEL_STATUS_LABELS,
		),
		'physical_model_failure_reason_counts': _format_uint8_counts(
			global_physical_model_failure_reason,
			labels=PHYSICAL_MODEL_FAILURE_LABELS,
		),
	}
	if (
		global_fine_center_abs_err is not None
		and global_fine_center_robust_abs_err is not None
		and global_fine_center_r127 is not None
	):
		global_fine_p90 = _percentile(global_fine_center_abs_err, 90.0)
		global_fine_p95 = _percentile(global_fine_center_abs_err, 95.0)
		global_fine_robust_p90 = _percentile(
			global_fine_center_robust_abs_err,
			90.0,
		)
		global_fine_robust_p95 = _percentile(
			global_fine_center_robust_abs_err,
			95.0,
		)
		global_fine_center_ready = _ready_from_valid_mask(global_fine_center_r127)
		global_row.update(
			{
				'fine_center_ready': global_fine_center_ready,
				'fine_center_R32': _rate(global_fine_center_abs_err <= 32),
				'fine_center_R64': _rate(global_fine_center_abs_err <= 64),
				'fine_center_R127': _rate(global_fine_center_r127),
				'fine_center_abs_err_median': _percentile(
					global_fine_center_abs_err,
					50.0,
				),
				'fine_center_abs_err_p90': global_fine_p90,
				'fine_center_abs_err_p95': global_fine_p95,
				'fine_center_delta_p90_vs_robust': (
					global_fine_p90 - global_fine_robust_p90
				),
				'fine_center_delta_p95_vs_robust': (
					global_fine_p95 - global_fine_robust_p95
				),
			}
		)
	if (
		global_physical_center_abs_err is not None
		and global_physical_center_r127 is not None
	):
		global_row.update(
			{
				'physical_center_R127': _rate(global_physical_center_r127),
				'physical_center_abs_err_p90': _percentile(
					global_physical_center_abs_err,
					90.0,
				),
				'physical_center_abs_err_p95': _percentile(
					global_physical_center_abs_err,
					95.0,
				),
			}
		)
	if (
		global_gt_in_actual_window is not None
		and global_final_pick_valid is not None
		and global_final_pick_r32 is not None
		and global_final_pick_r64 is not None
		and global_final_pick_r127 is not None
		and global_final_pick_abs_err is not None
	):
		global_row.update(
			{
				'actual_window_ready': _ready_from_valid_mask(
					global_gt_in_actual_window
				),
				'gt_in_actual_window_rate': _rate(global_gt_in_actual_window),
				'final_pick_valid_rate': _rate(global_final_pick_valid),
				'final_pick_R32': _rate(global_final_pick_r32),
				'final_pick_R64': _rate(global_final_pick_r64),
				'final_pick_R127': _rate(global_final_pick_r127),
				'final_pick_abs_err_median': _percentile(
					global_final_pick_abs_err,
					50.0,
				),
				'final_pick_abs_err_p90': _percentile(
					global_final_pick_abs_err,
					90.0,
				),
				'final_pick_abs_err_p95': _percentile(
					global_final_pick_abs_err,
					95.0,
				),
			}
		)

	summary_per_file_path = out_dir / 'summary_per_file.csv'
	summary_global_path = out_dir / 'summary_global.csv'
	if bool(vis_cfg['save_summary_csv']):
		_write_csv(summary_per_file_path, PER_FILE_COLUMNS, per_file_rows)
		_write_csv(summary_global_path, GLOBAL_COLUMNS, [global_row])
		print(str(summary_per_file_path))
		print(str(summary_global_path))
	if bool(vis_cfg['save_cdf']):
		cdf_all_path = out_dir / 'cdf_all.png'
		runtime.save_fbpick_physics_qc_cdf_png(
			cdf_all_path,
			coarse_abs_err=global_coarse_abs_err,
			robust_abs_err=global_robust_abs_err,
			title='global absolute error CDF',
		)
		print(str(cdf_all_path))
	return summary_global_path


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', required=True)
	args = parser.parse_args(argv)
	run_pipeline(args.config)


if __name__ == '__main__':
	main()
