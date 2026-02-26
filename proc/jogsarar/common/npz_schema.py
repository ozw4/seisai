from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAGE1_PROB_KEYS: tuple[str, ...] = (
    'prob',
    'dt_sec',
    'n_samples_orig',
    'ffid_values',
    'chno_values',
    'offsets',
    'trace_indices',
    'pick0',
    'pick_pre_snap',
    'delta_pick',
    'pick_ref',
    'pick_ref_i',
    'pick_final',
    'cmax',
    'score',
    'rs_valid_mask',
    'conf_prob0',
    'conf_prob1',
    'conf_trend0',
    'conf_trend1',
    'conf_rs1',
    'trend_t_sec',
    'trend_covered',
    'trend_offset_signed_proxy',
    'trend_split_index',
    'trend_source',
    'trend_method',
    'trend_cfg',
)

STAGE2_SIDECAR_BASE_KEYS: tuple[str, ...] = (
    'src_segy',
    'src_infer_npz',
    'out_segy',
    'dt_sec_in',
    'dt_sec_out',
    'dt_us_in',
    'dt_us_out',
    'n_traces',
    'n_samples_in',
    'n_samples_out',
    'window_start_i',
)

STAGE2_SIDECAR_TRAINING_KEYS: tuple[str, ...] = (
    'out_pick_csr_npz',
    'thresh_mode',
    'drop_low_frac',
    'local_global_diff_th_samples',
    'local_discard_radius_traces',
    'trend_center_i_raw',
    'trend_center_i_local',
    'trend_center_i_final',
    'trend_center_i_used',
    'trend_center_i_global',
    'nn_replaced_mask',
    'global_replaced_mask',
    'global_missing_filled_mask',
    'global_edges_all',
    'global_coef_all',
    'global_edges_left',
    'global_coef_left',
    'global_edges_right',
    'global_coef_right',
    'trend_center_i',
    'trend_filled_mask',
    'trend_center_i_round',
    'ffid_values',
    'ffid_unique_values',
    'shot_x_ffid',
    'shot_y_ffid',
    'pick_final_i',
    'pick_win_512',
    'keep_mask',
    'reason_mask',
    'th_conf_prob1',
    'th_conf_trend1',
    'th_conf_rs1',
    'conf_prob1',
    'conf_trend1',
    'conf_rs1',
)

STAGE2_PHASE_PICK_CSR_KEYS: tuple[str, ...] = (
    'n_traces',
    'p_indptr',
    'p_data',
    's_indptr',
    's_data',
)

STAGE4_OUTPUT_KEYS: tuple[str, ...] = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'ffid_values',
    'chno_values',
    'offsets',
    'trace_indices',
    'pick_psn512',
    'pmax_psn',
    'window_start_i',
    'pick_psn_orig_f',
    'pick_psn_orig_i',
    'delta_pick_rs',
    'cmax_rs',
    'rs_valid_mask',
    'pick_rs_i',
    'pick_final',
)


@dataclass(frozen=True)
class Stage1Prob:
    prob: np.ndarray
    dt_sec: float
    n_samples_orig: int
    ffid_values: np.ndarray
    chno_values: np.ndarray
    offsets: np.ndarray
    trace_indices: np.ndarray
    pick0: np.ndarray
    pick_pre_snap: np.ndarray
    delta_pick: np.ndarray
    pick_ref: np.ndarray
    pick_ref_i: np.ndarray
    pick_final: np.ndarray
    cmax: np.ndarray
    score: np.ndarray
    rs_valid_mask: np.ndarray
    conf_prob0: np.ndarray
    conf_prob1: np.ndarray
    conf_trend0: np.ndarray
    conf_trend1: np.ndarray
    conf_rs1: np.ndarray
    trend_t_sec: np.ndarray
    trend_covered: np.ndarray
    trend_offset_signed_proxy: np.ndarray
    trend_split_index: np.ndarray
    trend_source: str
    trend_method: str
    trend_cfg: str


@dataclass(frozen=True)
class Stage2SidecarTraining:
    out_pick_csr_npz: str
    thresh_mode: str
    drop_low_frac: float
    local_global_diff_th_samples: int
    local_discard_radius_traces: int
    trend_center_i_raw: np.ndarray
    trend_center_i_local: np.ndarray
    trend_center_i_final: np.ndarray
    trend_center_i_used: np.ndarray
    trend_center_i_global: np.ndarray
    nn_replaced_mask: np.ndarray
    global_replaced_mask: np.ndarray
    global_missing_filled_mask: np.ndarray
    global_edges_all: np.ndarray
    global_coef_all: np.ndarray
    global_edges_left: np.ndarray
    global_coef_left: np.ndarray
    global_edges_right: np.ndarray
    global_coef_right: np.ndarray
    trend_center_i: np.ndarray
    trend_filled_mask: np.ndarray
    trend_center_i_round: np.ndarray
    ffid_values: np.ndarray
    ffid_unique_values: np.ndarray
    shot_x_ffid: np.ndarray
    shot_y_ffid: np.ndarray
    pick_final_i: np.ndarray
    pick_win_512: np.ndarray
    keep_mask: np.ndarray
    reason_mask: np.ndarray
    th_conf_prob1: float
    th_conf_trend1: float
    th_conf_rs1: float
    conf_prob1: np.ndarray
    conf_trend1: np.ndarray
    conf_rs1: np.ndarray


@dataclass(frozen=True)
class Stage2Sidecar:
    src_segy: str
    src_infer_npz: str
    out_segy: str
    dt_sec_in: float
    dt_sec_out: float
    dt_us_in: int
    dt_us_out: int
    n_traces: int
    n_samples_in: int
    n_samples_out: int
    window_start_i: np.ndarray
    training: Stage2SidecarTraining | None


@dataclass(frozen=True)
class Stage4Output:
    dt_sec: float
    n_samples_orig: int
    n_traces: int
    ffid_values: np.ndarray
    chno_values: np.ndarray
    offsets: np.ndarray
    trace_indices: np.ndarray
    pick_psn512: np.ndarray
    pmax_psn: np.ndarray
    window_start_i: np.ndarray
    pick_psn_orig_f: np.ndarray
    pick_psn_orig_i: np.ndarray
    delta_pick_rs: np.ndarray
    cmax_rs: np.ndarray
    rs_valid_mask: np.ndarray
    pick_rs_i: np.ndarray
    pick_final: np.ndarray


def _require_keys(
    npz: Mapping[str, np.ndarray], keys: tuple[str, ...], *, context: str
) -> None:
    missing = [k for k in keys if k not in npz]
    if missing:
        raise KeyError(
            f'{context}: missing keys={missing}, available={sorted(npz.keys())}'
        )


def _as_array(npz: Mapping[str, np.ndarray], key: str, *, context: str) -> np.ndarray:
    if key not in npz:
        raise KeyError(f'{context}: missing key={key!r}')
    return np.asarray(npz[key])


def _require_scalar(arr: np.ndarray, key: str, *, context: str) -> np.ndarray:
    if arr.ndim != 0:
        raise ValueError(f'{context}: {key} must be scalar, got shape={arr.shape}')
    return arr


def _as_int_scalar(npz: Mapping[str, np.ndarray], key: str, *, context: str) -> int:
    arr = _require_scalar(_as_array(npz, key, context=context), key, context=context)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f'{context}: {key} must be int scalar, got dtype={arr.dtype}')
    return int(arr.item())


def _as_float_scalar(npz: Mapping[str, np.ndarray], key: str, *, context: str) -> float:
    arr = _require_scalar(_as_array(npz, key, context=context), key, context=context)
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f'{context}: {key} must be float scalar, got dtype={arr.dtype}')
    return float(arr.item())


def _as_str_scalar(npz: Mapping[str, np.ndarray], key: str, *, context: str) -> str:
    arr = _require_scalar(_as_array(npz, key, context=context), key, context=context)
    if arr.dtype.kind in ('U', 'S'):
        return str(arr.item())
    if arr.dtype.kind == 'O':
        val = arr.item()
        if isinstance(val, str):
            return val
    raise TypeError(f'{context}: {key} must be string scalar, got dtype={arr.dtype}')


def _require_1d(
    npz: Mapping[str, np.ndarray],
    key: str,
    *,
    context: str,
    n: int | None = None,
) -> np.ndarray:
    arr = _as_array(npz, key, context=context)
    if arr.ndim != 1:
        raise ValueError(f'{context}: {key} must be 1D, got shape={arr.shape}')
    if n is not None and arr.shape[0] != int(n):
        raise ValueError(f'{context}: {key} must have length {n}, got {arr.shape[0]}')
    return arr


def _require_shape(
    npz: Mapping[str, np.ndarray],
    key: str,
    *,
    context: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    arr = _as_array(npz, key, context=context)
    if arr.shape != shape:
        raise ValueError(f'{context}: {key} must have shape {shape}, got {arr.shape}')
    return arr


def _require_float_dtype(arr: np.ndarray, key: str, *, context: str) -> None:
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f'{context}: {key} must be float-like dtype, got {arr.dtype}')


def _require_int_dtype(arr: np.ndarray, key: str, *, context: str) -> None:
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f'{context}: {key} must be int-like dtype, got {arr.dtype}')


def _require_bool_dtype(arr: np.ndarray, key: str, *, context: str) -> None:
    if arr.dtype != np.bool_:
        raise TypeError(f'{context}: {key} must be bool dtype, got {arr.dtype}')


def _validate_positive_int(name: str, value: int, *, context: str) -> None:
    if int(value) <= 0:
        raise ValueError(f'{context}: {name} must be > 0, got {value}')


def _validate_positive_float(name: str, value: float, *, context: str) -> None:
    if (not np.isfinite(value)) or float(value) <= 0.0:
        raise ValueError(f'{context}: {name} must be finite and > 0, got {value}')


def _training_keys_state(
    npz: Mapping[str, np.ndarray],
) -> tuple[bool, list[str], list[str]]:
    present = [k for k in STAGE2_SIDECAR_TRAINING_KEYS if k in npz]
    missing = [k for k in STAGE2_SIDECAR_TRAINING_KEYS if k not in npz]
    enabled = len(present) > 0
    return enabled, present, missing


def validate_stage1_prob_npz(npz: Mapping[str, np.ndarray]) -> None:
    context = 'stage1_prob'
    _require_keys(npz, STAGE1_PROB_KEYS, context=context)

    dt_sec = _as_float_scalar(npz, 'dt_sec', context=context)
    n_samples_orig = _as_int_scalar(npz, 'n_samples_orig', context=context)
    _validate_positive_float('dt_sec', dt_sec, context=context)
    _validate_positive_int('n_samples_orig', n_samples_orig, context=context)

    ffid = _require_1d(npz, 'ffid_values', context=context)
    _require_int_dtype(ffid, 'ffid_values', context=context)
    n_traces = int(ffid.shape[0])
    _validate_positive_int('n_traces', n_traces, context=context)

    prob = _as_array(npz, 'prob', context=context)
    if prob.ndim != 2 or prob.shape[0] != n_traces:
        raise ValueError(
            f'{context}: prob must be 2D with shape (n_traces, Wpad), got {prob.shape}, '
            f'n_traces={n_traces}'
        )
    _require_float_dtype(prob, 'prob', context=context)
    if int(prob.shape[1]) < int(n_samples_orig):
        raise ValueError(
            f'{context}: prob width must be >= n_samples_orig; got prob.shape[1]={prob.shape[1]}, '
            f'n_samples_orig={n_samples_orig}'
        )

    chno = _require_1d(npz, 'chno_values', context=context, n=n_traces)
    _require_int_dtype(chno, 'chno_values', context=context)

    offsets = _require_1d(npz, 'offsets', context=context, n=n_traces)
    _require_float_dtype(offsets, 'offsets', context=context)

    trace_indices = _require_1d(npz, 'trace_indices', context=context, n=n_traces)
    _require_int_dtype(trace_indices, 'trace_indices', context=context)

    int_1d_keys = (
        'pick0',
        'pick_pre_snap',
        'pick_ref_i',
        'pick_final',
        'trend_split_index',
    )
    float_1d_keys = (
        'delta_pick',
        'pick_ref',
        'cmax',
        'score',
        'conf_prob0',
        'conf_prob1',
        'conf_trend0',
        'conf_trend1',
        'conf_rs1',
        'trend_t_sec',
        'trend_offset_signed_proxy',
    )
    bool_1d_keys = ('rs_valid_mask', 'trend_covered')

    for key in int_1d_keys:
        arr = _require_1d(npz, key, context=context, n=n_traces)
        _require_int_dtype(arr, key, context=context)

    for key in float_1d_keys:
        arr = _require_1d(npz, key, context=context, n=n_traces)
        _require_float_dtype(arr, key, context=context)

    for key in bool_1d_keys:
        arr = _require_1d(npz, key, context=context, n=n_traces)
        _require_bool_dtype(arr, key, context=context)

    _as_str_scalar(npz, 'trend_source', context=context)
    _as_str_scalar(npz, 'trend_method', context=context)
    _as_str_scalar(npz, 'trend_cfg', context=context)


def validate_stage2_sidecar_npz(npz: Mapping[str, np.ndarray]) -> None:
    context = 'stage2_sidecar'
    _require_keys(npz, STAGE2_SIDECAR_BASE_KEYS, context=context)

    _as_str_scalar(npz, 'src_segy', context=context)
    _as_str_scalar(npz, 'src_infer_npz', context=context)
    _as_str_scalar(npz, 'out_segy', context=context)

    dt_sec_in = _as_float_scalar(npz, 'dt_sec_in', context=context)
    dt_sec_out = _as_float_scalar(npz, 'dt_sec_out', context=context)
    _validate_positive_float('dt_sec_in', dt_sec_in, context=context)
    _validate_positive_float('dt_sec_out', dt_sec_out, context=context)

    dt_us_in = _as_int_scalar(npz, 'dt_us_in', context=context)
    dt_us_out = _as_int_scalar(npz, 'dt_us_out', context=context)
    _validate_positive_int('dt_us_in', dt_us_in, context=context)
    _validate_positive_int('dt_us_out', dt_us_out, context=context)

    n_traces = _as_int_scalar(npz, 'n_traces', context=context)
    n_samples_in = _as_int_scalar(npz, 'n_samples_in', context=context)
    n_samples_out = _as_int_scalar(npz, 'n_samples_out', context=context)
    _validate_positive_int('n_traces', n_traces, context=context)
    _validate_positive_int('n_samples_in', n_samples_in, context=context)
    _validate_positive_int('n_samples_out', n_samples_out, context=context)

    window_start_i = _require_1d(npz, 'window_start_i', context=context, n=n_traces)
    _require_int_dtype(window_start_i, 'window_start_i', context=context)

    training_enabled, present, missing = _training_keys_state(npz)
    if training_enabled and missing:
        raise KeyError(
            f'{context}: training sidecar keys are partial. '
            f'present={present}, missing={missing}'
        )
    if not training_enabled:
        return

    _as_str_scalar(npz, 'out_pick_csr_npz', context=context)

    thresh_mode = _as_str_scalar(npz, 'thresh_mode', context=context)
    if thresh_mode not in ('global', 'per_segy'):
        raise ValueError(
            f"{context}: thresh_mode must be 'global' or 'per_segy', got {thresh_mode!r}"
        )

    _as_float_scalar(npz, 'drop_low_frac', context=context)
    _as_int_scalar(npz, 'local_global_diff_th_samples', context=context)
    _as_int_scalar(npz, 'local_discard_radius_traces', context=context)

    for key in (
        'trend_center_i_raw',
        'trend_center_i_local',
        'trend_center_i_final',
        'trend_center_i_used',
        'trend_center_i_global',
        'trend_center_i',
        'pick_win_512',
        'conf_prob1',
        'conf_trend1',
        'conf_rs1',
    ):
        arr = _require_1d(npz, key, context=context, n=n_traces)
        _require_float_dtype(arr, key, context=context)

    for key in (
        'nn_replaced_mask',
        'global_replaced_mask',
        'global_missing_filled_mask',
        'trend_filled_mask',
        'keep_mask',
    ):
        arr = _require_1d(npz, key, context=context, n=n_traces)
        _require_bool_dtype(arr, key, context=context)

    for key in ('trend_center_i_round', 'pick_final_i', 'ffid_values'):
        arr = _require_1d(npz, key, context=context, n=n_traces)
        _require_int_dtype(arr, key, context=context)

    reason_mask = _require_1d(npz, 'reason_mask', context=context, n=n_traces)
    _require_int_dtype(reason_mask, 'reason_mask', context=context)

    ffid_unique = _require_1d(npz, 'ffid_unique_values', context=context)
    _require_int_dtype(ffid_unique, 'ffid_unique_values', context=context)
    n_ffid = int(ffid_unique.shape[0])

    shot_x_ffid = _require_1d(npz, 'shot_x_ffid', context=context, n=n_ffid)
    _require_float_dtype(shot_x_ffid, 'shot_x_ffid', context=context)
    shot_y_ffid = _require_1d(npz, 'shot_y_ffid', context=context, n=n_ffid)
    _require_float_dtype(shot_y_ffid, 'shot_y_ffid', context=context)

    for key in ('global_edges_all', 'global_edges_left', 'global_edges_right'):
        arr = _require_shape(npz, key, context=context, shape=(3,))
        _require_float_dtype(arr, key, context=context)

    for key in ('global_coef_all', 'global_coef_left', 'global_coef_right'):
        arr = _require_shape(npz, key, context=context, shape=(2, 2))
        _require_float_dtype(arr, key, context=context)

    _as_float_scalar(npz, 'th_conf_prob1', context=context)
    _as_float_scalar(npz, 'th_conf_trend1', context=context)
    _as_float_scalar(npz, 'th_conf_rs1', context=context)


def validate_stage4_output_npz(npz: Mapping[str, np.ndarray]) -> None:
    context = 'stage4_output'
    _require_keys(npz, STAGE4_OUTPUT_KEYS, context=context)

    dt_sec = _as_float_scalar(npz, 'dt_sec', context=context)
    _validate_positive_float('dt_sec', dt_sec, context=context)

    n_samples_orig = _as_int_scalar(npz, 'n_samples_orig', context=context)
    _validate_positive_int('n_samples_orig', n_samples_orig, context=context)

    n_traces = _as_int_scalar(npz, 'n_traces', context=context)
    _validate_positive_int('n_traces', n_traces, context=context)

    for key in (
        'ffid_values',
        'chno_values',
        'trace_indices',
        'pick_psn512',
        'window_start_i',
        'pick_psn_orig_i',
        'pick_rs_i',
        'pick_final',
    ):
        arr = _require_1d(npz, key, context=context, n=n_traces)
        _require_int_dtype(arr, key, context=context)

    for key in ('offsets', 'pmax_psn', 'pick_psn_orig_f', 'delta_pick_rs', 'cmax_rs'):
        arr = _require_1d(npz, key, context=context, n=n_traces)
        _require_float_dtype(arr, key, context=context)

    rs_valid = _require_1d(npz, 'rs_valid_mask', context=context, n=n_traces)
    _require_bool_dtype(rs_valid, 'rs_valid_mask', context=context)


def _load_npz_dict(path: Path | str) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _build_stage1_prob(data: Mapping[str, np.ndarray]) -> Stage1Prob:
    return Stage1Prob(
        prob=np.asarray(data['prob']),
        dt_sec=float(np.asarray(data['dt_sec']).item()),
        n_samples_orig=int(np.asarray(data['n_samples_orig']).item()),
        ffid_values=np.asarray(data['ffid_values']),
        chno_values=np.asarray(data['chno_values']),
        offsets=np.asarray(data['offsets']),
        trace_indices=np.asarray(data['trace_indices']),
        pick0=np.asarray(data['pick0']),
        pick_pre_snap=np.asarray(data['pick_pre_snap']),
        delta_pick=np.asarray(data['delta_pick']),
        pick_ref=np.asarray(data['pick_ref']),
        pick_ref_i=np.asarray(data['pick_ref_i']),
        pick_final=np.asarray(data['pick_final']),
        cmax=np.asarray(data['cmax']),
        score=np.asarray(data['score']),
        rs_valid_mask=np.asarray(data['rs_valid_mask']),
        conf_prob0=np.asarray(data['conf_prob0']),
        conf_prob1=np.asarray(data['conf_prob1']),
        conf_trend0=np.asarray(data['conf_trend0']),
        conf_trend1=np.asarray(data['conf_trend1']),
        conf_rs1=np.asarray(data['conf_rs1']),
        trend_t_sec=np.asarray(data['trend_t_sec']),
        trend_covered=np.asarray(data['trend_covered']),
        trend_offset_signed_proxy=np.asarray(data['trend_offset_signed_proxy']),
        trend_split_index=np.asarray(data['trend_split_index']),
        trend_source=str(np.asarray(data['trend_source']).item()),
        trend_method=str(np.asarray(data['trend_method']).item()),
        trend_cfg=str(np.asarray(data['trend_cfg']).item()),
    )


def _build_stage2_training(data: Mapping[str, np.ndarray]) -> Stage2SidecarTraining:
    return Stage2SidecarTraining(
        out_pick_csr_npz=str(np.asarray(data['out_pick_csr_npz']).item()),
        thresh_mode=str(np.asarray(data['thresh_mode']).item()),
        drop_low_frac=float(np.asarray(data['drop_low_frac']).item()),
        local_global_diff_th_samples=int(
            np.asarray(data['local_global_diff_th_samples']).item()
        ),
        local_discard_radius_traces=int(
            np.asarray(data['local_discard_radius_traces']).item()
        ),
        trend_center_i_raw=np.asarray(data['trend_center_i_raw']),
        trend_center_i_local=np.asarray(data['trend_center_i_local']),
        trend_center_i_final=np.asarray(data['trend_center_i_final']),
        trend_center_i_used=np.asarray(data['trend_center_i_used']),
        trend_center_i_global=np.asarray(data['trend_center_i_global']),
        nn_replaced_mask=np.asarray(data['nn_replaced_mask']),
        global_replaced_mask=np.asarray(data['global_replaced_mask']),
        global_missing_filled_mask=np.asarray(data['global_missing_filled_mask']),
        global_edges_all=np.asarray(data['global_edges_all']),
        global_coef_all=np.asarray(data['global_coef_all']),
        global_edges_left=np.asarray(data['global_edges_left']),
        global_coef_left=np.asarray(data['global_coef_left']),
        global_edges_right=np.asarray(data['global_edges_right']),
        global_coef_right=np.asarray(data['global_coef_right']),
        trend_center_i=np.asarray(data['trend_center_i']),
        trend_filled_mask=np.asarray(data['trend_filled_mask']),
        trend_center_i_round=np.asarray(data['trend_center_i_round']),
        ffid_values=np.asarray(data['ffid_values']),
        ffid_unique_values=np.asarray(data['ffid_unique_values']),
        shot_x_ffid=np.asarray(data['shot_x_ffid']),
        shot_y_ffid=np.asarray(data['shot_y_ffid']),
        pick_final_i=np.asarray(data['pick_final_i']),
        pick_win_512=np.asarray(data['pick_win_512']),
        keep_mask=np.asarray(data['keep_mask']),
        reason_mask=np.asarray(data['reason_mask']),
        th_conf_prob1=float(np.asarray(data['th_conf_prob1']).item()),
        th_conf_trend1=float(np.asarray(data['th_conf_trend1']).item()),
        th_conf_rs1=float(np.asarray(data['th_conf_rs1']).item()),
        conf_prob1=np.asarray(data['conf_prob1']),
        conf_trend1=np.asarray(data['conf_trend1']),
        conf_rs1=np.asarray(data['conf_rs1']),
    )


def _build_stage2_sidecar(data: Mapping[str, np.ndarray]) -> Stage2Sidecar:
    training_enabled, _, _ = _training_keys_state(data)
    training = _build_stage2_training(data) if training_enabled else None

    return Stage2Sidecar(
        src_segy=str(np.asarray(data['src_segy']).item()),
        src_infer_npz=str(np.asarray(data['src_infer_npz']).item()),
        out_segy=str(np.asarray(data['out_segy']).item()),
        dt_sec_in=float(np.asarray(data['dt_sec_in']).item()),
        dt_sec_out=float(np.asarray(data['dt_sec_out']).item()),
        dt_us_in=int(np.asarray(data['dt_us_in']).item()),
        dt_us_out=int(np.asarray(data['dt_us_out']).item()),
        n_traces=int(np.asarray(data['n_traces']).item()),
        n_samples_in=int(np.asarray(data['n_samples_in']).item()),
        n_samples_out=int(np.asarray(data['n_samples_out']).item()),
        window_start_i=np.asarray(data['window_start_i']),
        training=training,
    )


def _build_stage4_output(data: Mapping[str, np.ndarray]) -> Stage4Output:
    return Stage4Output(
        dt_sec=float(np.asarray(data['dt_sec']).item()),
        n_samples_orig=int(np.asarray(data['n_samples_orig']).item()),
        n_traces=int(np.asarray(data['n_traces']).item()),
        ffid_values=np.asarray(data['ffid_values']),
        chno_values=np.asarray(data['chno_values']),
        offsets=np.asarray(data['offsets']),
        trace_indices=np.asarray(data['trace_indices']),
        pick_psn512=np.asarray(data['pick_psn512']),
        pmax_psn=np.asarray(data['pmax_psn']),
        window_start_i=np.asarray(data['window_start_i']),
        pick_psn_orig_f=np.asarray(data['pick_psn_orig_f']),
        pick_psn_orig_i=np.asarray(data['pick_psn_orig_i']),
        delta_pick_rs=np.asarray(data['delta_pick_rs']),
        cmax_rs=np.asarray(data['cmax_rs']),
        rs_valid_mask=np.asarray(data['rs_valid_mask']),
        pick_rs_i=np.asarray(data['pick_rs_i']),
        pick_final=np.asarray(data['pick_final']),
    )


def load_stage1_prob(path: Path | str) -> Stage1Prob:
    data = _load_npz_dict(path)
    validate_stage1_prob_npz(data)
    return _build_stage1_prob(data)


def load_stage2_sidecar(path: Path | str) -> Stage2Sidecar:
    data = _load_npz_dict(path)
    validate_stage2_sidecar_npz(data)
    return _build_stage2_sidecar(data)


def load_stage4_output(path: Path | str) -> Stage4Output:
    data = _load_npz_dict(path)
    validate_stage4_output_npz(data)
    return _build_stage4_output(data)


__all__ = [
    'STAGE1_PROB_KEYS',
    'STAGE2_PHASE_PICK_CSR_KEYS',
    'STAGE2_SIDECAR_BASE_KEYS',
    'STAGE2_SIDECAR_TRAINING_KEYS',
    'STAGE4_OUTPUT_KEYS',
    'Stage1Prob',
    'Stage2Sidecar',
    'Stage2SidecarTraining',
    'Stage4Output',
    'load_stage1_prob',
    'load_stage2_sidecar',
    'load_stage4_output',
    'validate_stage1_prob_npz',
    'validate_stage2_sidecar_npz',
    'validate_stage4_output_npz',
]
