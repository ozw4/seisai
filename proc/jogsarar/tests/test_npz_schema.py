from __future__ import annotations

import numpy as np
import pytest
from common import npz_schema as schema


def _stage1_payload(
    n_traces: int = 4, wpad: int = 32, n_samples_orig: int = 20
) -> dict[str, np.ndarray]:
    idx = np.arange(n_traces, dtype=np.int64)
    return {
        'prob': np.zeros((n_traces, wpad), dtype=np.float16),
        'dt_sec': np.asarray(0.002, dtype=np.float32),
        'n_samples_orig': np.asarray(n_samples_orig, dtype=np.int32),
        'ffid_values': np.arange(n_traces, dtype=np.int32),
        'chno_values': np.arange(1, n_traces + 1, dtype=np.int32),
        'offsets': np.linspace(100.0, 500.0, n_traces, dtype=np.float32),
        'trace_indices': idx,
        'pick0': np.arange(n_traces, dtype=np.int32),
        'pick_pre_snap': np.arange(n_traces, dtype=np.int32),
        'delta_pick': np.zeros(n_traces, dtype=np.float32),
        'pick_ref': np.arange(n_traces, dtype=np.float32),
        'pick_ref_i': np.arange(n_traces, dtype=np.int32),
        'pick_final': np.arange(n_traces, dtype=np.int32),
        'cmax': np.ones(n_traces, dtype=np.float32),
        'score': np.ones(n_traces, dtype=np.float32),
        'rs_valid_mask': np.ones(n_traces, dtype=bool),
        'conf_prob0': np.ones(n_traces, dtype=np.float32),
        'conf_prob1': np.ones(n_traces, dtype=np.float32),
        'conf_trend0': np.ones(n_traces, dtype=np.float32),
        'conf_trend1': np.ones(n_traces, dtype=np.float32),
        'conf_rs1': np.ones(n_traces, dtype=np.float32),
        'trend_t_sec': np.linspace(0.1, 0.4, n_traces, dtype=np.float32),
        'trend_covered': np.ones(n_traces, dtype=bool),
        'trend_offset_signed_proxy': np.linspace(
            -10.0, 10.0, n_traces, dtype=np.float32
        ),
        'trend_split_index': np.full(n_traces, -1, dtype=np.int32),
        'trend_source': np.asarray('pick_final'),
        'trend_method': np.asarray('local_irls_split_sides'),
        'trend_cfg': np.asarray('section_len=16,stride=4'),
    }


def _stage2_base_payload(n_traces: int = 4) -> dict[str, np.ndarray]:
    return {
        'src_segy': np.asarray('/tmp/in.sgy'),
        'src_infer_npz': np.asarray('/tmp/in.prob.npz'),
        'out_segy': np.asarray('/tmp/out.win512.sgy'),
        'dt_sec_in': np.asarray(0.002, dtype=np.float32),
        'dt_sec_out': np.asarray(0.001, dtype=np.float32),
        'dt_us_in': np.asarray(2000, dtype=np.int32),
        'dt_us_out': np.asarray(1000, dtype=np.int32),
        'n_traces': np.asarray(n_traces, dtype=np.int32),
        'n_samples_in': np.asarray(1200, dtype=np.int32),
        'n_samples_out': np.asarray(512, dtype=np.int32),
        'window_start_i': np.arange(n_traces, dtype=np.int64),
    }


def _stage2_training_payload(
    n_traces: int = 4, n_ffid: int = 2
) -> dict[str, np.ndarray]:
    base = _stage2_base_payload(n_traces=n_traces)
    per_trace_float = np.linspace(10.0, 20.0, n_traces, dtype=np.float32)
    per_trace_bool = np.zeros(n_traces, dtype=bool)
    per_trace_int = np.arange(n_traces, dtype=np.int64)

    base.update(
        {
            'out_pick_csr_npz': np.asarray('/tmp/out.phase_pick.csr.npz'),
            'thresh_mode': np.asarray('global'),
            'drop_low_frac': np.asarray(0.05, dtype=np.float32),
            'local_global_diff_th_samples': np.asarray(128, dtype=np.int32),
            'local_discard_radius_traces': np.asarray(32, dtype=np.int32),
            'trend_center_i_raw': per_trace_float.copy(),
            'trend_center_i_local': per_trace_float.copy(),
            'trend_center_i_final': per_trace_float.copy(),
            'trend_center_i_used': per_trace_float.copy(),
            'trend_center_i_global': per_trace_float.copy(),
            'nn_replaced_mask': per_trace_bool.copy(),
            'global_replaced_mask': per_trace_bool.copy(),
            'global_missing_filled_mask': per_trace_bool.copy(),
            'global_edges_all': np.asarray([100.0, 500.0, 1000.0], dtype=np.float32),
            'global_coef_all': np.asarray(
                [[0.001, 0.1], [0.0005, 0.2]], dtype=np.float32
            ),
            'global_edges_left': np.asarray([100.0, 500.0, 1000.0], dtype=np.float32),
            'global_coef_left': np.asarray(
                [[0.001, 0.1], [0.0005, 0.2]], dtype=np.float32
            ),
            'global_edges_right': np.asarray([100.0, 500.0, 1000.0], dtype=np.float32),
            'global_coef_right': np.asarray(
                [[0.001, 0.1], [0.0005, 0.2]], dtype=np.float32
            ),
            'trend_center_i': per_trace_float.copy(),
            'trend_filled_mask': per_trace_bool.copy(),
            'trend_center_i_round': per_trace_int.copy(),
            'ffid_values': np.arange(n_traces, dtype=np.int64),
            'ffid_unique_values': np.arange(n_ffid, dtype=np.int64),
            'shot_x_ffid': np.arange(n_ffid, dtype=np.float64),
            'shot_y_ffid': np.arange(n_ffid, dtype=np.float64),
            'pick_final_i': per_trace_int.copy(),
            'pick_win_512': np.arange(n_traces, dtype=np.float32),
            'keep_mask': np.ones(n_traces, dtype=bool),
            'reason_mask': np.zeros(n_traces, dtype=np.uint8),
            'th_conf_prob1': np.asarray(0.5, dtype=np.float32),
            'th_conf_trend1': np.asarray(0.5, dtype=np.float32),
            'th_conf_rs1': np.asarray(0.5, dtype=np.float32),
            'conf_prob1': np.ones(n_traces, dtype=np.float32),
            'conf_trend1': np.ones(n_traces, dtype=np.float32),
            'conf_rs1': np.ones(n_traces, dtype=np.float32),
        }
    )
    return base


def _stage4_payload(n_traces: int = 4) -> dict[str, np.ndarray]:
    idx64 = np.arange(n_traces, dtype=np.int64)
    idx32 = np.arange(n_traces, dtype=np.int32)
    return {
        'dt_sec': np.asarray(0.002, dtype=np.float32),
        'n_samples_orig': np.asarray(1200, dtype=np.int32),
        'n_traces': np.asarray(n_traces, dtype=np.int32),
        'ffid_values': idx32.copy(),
        'chno_values': idx32 + 1,
        'offsets': np.linspace(100.0, 500.0, n_traces, dtype=np.float32),
        'trace_indices': idx64.copy(),
        'pick_psn512': idx32.copy(),
        'pmax_psn': np.ones(n_traces, dtype=np.float32),
        'window_start_i': idx64.copy(),
        'pick_psn_orig_f': np.arange(n_traces, dtype=np.float32),
        'pick_psn_orig_i': idx32.copy(),
        'delta_pick_rs': np.zeros(n_traces, dtype=np.float32),
        'cmax_rs': np.ones(n_traces, dtype=np.float32),
        'rs_valid_mask': np.ones(n_traces, dtype=bool),
        'pick_rs_i': idx32.copy(),
        'pick_final': idx32.copy(),
    }


def test_validate_stage1_prob_npz_passes() -> None:
    payload = _stage1_payload()
    schema.validate_stage1_prob_npz(payload)


def test_validate_stage1_prob_npz_missing_key_fails() -> None:
    payload = _stage1_payload()
    payload.pop('pick_final')
    with pytest.raises(KeyError):
        schema.validate_stage1_prob_npz(payload)


def test_validate_stage2_sidecar_npz_passes_inference_only() -> None:
    payload = _stage2_base_payload()
    schema.validate_stage2_sidecar_npz(payload)


def test_validate_stage2_sidecar_npz_rejects_partial_training_keys() -> None:
    payload = _stage2_base_payload()
    payload['keep_mask'] = np.ones(4, dtype=bool)
    with pytest.raises(KeyError):
        schema.validate_stage2_sidecar_npz(payload)


def test_validate_stage2_sidecar_npz_passes_training_mode() -> None:
    payload = _stage2_training_payload()
    schema.validate_stage2_sidecar_npz(payload)


def test_validate_stage4_output_npz_shape_mismatch_fails() -> None:
    payload = _stage4_payload()
    payload['pick_final'] = np.arange(3, dtype=np.int32)
    with pytest.raises(ValueError):
        schema.validate_stage4_output_npz(payload)


def test_loaders_roundtrip_npz(tmp_path) -> None:
    stage1_path = tmp_path / 'x.prob.npz'
    np.savez_compressed(stage1_path, **_stage1_payload())
    s1 = schema.load_stage1_prob(stage1_path)
    assert s1.prob.shape[0] == int(s1.ffid_values.shape[0])

    sidecar_path = tmp_path / 'x.sidecar.npz'
    np.savez_compressed(sidecar_path, **_stage2_training_payload())
    s2 = schema.load_stage2_sidecar(sidecar_path)
    assert s2.training is not None

    stage4_path = tmp_path / 'x.psn_pred.npz'
    np.savez_compressed(stage4_path, **_stage4_payload())
    s4 = schema.load_stage4_output(stage4_path)
    assert s4.n_traces == 4
