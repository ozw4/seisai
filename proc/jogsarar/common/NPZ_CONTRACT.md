# Jogsarar NPZ Contract (Stage1 / Stage2 / Stage4)

This document captures NPZ schemas exchanged by:
- `proc/jogsarar/stage1_fbp_infer_raw.py`
- `proc/jogsarar/stage2_make_psn512_windows.py`
- `proc/jogsarar/stage4_psn512_infer_to_raw.py`

No processing behavior is changed by this contract; this is documentation-only for shared schema stability.

## Axis and symbol conventions

| Symbol | Meaning |
|---|---|
| `Ntr` | Number of traces in one SEG-Y file (`tracecount`) |
| `NsRaw` | Number of samples in raw SEG-Y (`n_samples_orig` / `n_samples_in`) |
| `Wpad` | Stage1 padded width (`cfg.tile_w`, default `6016`) |
| `Wwin` | Stage2/Stage4 window width (`2 * half_win * up_factor`, default `512`) |
| `Nffid` | Number of unique FFIDs in one file |
| `NnzP` | Number of kept P picks in CSR artifact |

Notes:
- Stage loops use gather-local `H` (trace direction inside an FFID gather), but saved NPZ arrays are file-global and indexed by `trace_indices` (`0..Ntr-1`).
- Sample axis is always second axis when 2D (`(Ntr, samples)`).

## File-level contracts

| Artifact | Producer | Consumer | Naming / path rule |
|---|---|---|---|
| Stage1 probability NPZ | Stage1 | Stage2 | For raw `in_segy_root/<rel>/<stem>.sgy`, Stage1 writes `out_infer_root/<rel>/<stem>.prob.npz` |
| Stage2 sidecar NPZ | Stage2 | Stage4 | For Stage2 win SEG-Y `<stem>.win512.sgy`, sidecar path is `<stem>.win512.sidecar.npz` (`with_suffix('.sidecar.npz')`) |
| Stage2 phase-pick CSR NPZ (training artifact) | Stage2 | Training dataset loader (not Stage4) | `<stem>.win512.phase_pick.csr.npz` |
| Stage4 prediction NPZ | Stage4 | Downstream QC/analysis | For raw `<stem>.sgy`, output is `out_pred_root/<rel>/<stem>.psn_pred.npz` |

Stage4 sidecar lookup:
- First candidate: `win_path.with_suffix('.sidecar.npz')`
- Fallback candidate (legacy naming): `win_path.with_suffix('.win512.sidecar.npz')` if stem does not already end with `.win512`

## Stage1 `.prob.npz` schema

Required keys: all keys in this table are always written.

| Key | Required | Dtype family | Shape | Meaning |
|---|---|---|---|---|
| `prob` | Yes | float (`float16` saved) | `(Ntr, Wpad)` | Per-trace P probability over padded sample axis; values after `NsRaw` are zero |
| `dt_sec` | Yes | float (`float32`) | `()` | Raw sample interval in seconds |
| `n_samples_orig` | Yes | int (`int32`) | `()` | Raw sample count (`NsRaw`) |
| `ffid_values` | Yes | int (`int32`) | `(Ntr,)` | FFID per trace |
| `chno_values` | Yes | int (`int32`) | `(Ntr,)` | Channel number per trace |
| `offsets` | Yes | float (`float32`) | `(Ntr,)` | Offset header per trace (meters) |
| `trace_indices` | Yes | int (`int64`) | `(Ntr,)` | File trace index (`0..Ntr-1`) |
| `pick0` | Yes | int (`int32`) | `(Ntr,)` | Argmax pick before RS/final snap (0 means no-pick) |
| `pick_pre_snap` | Yes | int (`int32`) | `(Ntr,)` | Pre-snap pick used for RS baseline when RS base is `snap` |
| `delta_pick` | Yes | float (`float32`) | `(Ntr,)` | Residual-statics delta (samples) |
| `pick_ref` | Yes | float (`float32`) | `(Ntr,)` | RS-corrected pick (float samples) |
| `pick_ref_i` | Yes | int (`int32`) | `(Ntr,)` | Rounded `pick_ref` |
| `pick_final` | Yes | int (`int32`) | `(Ntr,)` | Final Stage1 pick (raw sample index) |
| `cmax` | Yes | float (`float32`) | `(Ntr,)` | RS correlation max metric |
| `score` | Yes | float (`float32`) | `(Ntr,)` | RS score metric |
| `rs_valid_mask` | Yes | bool | `(Ntr,)` | RS valid flag per trace |
| `conf_prob0` | Yes | float (`float32`) | `(Ntr,)` | Probability confidence around `pick_pre_snap` |
| `conf_prob1` | Yes | float (`float32`) | `(Ntr,)` | Probability confidence around `pick_final` |
| `conf_trend0` | Yes | float (`float32`) | `(Ntr,)` | Trend confidence at `pick_pre_snap` |
| `conf_trend1` | Yes | float (`float32`) | `(Ntr,)` | Trend confidence at `pick_final` |
| `conf_rs1` | Yes | float (`float32`) | `(Ntr,)` | RS confidence at final pick |
| `trend_t_sec` | Yes | float (`float32`) | `(Ntr,)` | Local trendline time in seconds |
| `trend_covered` | Yes | bool | `(Ntr,)` | Local trendline coverage mask |
| `trend_offset_signed_proxy` | Yes | float (`float32`) | `(Ntr,)` | Signed proxy offset used for split-side trend |
| `trend_split_index` | Yes | int (`int32`) | `(Ntr,)` | Split index for side split; `-1` when not used |
| `trend_source` | Yes | string | `()` | Trend source label (config string) |
| `trend_method` | Yes | string | `()` | Trend method label (config string) |
| `trend_cfg` | Yes | string | `()` | Serialized trend config summary |

Stage2-required subset from Stage1:
- `pick_final`
- `conf_prob1`
- `conf_rs1`
- `trend_t_sec`
- `trend_covered`
- `dt_sec`

## Stage2 `.sidecar.npz` schema

### Base keys (always required, inference + training)

| Key | Required | Dtype family | Shape | Meaning |
|---|---|---|---|---|
| `src_segy` | Yes | string | `()` | Source raw SEG-Y path |
| `src_infer_npz` | Yes | string | `()` | Source Stage1 NPZ path |
| `out_segy` | Yes | string | `()` | Output win512 SEG-Y path |
| `dt_sec_in` | Yes | float (`float32`) | `()` | Raw sample interval (sec) |
| `dt_sec_out` | Yes | float (`float32`) | `()` | Win512 sample interval (sec) |
| `dt_us_in` | Yes | int (`int32`) | `()` | Raw sample interval (microseconds) |
| `dt_us_out` | Yes | int (`int32`) | `()` | Win512 sample interval (microseconds) |
| `n_traces` | Yes | int (`int32`) | `()` | `Ntr` |
| `n_samples_in` | Yes | int (`int32`) | `()` | `NsRaw` |
| `n_samples_out` | Yes | int (`int32`) | `()` | `Wwin` |
| `window_start_i` | Yes | int (`int64`) | `(Ntr,)` | Window start sample on raw axis for each trace |

Stage4 reads `window_start_i` and validates consistency with `n_traces`, `n_samples_in`, `n_samples_out`, `dt_sec_in`, `dt_sec_out`.

### Training keys (present only when `emit_training_artifacts=True`)

| Key | Required in training mode | Dtype family | Shape | Meaning |
|---|---|---|---|---|
| `out_pick_csr_npz` | Yes | string | `()` | Path to `.phase_pick.csr.npz` |
| `thresh_mode` | Yes | string | `()` | `global` or `per_segy` |
| `drop_low_frac` | Yes | float (`float32`) | `()` | Low-score drop fraction |
| `local_global_diff_th_samples` | Yes | int (`int32`) | `()` | Local/global trend diff threshold (samples) |
| `local_discard_radius_traces` | Yes | int (`int32`) | `()` | Neighborhood expansion radius (traces) |
| `trend_center_i_raw` | Yes | float (`float32`) | `(Ntr,)` | Placeholder raw trend center index (currently NaN-filled) |
| `trend_center_i_local` | Yes | float (`float32`) | `(Ntr,)` | Stage1 local trend center index |
| `trend_center_i_final` | Yes | float (`float32`) | `(Ntr,)` | Global trend center index after missing fill |
| `trend_center_i_used` | Yes | float (`float32`) | `(Ntr,)` | Final trend center used for cropping |
| `trend_center_i_global` | Yes | float (`float32`) | `(Ntr,)` | Global trend center index |
| `nn_replaced_mask` | Yes | bool | `(Ntr,)` | NN replacement mask (currently all false) |
| `global_replaced_mask` | Yes | bool | `(Ntr,)` | Local trend values replaced by global/interp |
| `global_missing_filled_mask` | Yes | bool | `(Ntr,)` | Global trend missing filled inside FFID |
| `global_edges_all` | Yes | float (`float32`) | `(3,)` | Piecewise fit edges for all traces |
| `global_coef_all` | Yes | float (`float32`) | `(2,2)` | Piecewise fit coefficients for all traces |
| `global_edges_left` | Yes | float (`float32`) | `(3,)` | Piecewise fit edges (left side proxy<0) |
| `global_coef_left` | Yes | float (`float32`) | `(2,2)` | Piecewise fit coefficients (left side) |
| `global_edges_right` | Yes | float (`float32`) | `(3,)` | Piecewise fit edges (right side proxy>0) |
| `global_coef_right` | Yes | float (`float32`) | `(2,2)` | Piecewise fit coefficients (right side) |
| `trend_center_i` | Yes | float (`float32`) | `(Ntr,)` | Alias of `trend_center_i_used` |
| `trend_filled_mask` | Yes | bool | `(Ntr,)` | Final trend fill mask |
| `trend_center_i_round` | Yes | int (`int64`) | `(Ntr,)` | Rounded trend center for crop (`-1` invalid) |
| `ffid_values` | Yes | int (`int64`) | `(Ntr,)` | FFID per trace |
| `ffid_unique_values` | Yes | int (`int64`) | `(Nffid,)` | Unique FFIDs |
| `shot_x_ffid` | Yes | float (`float64`) | `(Nffid,)` | Shot X per FFID |
| `shot_y_ffid` | Yes | float (`float64`) | `(Nffid,)` | Shot Y per FFID |
| `pick_final_i` | Yes | int (`int64`) | `(Ntr,)` | Stage1 final pick on raw axis |
| `pick_win_512` | Yes | float (`float32`) | `(Ntr,)` | Pick mapped to win512 axis: `(pick_final_i - window_start_i) * up_factor`; dropped traces are `NaN` |
| `keep_mask` | Yes | bool | `(Ntr,)` | Label keep flag |
| `reason_mask` | Yes | uint8/int | `(Ntr,)` | Bit-mask reason for drop |
| `th_conf_prob1` | Yes | float (`float32`) | `()` | Applied threshold for `conf_prob1` |
| `th_conf_trend1` | Yes | float (`float32`) | `()` | Applied threshold for `conf_trend1` |
| `th_conf_rs1` | Yes | float (`float32`) | `()` | Applied threshold for `conf_rs1` |
| `conf_prob1` | Yes | float (`float32`) | `(Ntr,)` | Stage1 confidence copied for filtering |
| `conf_trend1` | Yes | float (`float32`) | `(Ntr,)` | Recomputed trend confidence |
| `conf_rs1` | Yes | float (`float32`) | `(Ntr,)` | Stage1 RS confidence copied for filtering |

## Stage2 `.phase_pick.csr.npz` schema (training artifact)

| Key | Required | Dtype family | Shape | Meaning |
|---|---|---|---|---|
| `n_traces` | Yes | int (`int32`) | `()` | `Ntr` |
| `p_indptr` | Yes | int (`int64`) | `(Ntr+1,)` | CSR row pointer for P picks |
| `p_data` | Yes | int (`int64`) | `(NnzP,)` | Win512 sample pick index per kept trace |
| `s_indptr` | Yes | int (`int64`) | `(Ntr+1,)` | CSR row pointer for S picks (currently empty rows) |
| `s_data` | Yes | int (`int64`) | `(0,)` | S picks (currently empty) |

## Stage4 `.psn_pred.npz` schema

Required keys: all keys in this table are always written.

| Key | Required | Dtype family | Shape | Meaning |
|---|---|---|---|---|
| `dt_sec` | Yes | float (`float32`) | `()` | Raw sample interval in seconds |
| `n_samples_orig` | Yes | int (`int32`) | `()` | Raw sample count (`NsRaw`) |
| `n_traces` | Yes | int (`int32`) | `()` | `Ntr` |
| `ffid_values` | Yes | int (`int32`) | `(Ntr,)` | FFID per trace |
| `chno_values` | Yes | int (`int32`) | `(Ntr,)` | Channel number per trace |
| `offsets` | Yes | float (`float32`) | `(Ntr,)` | Offset header per trace |
| `trace_indices` | Yes | int (`int64`) | `(Ntr,)` | File trace index |
| `pick_psn512` | Yes | int (`int32`) | `(Ntr,)` | Argmax pick on win512 sample axis |
| `pmax_psn` | Yes | float (`float32`) | `(Ntr,)` | Max P probability at `pick_psn512` |
| `window_start_i` | Yes | int (`int64`) | `(Ntr,)` | Copied from sidecar; raw window start index |
| `pick_psn_orig_f` | Yes | float (`float32`) | `(Ntr,)` | Win->raw mapped pick (float): `window_start_i + pick_psn512/up_factor` |
| `pick_psn_orig_i` | Yes | int (`int32`) | `(Ntr,)` | Rounded `pick_psn_orig_f` |
| `delta_pick_rs` | Yes | float (`float32`) | `(Ntr,)` | Residual-statics delta on raw axis |
| `cmax_rs` | Yes | float (`float32`) | `(Ntr,)` | RS correlation max metric |
| `rs_valid_mask` | Yes | bool | `(Ntr,)` | RS valid flag |
| `pick_rs_i` | Yes | int (`int32`) | `(Ntr,)` | RS-corrected pick on raw axis |
| `pick_final` | Yes | int (`int32`) | `(Ntr,)` | Final Stage4 pick on raw axis |
