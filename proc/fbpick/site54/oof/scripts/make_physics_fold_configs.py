#!/usr/bin/env python3
"""Render site54 OOF physics and physics QC configs."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

FOLDS = [f"fold{i:02d}" for i in range(6)]


def parse_bool(value: object) -> bool:
    """Parse a command-line boolean value."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        msg = f'expected boolean value, got {value!r}'
        raise argparse.ArgumentTypeError(msg)
    lowered = value.lower()
    if lowered in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if lowered in {'0', 'false', 'no', 'n', 'off'}:
        return False
    msg = f'expected boolean value, got {value!r}'
    raise argparse.ArgumentTypeError(msg)


def partial_trend_fallback_cfg(args: argparse.Namespace) -> dict:
    """Build the partial existing-trend fallback policy."""
    return {
        'enabled': True,
        'max_fraction': float(args.partial_trend_fallback_max_fraction),
        'max_traces': int(args.partial_trend_fallback_max_traces),
        'cluster_consecutive_indices': True,
        'use_global_fallback': True,
        'fallback_if_too_many': str(args.partial_trend_fallback_if_too_many),
        'local_window_from_trend_config': True,
        'emit_progress': True,
    }


def physics_runtime_cfg(args: argparse.Namespace) -> dict:
    """Build the baseline physical-center runtime config."""
    return {
        'fit_policy': 'anchor_source_xy',
        'fallback_existing_trend_mode': str(args.fallback_existing_trend_mode),
        'partial_trend_fallback': partial_trend_fallback_cfg(args),
        'diagnostics_enabled': True,
        'write_runtime_summary': True,
        'diagnostics': {
            'enabled': True,
            'detailed_timing': False,
            'save_json': True,
            'save_npz_scalars': True,
            'save_per_trace_context': False,
        },
        'observation_sampling': {
            'enabled': True,
            'method': 'offset_bin',
            'max_obs_per_fit': 64,
            'n_offset_bins': 64,
            'bin_pick': 'pmax_max',
            'min_obs_per_fit_after_sampling': 16,
            'preserve_edge_bins': True,
        },
        'fit_executor': {
            'enabled': True,
            'backend': 'thread',
            'max_workers': 4,
            'torch_num_threads_per_worker': 1,
            'chunksize': 4,
        },
        'anchor_selection': {
            'enabled': True,
            'mode': 'source_xy_stride',
            'anchor_stride_source_groups': 5,
            'anchor_spacing_m': None,
            'include_first': True,
            'include_last': True,
        },
        'anchor_reuse': {
            'enabled': True,
            'non_anchor_mode': 'nearest_anchor_plus_t0_shift',
            'max_anchor_distance_m': None,
            'reuse_segment_policy': 'same_side_and_gap',
            'fallback_if_no_compatible_segment': 'robust',
        },
        't0_shift': {
            'enabled': True,
            'estimator': 'median',
            'min_valid_for_t0_shift': 8,
            't0_shift_clip_ms': 60.0,
            'use_physical_prefilter_mask': True,
            'use_pmax_min': True,
        },
        'adaptive_refit': {
            'enabled': False,
            'resid_p90_ms_gt': 50.0,
            'median_abs_shift_ms_gt': 40.0,
            'min_valid_for_resid_check': 8,
            'fallback_if_refit_fails': 'nearest_anchor_plus_t0_shift',
        },
        'progress': {
            'enabled': True,
            'level': 'fit',
            'interval_sec': 5.0,
            'min_interval_fit_calls': 10,
            'stream': 'stderr',
            'use_tqdm': 'auto',
        },
    }


def physical_center_cfg(args: argparse.Namespace, *, qc: bool = False) -> dict:
    """Build physical-center settings shared by physics and QC configs."""
    cfg = {
        'physical_trend': {
            'enabled': True,
            'fit_kind': 'two_piece_irls_autobreak',
            'use_geometry_offset': True,
            'min_offset_spread_m': 1.0,
            'coord_group_tol_m': 1.0,
            'segment_by_offset_sign': True,
            'split_by_offset_gap': True,
            'gap_ratio': 5.0,
            'min_gap_m': None,
        },
        'neighbor_context': {
            'enabled': True,
            'mode': 'nearest_source_xy',
            'k_neighbors': 5 if qc else 3,
            'max_source_distance_m': None,
            'include_self': True,
        },
        'physical_prefilter': {
            'enabled': True,
            'vmin_m_s': 300.0,
            'vmax_m_s': 6000.0,
            't0_lo_ms': -20.0,
            't0_hi_ms': 200.0,
            'pmax_min': 0.0 if qc else 0.05,
            'use_existing_feasible_mask': False,
        },
        'physical_projection': {'mode': 'model'},
        'two_piece_ransac': {
            'n_iter': 200,
            'inlier_th_ms': 40.0,
            'min_pts': 8,
            'n_break_cand': 64,
            'q_lo': 0.15,
            'q_hi': 0.85,
            'seed': 0,
            'slope_eps': 1.0e-6,
            'sort_offsets': True,
        },
        'two_piece_irls': {
            'huber_c': 1.345,
            'iters': 5 if qc else 3,
            'min_pts': 8,
            'n_break_cand': 64 if qc else 24,
            'q_lo': 0.15,
            'q_hi': 0.85,
            'slope_eps': 1.0e-6,
            'sort_offsets': True,
        },
    }
    runtime = physics_runtime_cfg(args)
    if qc:
        runtime.update(
            {
                'fit_policy': 'full',
                'trend_result_mode': 'lazy',
                'geometry_invalid_fallback': 'robust',
                'group_invalid_fallback': 'robust',
            }
        )
        runtime['diagnostics']['detailed_timing'] = True
        runtime['observation_sampling']['enabled'] = False
        runtime_keys = (
            'fit_executor',
            'anchor_selection',
            'anchor_reuse',
            't0_shift',
            'adaptive_refit',
        )
        for key in runtime_keys:
            runtime.pop(key, None)
        runtime['progress']['include_stage_events'] = True
        runtime['progress']['include_summary'] = True
        runtime['progress']['print_on_non_tty'] = True
    else:
        runtime.update(
            {
                'trend_result_mode': 'lazy',
                'geometry_invalid_fallback': 'robust',
                'group_invalid_fallback': 'robust',
                'legacy_trend_output': 'auto',
            }
        )
        runtime['progress']['include_stage_events'] = True
        runtime['progress']['include_summary'] = True
        runtime['progress']['print_on_non_tty'] = True
    cfg['physical_runtime'] = runtime
    return cfg


def physics_config(args: argparse.Namespace, fold: str) -> dict:
    """Build one fold's robust physics config."""
    fold_root = args.fold_list_root / 'folds' / fold
    return {
        'paths': {
            'segy_files': str(fold_root / 'heldout_sgy.txt'),
            'coarse_npz_dir': str(args.run_root / fold / '02_coarse_infer'),
            'out_dir': str(args.run_root / fold / '03_physics'),
        },
        'feasible_band': {},
        'trend': {},
        'residual_statics': {},
        'keep_reject': {},
        'robust_center': {},
        **physical_center_cfg(args, qc=False),
    }


def physics_qc_config(args: argparse.Namespace, fold: str) -> dict:
    """Build one fold's physics QC config."""
    fold_root = args.fold_list_root / 'folds' / fold
    return {
        'paths': {
            'segy_files': str(fold_root / 'heldout_sgy.txt'),
            'fb_files': str(fold_root / 'heldout_fb.txt'),
            'coarse_npz_dir': str(args.run_root / fold / '02_coarse_infer'),
            'robust_npz_dir': str(args.run_root / fold / '03_physics'),
            'out_dir': str(args.run_root / fold / '04_physics_qc'),
        },
        'dataset': {
            'primary_keys': ['ffid'],
            'infer_endian': args.infer_endian,
            'use_header_cache': True,
        },
        'vis': {
            'max_gathers_per_file': 8,
            'gather_selection': 'even',
            'skip_gather_keys': {'ffid': [0]},
            'max_traces_per_gather': 10000,
            'save_cdf': True,
            'save_summary_csv': True,
            'waveform_norm': 'per_trace',
            'clip_percentile': 99.0,
            'first_panel_only': True,
            'auto_figsize': True,
            'traces_per_inch': 160.0,
            'samples_per_inch': 550.0,
            'min_fig_width': 7.0,
            'max_fig_width': 14.0,
            'min_fig_height': 5.5,
            'max_fig_height': 12.0,
            'min_panel_aspect': 0.9,
            'max_panel_aspect': 1.8,
            'max_display_traces': 1200,
            'overlays': {
                'gt_pick': True,
                'coarse_pick': True,
                'robust_pick': False,
                'trend_center': True,
                'physical_center': True,
                'fine_center': True,
                'window': True,
                'final_pick': True,
                'physical_model_status': True,
            },
        },
        **physical_center_cfg(args, qc=True),
    }


def write_yaml(path: Path, data: dict, *, overwrite: bool) -> None:
    """Write one YAML file unless it exists and overwrite is disabled."""
    if path.exists() and not overwrite:
        print(f'[skip] exists: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    print(f'[write] {path}')


def main() -> None:
    """Render all fold configs."""
    parser = argparse.ArgumentParser(
        description='Render site54 OOF physics batch configs.'
    )
    parser.add_argument(
        '--cv-root',
        type=Path,
        default=Path('/workspace/proc/fbpick/site54/oof'),
    )
    parser.add_argument('--run-id', default='baseline_physical_center')
    parser.add_argument('--run-root', type=Path, default=None)
    parser.add_argument('--fold-list-root', type=Path, default=None)
    parser.add_argument('--config-root', type=Path, default=None)
    parser.add_argument(
        '--legacy-flat-configs',
        nargs='?',
        const=True,
        default=False,
        type=parse_bool,
    )
    parser.add_argument('--infer-endian', default='big')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--fallback-existing-trend-mode',
        choices=('full', 'partial'),
        default='partial',
    )
    parser.add_argument(
        '--partial-trend-fallback-max-fraction',
        type=float,
        default=0.05,
    )
    parser.add_argument(
        '--partial-trend-fallback-max-traces',
        type=int,
        default=50000,
    )
    parser.add_argument(
        '--partial-trend-fallback-if-too-many',
        choices=('robust', 'full', 'error'),
        default='robust',
    )
    args = parser.parse_args()

    args.cv_root = args.cv_root.resolve()
    args.run_root = (args.run_root or args.cv_root / 'runs' / args.run_id).resolve()
    args.fold_list_root = (args.fold_list_root or args.cv_root / 'fold_lists').resolve()
    args.config_root = (args.config_root or args.run_root / 'configs').resolve()

    for fold in FOLDS:
        heldout_sgy = args.fold_list_root / 'folds' / fold / 'heldout_sgy.txt'
        heldout_fb = args.fold_list_root / 'folds' / fold / 'heldout_fb.txt'
        if not heldout_sgy.is_file():
            raise FileNotFoundError(heldout_sgy)
        if not heldout_fb.is_file():
            raise FileNotFoundError(heldout_fb)

        physics = physics_config(args, fold)
        physics_qc = physics_qc_config(args, fold)
        fold_config_dir = args.config_root / fold
        write_yaml(
            fold_config_dir / '03_physics.yaml',
            physics,
            overwrite=args.overwrite,
        )
        write_yaml(
            fold_config_dir / '04_physics_qc.yaml',
            physics_qc,
            overwrite=args.overwrite,
        )
        if args.legacy_flat_configs:
            write_yaml(
                args.config_root / f'config_run_fbpick_physics_{fold}_heldout.yaml',
                physics,
                overwrite=args.overwrite,
            )
            write_yaml(
                args.config_root / f'config_run_fbpick_physics_qc_{fold}.yaml',
                physics_qc,
                overwrite=args.overwrite,
            )


if __name__ == '__main__':
    main()
