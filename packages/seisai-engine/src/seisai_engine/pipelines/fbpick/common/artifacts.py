from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from seisai_engine.infer.segy2segy_cli_common import cfg_hash

ROBUST_SOURCE_COARSE_OBSERVED = 0
ROBUST_SOURCE_THEORETICAL = 1
ROBUST_SOURCE_TREND_FILL = 2

COARSE_REQUIRED_KEYS = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'ffid_values',
    'chno_values',
    'offsets_m',
    'trace_indices',
    'coarse_pick_i',
    'coarse_pick_t_sec',
    'coarse_pmax',
    'coarse_prob_summary',
    'lineage',
)

COARSE_GEOMETRY_OPTIONAL_KEYS = (
    'source_x_m',
    'source_y_m',
    'receiver_x_m',
    'receiver_y_m',
    'offset_abs_geom_m',
    'geometry_valid_mask',
)

COARSE_GEOMETRY_EXTRA_OPTIONAL_KEYS = (
    'offset_signed_geom_m',
)

COARSE_GEOMETRY_SCALE_KEY = 'geometry_coord_unit_scale_to_m'

ROBUST_REQUIRED_KEYS = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'ffid_values',
    'chno_values',
    'offsets_m',
    'trace_indices',
    'robust_pick_i',
    'robust_pick_t_sec',
    'robust_conf',
    'robust_source',
    'used_theoretical_mask',
    'reason_mask',
    'conf_prob1',
    'conf_trend1',
    'conf_rs1',
    'lineage',
)

ROBUST_CENTER_OPTIONAL_KEYS = (
    'trend_center_i',
    'trend_center_t_sec',
    'physical_center_i',
    'physical_center_t_sec',
    'fine_center_i',
    'fine_center_t_sec',
)

ROBUST_PHYSICAL_DIAGNOSTIC_OPTIONAL_KEYS = (
    'physical_model_status',
    'physical_model_failure_reason',
    'physical_model_break_offset_m',
    'physical_model_slope_near_s_per_m',
    'physical_model_slope_far_s_per_m',
    'physical_model_velocity_near_m_s',
    'physical_model_velocity_far_m_s',
    'physical_model_neighbor_count',
    'physical_prefilter_valid_count',
    'physical_model_segment_id',
    'physical_model_side',
    'physical_model_resid_p50_ms',
    'physical_model_resid_p90_ms',
)

ROBUST_OPTIONAL_KEYS = (
    *ROBUST_CENTER_OPTIONAL_KEYS,
    *ROBUST_PHYSICAL_DIAGNOSTIC_OPTIONAL_KEYS,
)

ROBUST_PHYSICAL_OPTIONAL_KEYS = ROBUST_OPTIONAL_KEYS

FINE_RESULT_REQUIRED_KEYS = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'trace_indices',
    'fine_pick_local_i',
    'fine_pick_local_f',
    'fine_pmax',
    'final_pick_i',
    'final_pick_f',
    'final_pick_t_sec',
    'final_conf',
    'window_start_i',
    'window_end_i',
)

FINAL_REQUIRED_KEYS = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'ffid_values',
    'chno_values',
    'offsets_m',
    'trace_indices',
    'coarse_pick_i',
    'coarse_pmax',
    'robust_pick_i',
    'robust_conf',
    'robust_source',
    'used_theoretical_mask',
    'reason_mask',
    'window_start_i',
    'window_end_i',
    'fine_pick_local_f',
    'fine_pick_local_i',
    'fine_pmax',
    'final_pick_f',
    'final_pick_i',
    'final_pick_t_sec',
    'final_conf',
    'high_conf_mask',
    'reject_mask',
    'lineage',
)

ROBUST_SOURCE_LABELS = {
    ROBUST_SOURCE_COARSE_OBSERVED: 'coarse_observed',
    ROBUST_SOURCE_THEORETICAL: 'theoretical_replacement',
    ROBUST_SOURCE_TREND_FILL: 'trend_or_global_fill',
}

REASON_MASK_INFEASIBLE = 1 << 0
REASON_MASK_LOW_SCORE = 1 << 1
REASON_MASK_FILLED_FROM_TREND = 1 << 2

REASON_MASK_LABELS = {
    REASON_MASK_INFEASIBLE: 'infeasible',
    REASON_MASK_LOW_SCORE: 'low_score',
    REASON_MASK_FILLED_FROM_TREND: 'filled_from_trend',
}

__all__ = [
    'COARSE_GEOMETRY_EXTRA_OPTIONAL_KEYS',
    'COARSE_GEOMETRY_OPTIONAL_KEYS',
    'COARSE_GEOMETRY_SCALE_KEY',
    'COARSE_REQUIRED_KEYS',
    'FINAL_REQUIRED_KEYS',
    'FINE_RESULT_REQUIRED_KEYS',
    'REASON_MASK_FILLED_FROM_TREND',
    'REASON_MASK_INFEASIBLE',
    'REASON_MASK_LABELS',
    'REASON_MASK_LOW_SCORE',
    'ROBUST_CENTER_OPTIONAL_KEYS',
    'ROBUST_OPTIONAL_KEYS',
    'ROBUST_PHYSICAL_DIAGNOSTIC_OPTIONAL_KEYS',
    'ROBUST_REQUIRED_KEYS',
    'ROBUST_PHYSICAL_OPTIONAL_KEYS',
    'ROBUST_SOURCE_COARSE_OBSERVED',
    'ROBUST_SOURCE_LABELS',
    'ROBUST_SOURCE_THEORETICAL',
    'ROBUST_SOURCE_TREND_FILL',
    'build_lineage_payload',
    'read_git_sha',
]


def _resolve_git_dir(repo_root: Path) -> Path | None:
    resolved_root = repo_root.resolve()
    for candidate_root in (resolved_root, *resolved_root.parents):
        git_path = candidate_root / '.git'
        if git_path.is_dir():
            return git_path
        if not git_path.is_file():
            continue

        text = git_path.read_text(encoding='utf-8').strip()
        prefix = 'gitdir:'
        if not text.startswith(prefix):
            msg = f'unsupported .git file format: {git_path}'
            raise ValueError(msg)
        rel = text[len(prefix) :].strip()
        git_dir = Path(rel)
        if not git_dir.is_absolute():
            git_dir = (candidate_root / git_dir).resolve()
        if not git_dir.is_dir():
            msg = f'git dir not found: {git_dir}'
            raise FileNotFoundError(msg)
        return git_dir
    return None


def _lookup_packed_ref(*, git_dir: Path, ref_name: str) -> str:
    packed_refs = git_dir / 'packed-refs'
    if not packed_refs.is_file():
        msg = f'git ref not found: {ref_name}'
        raise FileNotFoundError(msg)

    for line in packed_refs.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('^'):
            continue
        sha, name = stripped.split(' ', maxsplit=1)
        if name == ref_name:
            return sha

    msg = f'git ref not found: {ref_name}'
    raise FileNotFoundError(msg)


def read_git_sha(repo_root: Path | None = None) -> str | None:
    if repo_root is None:
        repo_root = Path.cwd()
    if not isinstance(repo_root, Path):
        msg = 'repo_root must be Path'
        raise TypeError(msg)

    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return None
    head_path = git_dir / 'HEAD'
    if not head_path.is_file():
        msg = f'git HEAD not found: {head_path}'
        raise FileNotFoundError(msg)

    head = head_path.read_text(encoding='utf-8').strip()
    if not head:
        msg = f'git HEAD is empty: {head_path}'
        raise ValueError(msg)

    if head.startswith('ref:'):
        ref_name = head[len('ref:') :].strip()
        ref_path = git_dir / ref_name
        if ref_path.is_file():
            sha = ref_path.read_text(encoding='utf-8').strip()
        else:
            sha = _lookup_packed_ref(git_dir=git_dir, ref_name=ref_name)
    else:
        sha = head

    if len(sha) < 7:
        msg = f'invalid git sha: {sha!r}'
        raise ValueError(msg)
    return sha


def _normalize_iter_id(iter_id: int | str | None) -> int | str | None:
    if iter_id is None:
        return None
    if isinstance(iter_id, str):
        return iter_id
    return int(iter_id)


def build_lineage_payload(
    cfg: dict[str, Any],
    *,
    repo_root: Path | None = None,
    source_model_id: str | None,
    iter_id: int | str | None,
) -> np.ndarray:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    payload = {
        'iter_id': _normalize_iter_id(iter_id),
        'source_model_id': source_model_id,
        'cfg_hash': cfg_hash(cfg),
        'git_sha': read_git_sha(repo_root),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    return np.asarray(encoded)
