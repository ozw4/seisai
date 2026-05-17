"""Observation planning for physical-center fitting."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from .config import PhysicsLiteConfig
from .geometry import (
    CoarseGeometry,
    SourceGroup,
    select_nearest_source_groups,
    signed_offset_side_from_geometry,
    split_offset_gap_segments,
)
from .physical_center_geometry import (
    _signed_offset_side_labels,
    _SignedOffsetSideLabels,
)
from .runtime_diagnostics import PhysicalRuntimeDiagnostics


@dataclass(frozen=True)
class _SideObservationContext:
    obs_indices_by_side: tuple[np.ndarray, np.ndarray, np.ndarray]
    obs_key_by_side: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    finite_count: int
    nonzero_count: int


@dataclass(frozen=True)
class _GroupObservationContext:
    group_id: int
    neighbor_group_ids: np.ndarray
    neighbor_indices: np.ndarray
    valid_obs_indices: np.ndarray
    valid_obs_key: tuple[int, ...]
    neighbor_count: int
    prefilter_valid_count: int
    side_context: _SideObservationContext | None = None


@dataclass(frozen=True)
class _ObservationPlan:
    obs_indices: np.ndarray
    obs_key: tuple[int, ...] | None
    neighbor_count: int
    prefilter_valid_count: int
    segment_id: int
    side: int
    relaxed: bool


@dataclass(frozen=True)
class _GapSegmentContext:
    segment_id_by_trace_idx: dict[int, int]
    obs_indices_by_segment_id: dict[int, np.ndarray]
    obs_key_by_segment_id: dict[int, tuple[int, ...]]


@dataclass
class _ObservationPlanCache:
    offset_signed_labels: _SignedOffsetSideLabels | None = None
    index_members: dict[tuple[int, ...], frozenset[int]] = field(
        default_factory=dict,
    )
    signed_side_context: dict[
        tuple[int, tuple[int, ...]], _SideObservationContext
    ] = field(default_factory=dict)
    geometry_side: dict[
        tuple[tuple[int, ...], int],
        tuple[np.ndarray, tuple[int, ...], int, bool],
    ] = field(default_factory=dict)
    gap_segment: dict[
        tuple[tuple[int, ...], int, float, float | None],
        tuple[np.ndarray, tuple[int, ...], int],
    ] = field(default_factory=dict)
    gap_context: dict[
        tuple[tuple[int, ...], float, float | None],
        _GapSegmentContext,
    ] = field(default_factory=dict)


def _stable_unique(indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    if arr.ndim != 1:
        msg = 'indices must be 1D'
        raise ValueError(msg)
    seen: set[int] = set()
    out: list[int] = []
    for value in arr.tolist():
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return np.asarray(out, dtype=np.int64)


def _append_index(indices: np.ndarray, trace_idx: int) -> np.ndarray:
    return _stable_unique(
        np.concatenate(
            [
                np.asarray(indices, dtype=np.int64),
                np.asarray([int(trace_idx)], dtype=np.int64),
            ]
        )
    )


def _indices_key(indices: np.ndarray) -> tuple[int, ...]:
    return tuple(np.asarray(indices, dtype=np.int64).tolist())


def _concat_group_traces(
    group_ids: np.ndarray,
    *,
    groups_by_id: Mapping[int, SourceGroup],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for group_id in np.asarray(group_ids, dtype=np.int64).tolist():
        group = groups_by_id.get(int(group_id))
        if group is None:
            continue
        chunks.append(np.asarray(group.trace_indices, dtype=np.int64))
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return _stable_unique(np.concatenate(chunks))


def _trace_position_map(indices: np.ndarray) -> dict[int, int]:
    return {int(trace_idx): int(pos) for pos, trace_idx in enumerate(indices.tolist())}


def _obs_key_or_build(
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if obs_key is not None:
        return obs_key
    return _indices_key(obs_indices)


def _index_key_contains(
    *,
    obs_key: tuple[int, ...],
    trace_idx: int,
    cache: _ObservationPlanCache,
) -> bool:
    members = cache.index_members.get(obs_key)
    if members is None:
        members = frozenset(obs_key)
        cache.index_members[obs_key] = members
    return int(trace_idx) in members


def _side_slot(side: int) -> int:
    value = int(side)
    if value < -1 or value > 1:
        msg = 'side must be -1, 0, or 1'
        raise ValueError(msg)
    return value + 1


def _filtered_obs_key(
    *,
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...],
    keep_mask: np.ndarray,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    obs = np.asarray(obs_indices, dtype=np.int64)
    mask = np.asarray(keep_mask, dtype=np.bool_)
    if mask.shape != obs.shape:
        msg = 'keep_mask must match obs_indices shape'
        raise ValueError(msg)
    if bool(np.all(mask)):
        return obs, obs_key
    filtered = obs[mask]
    with (
        runtime_diagnostics.time_block('side_segment_key_build_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        filtered_key = _indices_key(filtered)
    return filtered, filtered_key


def _build_side_observation_context(
    *,
    labels: _SignedOffsetSideLabels,
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...],
    cache: _ObservationPlanCache,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> _SideObservationContext:
    cache_key = (id(labels.side), obs_key)
    cached = cache.signed_side_context.get(cache_key)
    if cached is not None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_side_context_cache_hits')
        return cached

    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_side_context_cache_misses')
    with (
        runtime_diagnostics.time_block('side_filter_precompute_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        obs = np.asarray(obs_indices, dtype=np.int64)
        finite_count = int(np.count_nonzero(labels.finite[obs]))
        nonzero_count = int(np.count_nonzero(labels.side[obs] != 0))
        obs_by_side: list[np.ndarray] = []
        key_by_side: list[tuple[int, ...]] = []
        for side in (-1, 0, 1):
            side_obs, side_key = _filtered_obs_key(
                obs_indices=obs,
                obs_key=obs_key,
                keep_mask=labels.side[obs] == int(side),
                runtime_diagnostics=runtime_diagnostics,
            )
            obs_by_side.append(side_obs)
            key_by_side.append(side_key)
            if runtime_diagnostics is not None:
                runtime_diagnostics.record_side_obs_count(int(side_obs.size))
        context = _SideObservationContext(
            obs_indices_by_side=tuple(obs_by_side),  # type: ignore[arg-type]
            obs_key_by_side=tuple(key_by_side),  # type: ignore[arg-type]
            finite_count=finite_count,
            nonzero_count=nonzero_count,
        )
    cache.signed_side_context[cache_key] = context
    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_side_contexts_built')
        runtime_diagnostics.inc(
            'n_side_gap_precomputed_fit_keys',
            len(set(context.obs_key_by_side)),
        )
    return context


def _obs_with_target_labeled_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...],
    labels: _SignedOffsetSideLabels,
    cache: _ObservationPlanCache,
    min_finite_count: int,
    side_context: _SideObservationContext | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int, bool]:
    trace = int(trace_idx)
    obs = np.asarray(obs_indices, dtype=np.int64)
    if side_context is not None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_side_context_lookup_calls')
        context = side_context
    else:
        context = _build_side_observation_context(
            labels=labels,
            obs_indices=obs,
            obs_key=obs_key,
            cache=cache,
            runtime_diagnostics=runtime_diagnostics,
        )
    finite_count = int(context.finite_count)
    nonzero_count = int(context.nonzero_count)
    if not _index_key_contains(obs_key=obs_key, trace_idx=trace, cache=cache):
        if bool(labels.finite[trace]):
            finite_count += 1
        if int(labels.side[trace]) != 0:
            nonzero_count += 1

    if finite_count < int(min_finite_count) or nonzero_count == 0:
        return obs, obs_key, 0, False

    target_side = int(labels.side[trace])
    with (
        runtime_diagnostics.time_block('side_filter_lookup_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        side_slot = _side_slot(target_side)
        side_obs = context.obs_indices_by_side[side_slot]
        side_obs_key = context.obs_key_by_side[side_slot]
    return side_obs, side_obs_key, target_side, True


def _select_group_ids(
    *,
    groups: tuple[SourceGroup, ...],
    target_group_id: int,
    cfg: PhysicsLiteConfig,
    use_neighbor_context: bool,
) -> np.ndarray:
    if not bool(use_neighbor_context) or not bool(cfg.neighbor_context.enabled):
        return np.asarray([int(target_group_id)], dtype=np.int64)
    return select_nearest_source_groups(
        groups,
        target_group_id=int(target_group_id),
        k_neighbors=int(cfg.neighbor_context.k_neighbors),
        max_source_distance_m=cfg.neighbor_context.max_source_distance_m,
        include_self=bool(cfg.neighbor_context.include_self),
    )


def _build_group_observation_contexts(
    *,
    groups: tuple[SourceGroup, ...],
    groups_by_id: Mapping[int, SourceGroup],
    valid_for_fit: np.ndarray,
    cfg: PhysicsLiteConfig,
    use_neighbor_context: bool,
    offset_signed_labels: _SignedOffsetSideLabels | None = None,
    plan_cache: _ObservationPlanCache | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> dict[int, _GroupObservationContext]:
    valid_mask = np.asarray(valid_for_fit, dtype=np.bool_)
    observation_cache = (
        plan_cache if plan_cache is not None else _ObservationPlanCache()
    )
    contexts: dict[int, _GroupObservationContext] = {}
    for group in groups:
        group_id = int(group.group_id)
        neighbor_group_ids = _select_group_ids(
            groups=groups,
            target_group_id=group_id,
            cfg=cfg,
            use_neighbor_context=use_neighbor_context,
        )
        neighbor_indices = _concat_group_traces(
            neighbor_group_ids,
            groups_by_id=groups_by_id,
        )
        valid_obs_indices = neighbor_indices[valid_mask[neighbor_indices]]
        valid_obs_key = _indices_key(valid_obs_indices)
        side_context = None
        if offset_signed_labels is not None and bool(
            cfg.physical_trend.segment_by_offset_sign
        ):
            side_context = _build_side_observation_context(
                labels=offset_signed_labels,
                obs_indices=valid_obs_indices,
                obs_key=valid_obs_key,
                cache=observation_cache,
                runtime_diagnostics=runtime_diagnostics,
            )
        contexts[group_id] = _GroupObservationContext(
            group_id=group_id,
            neighbor_group_ids=neighbor_group_ids,
            neighbor_indices=neighbor_indices,
            valid_obs_indices=valid_obs_indices,
            valid_obs_key=valid_obs_key,
            neighbor_count=int(neighbor_group_ids.size),
            prefilter_valid_count=int(valid_obs_indices.size),
            side_context=side_context,
        )
    return contexts


def _obs_with_target_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    geometry: CoarseGeometry,
    obs_key: tuple[int, ...] | None = None,
    cache: _ObservationPlanCache | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int, bool]:
    plan_cache = cache if cache is not None else _ObservationPlanCache()
    key = _obs_key_or_build(obs_indices, obs_key)
    cache_key = (key, int(trace_idx))
    cached = plan_cache.geometry_side.get(cache_key)
    if cached is not None:
        return cached

    obs = np.asarray(obs_indices, dtype=np.int64)
    context_indices = _append_index(obs_indices, trace_idx)
    signed = signed_offset_side_from_geometry(geometry, context_indices)
    if not bool(signed.reliable):
        result = (obs, key, 0, False)
        plan_cache.geometry_side[cache_key] = result
        return result

    pos = _trace_position_map(context_indices)
    target_side = int(signed.side[pos[int(trace_idx)]])
    obs_side = np.asarray(
        [int(signed.side[pos[int(obs_idx)]]) for obs_idx in obs.tolist()],
        dtype=np.int8,
    )
    side_obs, side_obs_key = _filtered_obs_key(
        obs_indices=obs,
        obs_key=key,
        keep_mask=obs_side == target_side,
    )
    result = (side_obs, side_obs_key, target_side, True)
    plan_cache.geometry_side[cache_key] = result
    return result


def _obs_with_target_signed_offset_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    signed_offset_m: np.ndarray,
    zero_tol_m: float = 1.0e-6,
    obs_key: tuple[int, ...] | None = None,
    cache: _ObservationPlanCache | None = None,
    labels: _SignedOffsetSideLabels | None = None,
    finite_mask: np.ndarray | None = None,
    min_finite_count: int = 2,
    side_context: _SideObservationContext | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int, bool]:
    plan_cache = cache if cache is not None else _ObservationPlanCache()
    if labels is not None:
        plan_cache.offset_signed_labels = labels
    elif plan_cache.offset_signed_labels is None:
        plan_cache.offset_signed_labels = _signed_offset_side_labels(
            signed_offset_m,
            zero_tol_m=zero_tol_m,
            finite_mask=finite_mask,
        )
    key = _obs_key_or_build(obs_indices, obs_key)
    return _obs_with_target_labeled_side(
        trace_idx=trace_idx,
        obs_indices=obs_indices,
        obs_key=key,
        labels=plan_cache.offset_signed_labels,
        cache=plan_cache,
        min_finite_count=int(min_finite_count),
        side_context=side_context,
        runtime_diagnostics=runtime_diagnostics,
    )


def _gap_context_key(
    *,
    obs_key: tuple[int, ...],
    cfg: PhysicsLiteConfig,
) -> tuple[tuple[int, ...], float, float | None]:
    min_gap = cfg.physical_trend.min_gap_m
    return (
        obs_key,
        float(cfg.physical_trend.gap_ratio),
        None if min_gap is None else float(min_gap),
    )


def _build_gap_segment_context(
    *,
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...],
    offset_abs_m: np.ndarray,
    cfg: PhysicsLiteConfig,
    cache: _ObservationPlanCache,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> _GapSegmentContext:
    cache_key = _gap_context_key(obs_key=obs_key, cfg=cfg)
    cached = cache.gap_context.get(cache_key)
    if cached is not None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_gap_context_cache_hits')
        return cached

    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_gap_context_cache_misses')
    with (
        runtime_diagnostics.time_block('gap_segment_precompute_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        obs = np.asarray(obs_indices, dtype=np.int64)
        segment_id = split_offset_gap_segments(
            np.asarray(offset_abs_m, dtype=np.float32)[obs],
            split_by_offset_gap=bool(cfg.physical_trend.split_by_offset_gap),
            gap_ratio=float(cfg.physical_trend.gap_ratio),
            min_gap_m=cfg.physical_trend.min_gap_m,
        )
        segment_id_by_trace_idx = {
            int(trace_idx): int(seg_id)
            for trace_idx, seg_id in zip(
                obs.tolist(),
                segment_id.tolist(),
                strict=True,
            )
        }
        obs_indices_by_segment_id: dict[int, np.ndarray] = {}
        obs_key_by_segment_id: dict[int, tuple[int, ...]] = {}
        for seg_id in np.unique(segment_id).astype(np.int64).tolist():
            segment_obs, segment_key = _filtered_obs_key(
                obs_indices=obs,
                obs_key=obs_key,
                keep_mask=segment_id == int(seg_id),
                runtime_diagnostics=runtime_diagnostics,
            )
            obs_indices_by_segment_id[int(seg_id)] = segment_obs
            obs_key_by_segment_id[int(seg_id)] = segment_key
            if runtime_diagnostics is not None:
                runtime_diagnostics.record_gap_segment_obs_count(
                    int(segment_obs.size)
                )
        context = _GapSegmentContext(
            segment_id_by_trace_idx=segment_id_by_trace_idx,
            obs_indices_by_segment_id=obs_indices_by_segment_id,
            obs_key_by_segment_id=obs_key_by_segment_id,
        )
    cache.gap_context[cache_key] = context
    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_gap_contexts_built')
        runtime_diagnostics.inc(
            'n_side_gap_precomputed_fit_keys',
            len(context.obs_key_by_segment_id),
        )
    return context


def _obs_with_target_gap_segment(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    cfg: PhysicsLiteConfig,
    obs_key: tuple[int, ...] | None = None,
    cache: _ObservationPlanCache | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    plan_cache = cache if cache is not None else _ObservationPlanCache()
    key = _obs_key_or_build(obs_indices, obs_key)
    min_gap = cfg.physical_trend.min_gap_m
    fallback_gap_key = (
        key,
        int(trace_idx),
        float(cfg.physical_trend.gap_ratio),
        None if min_gap is None else float(min_gap),
    )
    obs = np.asarray(obs_indices, dtype=np.int64)
    if _index_key_contains(obs_key=key, trace_idx=int(trace_idx), cache=plan_cache):
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_gap_trace_in_obs')
            runtime_diagnostics.inc('n_gap_fast_path_calls')
        context = _build_gap_segment_context(
            obs_indices=obs,
            obs_key=key,
            offset_abs_m=offset_abs_m,
            cfg=cfg,
            cache=plan_cache,
            runtime_diagnostics=runtime_diagnostics,
        )
        with (
            runtime_diagnostics.time_block('gap_segment_lookup_sec')
            if runtime_diagnostics is not None
            else nullcontext()
        ):
            target_segment_id = int(
                context.segment_id_by_trace_idx[int(trace_idx)]
            )
            return (
                context.obs_indices_by_segment_id[target_segment_id],
                context.obs_key_by_segment_id[target_segment_id],
                target_segment_id,
            )

    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_gap_trace_not_in_obs')
        runtime_diagnostics.inc('n_gap_fallback_calls')
    cached = plan_cache.gap_segment.get(fallback_gap_key)
    if cached is not None:
        return cached

    with (
        runtime_diagnostics.time_block('gap_segment_fallback_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        context_indices = _append_index(obs_indices, trace_idx)
        segment_id = split_offset_gap_segments(
            np.asarray(offset_abs_m, dtype=np.float32)[context_indices],
            split_by_offset_gap=bool(cfg.physical_trend.split_by_offset_gap),
            gap_ratio=float(cfg.physical_trend.gap_ratio),
            min_gap_m=cfg.physical_trend.min_gap_m,
        )
        pos = _trace_position_map(context_indices)
        target_segment_id = int(segment_id[pos[int(trace_idx)]])
        obs_segment_id = np.asarray(
            [int(segment_id[pos[int(obs_idx)]]) for obs_idx in obs.tolist()],
            dtype=np.int64,
        )
        segment_obs, segment_obs_key = _filtered_obs_key(
            obs_indices=obs,
            obs_key=key,
            keep_mask=obs_segment_id == target_segment_id,
            runtime_diagnostics=runtime_diagnostics,
        )
        result = (segment_obs, segment_obs_key, target_segment_id)
    plan_cache.gap_segment[fallback_gap_key] = result
    return result


def _build_observation_plan(
    *,
    trace_idx: int,
    target_group_id: int,
    group_context_by_id: Mapping[int, _GroupObservationContext],
    geometry: CoarseGeometry | None,
    offset_abs_m: np.ndarray,
    offset_signed_m: np.ndarray | None,
    cfg: PhysicsLiteConfig,
    min_fit_obs: int,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    plan_cache: _ObservationPlanCache | None = None,
) -> _ObservationPlan | None:
    observation_cache = (
        plan_cache if plan_cache is not None else _ObservationPlanCache()
    )
    with (
        runtime_diagnostics.time_block('neighbor_plan_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        group_context = group_context_by_id.get(int(target_group_id))
    if group_context is None:
        msg = f'observation context not found for group_id={int(target_group_id)}'
        raise ValueError(msg)

    valid_obs = group_context.valid_obs_indices
    valid_obs_key = group_context.valid_obs_key
    neighbor_count = int(group_context.neighbor_count)
    prefilter_valid_count = int(group_context.prefilter_valid_count)
    min_fit_obs = int(min_fit_obs)

    if prefilter_valid_count < min_fit_obs:
        return _ObservationPlan(
            obs_indices=valid_obs,
            obs_key=valid_obs_key,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=-1,
            side=0,
            relaxed=False,
        )

    side_obs = valid_obs
    side_obs_key = valid_obs_key
    side = 0
    side_reliable = False
    with (
        runtime_diagnostics.time_block('side_segment_build_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        if bool(cfg.physical_trend.segment_by_offset_sign):
            if offset_signed_m is not None:
                (
                    side_obs,
                    side_obs_key,
                    side,
                    side_reliable,
                ) = _obs_with_target_signed_offset_side(
                    trace_idx=trace_idx,
                    obs_indices=valid_obs,
                    signed_offset_m=offset_signed_m,
                    obs_key=valid_obs_key,
                    cache=observation_cache,
                    labels=observation_cache.offset_signed_labels,
                    min_finite_count=(
                        1
                        if (
                            geometry is not None
                            and bool(cfg.physical_trend.use_geometry_offset)
                            and geometry.offset_signed_geom_m is not None
                        )
                        else 2
                    ),
                    side_context=group_context.side_context,
                    runtime_diagnostics=runtime_diagnostics,
                )
            elif geometry is not None:
                (
                    side_obs,
                    side_obs_key,
                    side,
                    side_reliable,
                ) = _obs_with_target_side(
                    trace_idx=trace_idx,
                    obs_indices=valid_obs,
                    geometry=geometry,
                    obs_key=valid_obs_key,
                    cache=observation_cache,
                )

        segment_obs = side_obs
        segment_obs_key = side_obs_key
        segment_id = 0
        if bool(cfg.physical_trend.split_by_offset_gap):
            segment_obs, segment_obs_key, segment_id = _obs_with_target_gap_segment(
                trace_idx=trace_idx,
                obs_indices=side_obs,
                obs_key=side_obs_key,
                offset_abs_m=offset_abs_m,
                cfg=cfg,
                cache=observation_cache,
                runtime_diagnostics=runtime_diagnostics,
            )

    if int(segment_obs.size) >= min_fit_obs:
        return _ObservationPlan(
            obs_indices=segment_obs,
            obs_key=segment_obs_key,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=segment_id,
            side=side,
            relaxed=False,
        )

    if (
        bool(cfg.physical_trend.split_by_offset_gap)
        and int(side_obs.size) >= min_fit_obs
    ):
        return _ObservationPlan(
            obs_indices=side_obs,
            obs_key=side_obs_key,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=side,
            relaxed=True,
        )

    if side_reliable and int(valid_obs.size) >= min_fit_obs:
        return _ObservationPlan(
            obs_indices=valid_obs,
            obs_key=valid_obs_key,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=0,
            relaxed=True,
        )

    return _ObservationPlan(
        obs_indices=segment_obs,
        obs_key=segment_obs_key,
        neighbor_count=neighbor_count,
        prefilter_valid_count=prefilter_valid_count,
        segment_id=segment_id,
        side=side,
        relaxed=False,
    )
