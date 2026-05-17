"""Anchor-source-xy policy runner for physical-center fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .physical_center_anchor import build_anchor_source_xy_context
from .physical_center_anchor_reuse import (
    _assign_anchor_reuse_predictions,
    _build_anchor_reuse_group_plan,
    _compatible_anchor_contexts_for_reuse,
    _nearest_anchor_reuse_context,
    _run_no_compatible_full_fit_fallback,
)
from .physical_center_fallback import _finalize_result_with_pending_trend_fallback
from .physical_center_fit import _fit_model_for_plan

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .physical_center_context import (
        PhysicalCenterBuildContext,
        PhysicalCenterInputs,
        PhysicalCenterWorkspace,
    )
    from .physical_center_context_fit import _FitModelForPlan
    from .physical_center_types import PhysicalCenterResult
    from .runtime_policy import SourceXYAnchorSelectionResult


def run_anchor_source_xy_policy(  # noqa: PLR0913
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    anchor_selection: SourceXYAnchorSelectionResult | None,
    strategy: object,
    progress: object,
    progress_context: Mapping[str, object] | None = None,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> PhysicalCenterResult:
    """Run the fit_policy='anchor_source_xy' physical-center orchestration."""
    reporter = progress
    context = dict(progress_context or {})
    groups = build_context.groups
    arrays = workspace.arrays
    fit_cache = workspace.fit_cache
    runtime_diagnostics = workspace.runtime_diagnostics
    pending_trend_fallback = workspace.pending_trend_fallback
    fit_model = (
        _fit_model_for_plan if fit_model_for_plan is None else fit_model_for_plan
    )

    anchor_source_xy = build_anchor_source_xy_context(
        anchor_selection=anchor_selection,
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
        strategy=strategy,
        progress=reporter,
        progress_context=context,
        fit_model_for_plan=fit_model,
    )
    n_reused_predictions = 0
    fallback_full_group_ids: set[int] = set()

    for group in groups:
        group_id = int(group.group_id)
        if bool(anchor_source_xy.is_anchor_by_id.get(group_id, False)):
            continue
        group_trace_indices = np.asarray(group.trace_indices, dtype=np.int64)
        nearest_context = _nearest_anchor_reuse_context(
            group_id=group_id,
            anchor_source_xy=anchor_source_xy,
            inputs=inputs,
            runtime_diagnostics=runtime_diagnostics,
        )
        compatible_anchor_context_by_plan_key = _compatible_anchor_contexts_for_reuse(
            anchor_source_xy=anchor_source_xy,
            nearest_context=nearest_context,
            inputs=inputs,
            runtime_diagnostics=runtime_diagnostics,
        )
        reuse_group_plan = _build_anchor_reuse_group_plan(
            group=group,
            nearest_context=nearest_context,
            compatible_anchor_context_by_plan_key=compatible_anchor_context_by_plan_key,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
        )
        reuse_items = reuse_group_plan.reuse_items

        if reuse_group_plan.uses_fallback_full_fit:
            fallback_full_group_ids.add(group_id)
            _run_no_compatible_full_fit_fallback(
                reuse_group_plan.fallback_full_trace_indices,
                inputs=inputs,
                build_context=build_context,
                workspace=workspace,
                strategy=strategy,
                progress=reporter,
                progress_context=context,
                fit_model_for_plan=fit_model,
            )

        n_reused_predictions += _assign_anchor_reuse_predictions(
            reuse_items,
            group_trace_indices=group_trace_indices,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
            strategy=strategy,
            progress=reporter,
            progress_context=context,
            fit_model_for_plan=fit_model,
        )

    if runtime_diagnostics is not None:
        runtime_diagnostics.record_reused_predictions(n_reused_predictions)
        runtime_diagnostics.record_fallback_full_fit_no_compatible_anchor(
            len(fallback_full_group_ids)
        )
        runtime_diagnostics.set_anchor_reuse_groups(
            n_non_anchor_groups=sum(
                1
                for group in groups
                if not bool(
                    anchor_source_xy.is_anchor_by_id.get(
                        int(group.group_id),
                        False,
                    )
                )
            )
        )
        runtime_diagnostics.set_fit_call_reduction_rate_vs_full(
            full_fit_call_count_estimate=len(groups)
        )
        runtime_diagnostics.set_unique_fit_contexts(len(fit_cache))
    result = _finalize_result_with_pending_trend_fallback(
        arrays,
        pending_trend_fallback=pending_trend_fallback,
        table=inputs.table,
        feasible=inputs.feasible,
        trend=inputs.trend,
        merged=inputs.merged,
        trend_provider=inputs.trend_provider,
    )
    reporter.emit(
        'physical-center.done',
        **context,
        status='ok',
        n_traces=int(inputs.table.n_traces),
        n_source_groups=len(groups),
        n_unique_fit_contexts=len(fit_cache),
    )
    return result
