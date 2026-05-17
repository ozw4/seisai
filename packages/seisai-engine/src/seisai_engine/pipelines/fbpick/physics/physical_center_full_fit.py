"""Full fit-policy runner for physical-center fitting."""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np

from .physical_center_context import (
    PhysicalCenterBuildContext,
    PhysicalCenterInputs,
    PhysicalCenterWorkspace,
)
from .physical_center_context_fit import (
    _build_fit_context_work_items,
    _fit_and_assign_context_work_items,
    _FitModelForPlan,
    _prepare_fit_context_assignments_for_trace_indices,
)
from .physical_center_fallback import _finalize_result_with_pending_trend_fallback
from .physical_center_fit import _fit_model_for_plan
from .physical_center_types import (
    PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT,
    PhysicalCenterResult,
)
from .progress import NullProgressReporter


def run_full_fit_policy(
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    strategy: object,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> PhysicalCenterResult:
    """Run the fit_policy='full' physical-center orchestration."""

    table = inputs.table
    reporter = progress if progress is not None else NullProgressReporter()
    context = dict(progress_context or {})
    n = int(table.n_traces)

    reporter.emit(
        'physical-center.stage_start',
        **context,
        stage='fit_context_preparation',
    )
    stage_start = time.perf_counter()
    fit_context_assignments, _fit_context_ordered_assignments = (
        _prepare_fit_context_assignments_for_trace_indices(
            np.arange(n, dtype=np.int64),
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
        )
    )
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='fit_context_preparation',
        elapsed=time.perf_counter() - stage_start,
        unique_keys=len(fit_context_assignments),
    )

    work_items = _build_fit_context_work_items(
        fit_context_assignments,
        offset_abs_m=build_context.offset_abs_m,
        pick_t_sec=build_context.pick_t_sec,
        coarse_pmax=table.coarse_pmax,
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT,
    )
    reporter.emit(
        'physical-center.contexts_built',
        **context,
        work_items=len(work_items),
        unique_keys=len(fit_context_assignments),
    )
    _fit_and_assign_context_work_items(
        work_items,
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
        strategy=strategy,
        progress=reporter,
        progress_context=context,
        fit_model_for_plan=(
            _fit_model_for_plan
            if fit_model_for_plan is None
            else fit_model_for_plan
        ),
    )

    if workspace.runtime_diagnostics is not None:
        workspace.runtime_diagnostics.set_unique_fit_contexts(len(workspace.fit_cache))

    result = _finalize_result_with_pending_trend_fallback(
        workspace.arrays,
        pending_trend_fallback=workspace.pending_trend_fallback,
        table=table,
        feasible=inputs.feasible,
        trend=inputs.trend,
        merged=inputs.merged,
        trend_provider=inputs.trend_provider,
    )
    reporter.emit(
        'physical-center.done',
        **context,
        status='ok',
        n_traces=n,
        n_source_groups=len(build_context.groups),
        n_unique_fit_contexts=len(workspace.fit_cache),
    )
    return result
