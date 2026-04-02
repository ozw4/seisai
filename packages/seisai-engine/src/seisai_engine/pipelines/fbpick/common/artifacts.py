"""Canonical artifact contracts for fbpick stage outputs.

`common/io.py` is the only loader/saver that should enforce these structural
contracts. Stage consumers may add only stage-specific semantic checks on top.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    'ARTIFACT_VERSION',
    'ARTIFACT_META_REQUIRED_KEYS',
    'ARTIFACT_SPECS',
    'COARSE_ARTIFACT_SPEC',
    'FINE_ARTIFACT_SPEC',
    'GLOBAL_QC_ARTIFACT_SPEC',
    'QC_STATUS_ADJUST',
    'QC_STATUS_KEEP',
    'QC_STATUS_REJECT',
    'ArtifactFieldSpec',
    'ArtifactMeta',
    'ArtifactSpec',
    'get_artifact_spec',
]

ARTIFACT_VERSION = 1
ARTIFACT_META_REQUIRED_KEYS = (
    'artifact_version',
    'stage',
    'survey_id',
    'npz_filename',
    'source_refs',
    'dimensions',
)

QC_STATUS_KEEP = 0
QC_STATUS_ADJUST = 1
QC_STATUS_REJECT = 2


@dataclass(frozen=True)
class ArtifactFieldSpec:
    key: str
    dtype_name: str
    shape: Sequence[str]
    description: str
    required: bool = True


@dataclass(frozen=True)
class ArtifactMeta:
    artifact_version: int
    stage: str
    survey_id: str
    npz_filename: str
    source_refs: dict[str, str]
    dimensions: dict[str, int]


@dataclass(frozen=True)
class ArtifactSpec:
    stage: str
    npz_filename: str
    meta_filename: str
    fields: Sequence[ArtifactFieldSpec]

    @property
    def required_field_keys(self) -> tuple[str, ...]:
        return tuple(field.key for field in self.fields if field.required)


COARSE_ARTIFACT_SPEC = ArtifactSpec(
    stage='coarse',
    npz_filename='coarse_artifact.npz',
    meta_filename='coarse_meta.json',
    fields=(
        ArtifactFieldSpec(
            key='prob',
            dtype_name='float32',
            shape=('n_traces', 'n_samples'),
            description='Per-trace coarse first-break probability on the raw time axis.',
        ),
        ArtifactFieldSpec(
            key='pick_idx',
            dtype_name='int32',
            shape=('n_traces',),
            description='Argmax coarse pick index on the raw time axis.',
        ),
        ArtifactFieldSpec(
            key='confidence',
            dtype_name='float32',
            shape=('n_traces',),
            description='Per-trace confidence for the coarse pick.',
        ),
        ArtifactFieldSpec(
            key='trace_valid',
            dtype_name='bool',
            shape=('n_traces',),
            description='Per-trace validity mask after coarse screening.',
        ),
        ArtifactFieldSpec(
            key='raw_trace_idx',
            dtype_name='int64',
            shape=('n_traces',),
            description='Raw trace index on the survey trace axis.',
        ),
        ArtifactFieldSpec(
            key='offsets',
            dtype_name='float32',
            shape=('n_traces',),
            description='Trace offsets aligned with raw_trace_idx.',
        ),
        ArtifactFieldSpec(
            key='time_axis',
            dtype_name='float32',
            shape=('n_samples',),
            description='Shared raw-axis time values for prob columns.',
        ),
    ),
)

FINE_ARTIFACT_SPEC = ArtifactSpec(
    stage='fine',
    npz_filename='fine_artifact.npz',
    meta_filename='fine_meta.json',
    fields=(
        ArtifactFieldSpec(
            key='local_prob',
            dtype_name='float32',
            shape=('n_traces', 'local_window_len'),
            description='Fixed-length local-window fine probability per trace.',
        ),
        ArtifactFieldSpec(
            key='local_pick_idx',
            dtype_name='int32',
            shape=('n_traces',),
            description='Fine pick index on the local-window axis.',
        ),
        ArtifactFieldSpec(
            key='raw_pick_idx',
            dtype_name='int32',
            shape=('n_traces',),
            description='Fine pick index mapped back to the raw time axis.',
        ),
        ArtifactFieldSpec(
            key='local_window_start_idx',
            dtype_name='int64',
            shape=('n_traces',),
            description='Inclusive raw-axis start index for each local window.',
        ),
        ArtifactFieldSpec(
            key='local_window_end_idx',
            dtype_name='int64',
            shape=('n_traces',),
            description='Exclusive raw-axis end index for each local window.',
        ),
        ArtifactFieldSpec(
            key='raw_trace_idx',
            dtype_name='int64',
            shape=('n_traces',),
            description='Raw trace index on the survey trace axis.',
        ),
        ArtifactFieldSpec(
            key='confidence',
            dtype_name='float32',
            shape=('n_traces',),
            description='Per-trace confidence for the fine pick.',
        ),
    ),
)

GLOBAL_QC_ARTIFACT_SPEC = ArtifactSpec(
    stage='global_qc',
    npz_filename='global_qc_artifact.npz',
    meta_filename='global_qc_meta.json',
    fields=(
        ArtifactFieldSpec(
            key='pick_global',
            dtype_name='int32',
            shape=('n_traces',),
            description='Final pick index after global QC on the raw time axis.',
        ),
        ArtifactFieldSpec(
            key='confidence_global',
            dtype_name='float32',
            shape=('n_traces',),
            description='Per-trace confidence after global QC.',
        ),
        ArtifactFieldSpec(
            key='reject_flag',
            dtype_name='bool',
            shape=('n_traces',),
            description='Per-trace rejection flag after global QC.',
        ),
        ArtifactFieldSpec(
            key='qc_status',
            dtype_name='int8',
            shape=('n_traces',),
            description='Per-trace QC status code: 0=keep, 1=adjust, 2=reject.',
        ),
        ArtifactFieldSpec(
            key='raw_trace_idx',
            dtype_name='int64',
            shape=('n_traces',),
            description='Raw trace index on the survey trace axis.',
        ),
    ),
)

ARTIFACT_SPECS = {
    COARSE_ARTIFACT_SPEC.stage: COARSE_ARTIFACT_SPEC,
    FINE_ARTIFACT_SPEC.stage: FINE_ARTIFACT_SPEC,
    GLOBAL_QC_ARTIFACT_SPEC.stage: GLOBAL_QC_ARTIFACT_SPEC,
}


def get_artifact_spec(stage: str) -> ArtifactSpec:
    if not isinstance(stage, str):
        msg = 'stage must be str'
        raise TypeError(msg)
    try:
        return ARTIFACT_SPECS[stage]
    except KeyError as exc:
        supported = ', '.join(sorted(ARTIFACT_SPECS))
        msg = f'unsupported fbpick stage "{stage}"; supported: {supported}'
        raise ValueError(msg) from exc
