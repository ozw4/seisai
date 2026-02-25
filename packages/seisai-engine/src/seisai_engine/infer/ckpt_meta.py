from __future__ import annotations

from typing import Any

__all__ = ['resolve_output_ids', 'resolve_softmax_axis']


def resolve_softmax_axis(
    *,
    ckpt: dict[str, Any],
    out_chans: int,
    pipeline_name: str,
) -> str:
    axis_raw = ckpt.get('softmax_axis')
    if axis_raw is not None:
        if not isinstance(axis_raw, str):
            msg = 'checkpoint softmax_axis must be str'
            raise TypeError(msg)
        axis = axis_raw.strip().lower()
        if axis not in ('time', 'channel'):
            msg = 'checkpoint softmax_axis must be "time" or "channel"'
            raise ValueError(msg)
    elif ckpt['pipeline'] == pipeline_name:
        axis = 'channel'
    elif int(out_chans) == 1:
        axis = 'time'
    else:
        msg = 'softmax_axis is ambiguous; set checkpoint softmax_axis'
        raise ValueError(msg)

    if axis == 'time' and int(out_chans) != 1:
        msg = 'softmax_axis="time" requires out_chans==1'
        raise ValueError(msg)
    return axis


def resolve_output_ids(
    *,
    ckpt: dict[str, Any],
    out_chans: int,
    pipeline_name: str,
) -> tuple[str, ...]:
    output_ids_raw = ckpt.get('output_ids')
    if output_ids_raw is not None:
        if not isinstance(output_ids_raw, list | tuple):
            msg = 'checkpoint output_ids must be list[str] or tuple[str, ...]'
            raise TypeError(msg)
        output_ids: list[str] = []
        for idx, item in enumerate(output_ids_raw):
            if not isinstance(item, str) or len(item.strip()) == 0:
                msg = f'checkpoint output_ids[{idx}] must be non-empty str'
                raise TypeError(msg)
            output_ids.append(item.strip())
        if len(output_ids) != int(out_chans):
            msg = (
                f'checkpoint output_ids length {len(output_ids)} '
                f'!= out_chans {int(out_chans)}'
            )
            raise ValueError(msg)
        if len(set(output_ids)) != len(output_ids):
            msg = 'checkpoint output_ids must be unique'
            raise ValueError(msg)
        return tuple(output_ids)

    if ckpt['pipeline'] == pipeline_name and int(out_chans) == 3:
        return ('P', 'S', 'N')
    if int(out_chans) == 1:
        return ('P',)
    return tuple(f'ch{idx}' for idx in range(int(out_chans)))
