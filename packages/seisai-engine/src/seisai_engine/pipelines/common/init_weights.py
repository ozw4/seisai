from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint_io import load_checkpoint

__all__ = ['maybe_load_init_weights']


def _extract_chans(*, sig: dict, label: str, init_ckpt_path: Path) -> tuple[int, int]:
    in_chans = sig.get('in_chans')
    out_chans = sig.get('out_chans')

    in_ok = isinstance(in_chans, int) and not isinstance(in_chans, bool)
    out_ok = isinstance(out_chans, int) and not isinstance(out_chans, bool)
    if not in_ok or not out_ok:
        msg = (
            'cannot validate init checkpoint compatibility '
            '(このモデルでは互換性検証できない): '
            f'{label}.model_sig must contain int in_chans/out_chans; '
            f'got in_chans={in_chans!r}, out_chans={out_chans!r}; '
            f'init_ckpt={init_ckpt_path}'
        )
        raise RuntimeError(msg)

    return int(in_chans), int(out_chans)


def _resolve_init_ckpt_path(*, cfg: dict, base_dir: Path) -> Path | None:
    train_cfg = cfg.get('train')
    if train_cfg is None:
        return None
    if not isinstance(train_cfg, dict):
        msg = 'train must be dict'
        raise TypeError(msg)

    init_ckpt = train_cfg.get('init_ckpt')
    if init_ckpt is None:
        return None
    if not isinstance(init_ckpt, str):
        msg = 'train.init_ckpt must be str or null'
        raise TypeError(msg)

    init_ckpt_str = init_ckpt.strip()
    if not init_ckpt_str:
        return None

    init_ckpt_path = Path(init_ckpt_str)
    if not init_ckpt_path.is_absolute():
        init_ckpt_path = base_dir / init_ckpt_path
    init_ckpt_path = init_ckpt_path.resolve()
    if not init_ckpt_path.is_file():
        raise FileNotFoundError(init_ckpt_path)
    return init_ckpt_path


def _format_key_block(*, keys: list[str]) -> str:
    return '\n'.join(keys)


def maybe_load_init_weights(
    *,
    cfg: dict,
    base_dir: Path,
    model: torch.nn.Module,
    model_sig: dict,
) -> None:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)
    if not isinstance(base_dir, Path):
        msg = 'base_dir must be Path'
        raise TypeError(msg)

    init_ckpt_path = _resolve_init_ckpt_path(cfg=cfg, base_dir=base_dir)
    if init_ckpt_path is None:
        return

    if not isinstance(model_sig, dict):
        msg = 'model_sig must be dict'
        raise TypeError(msg)

    ckpt = load_checkpoint(init_ckpt_path)
    ckpt_model_sig = ckpt['model_sig']
    ckpt_in, ckpt_out = _extract_chans(
        sig=ckpt_model_sig,
        label='checkpoint',
        init_ckpt_path=init_ckpt_path,
    )
    cur_in, cur_out = _extract_chans(
        sig=model_sig,
        label='current',
        init_ckpt_path=init_ckpt_path,
    )

    if ckpt_in != cur_in:
        msg = (
            'init_ckpt in_chans mismatch: '
            f'ckpt_in={ckpt_in}, cur_in={cur_in}, init_ckpt={init_ckpt_path}'
        )
        raise RuntimeError(msg)

    ckpt_state_dict = ckpt['model_state_dict']
    if not isinstance(ckpt_state_dict, dict):
        msg = f'checkpoint model_state_dict must be dict: init_ckpt={init_ckpt_path}'
        raise RuntimeError(msg)

    out_chans_match = ckpt_out == cur_out
    removed_seg_head_count = 0

    if out_chans_match:
        load_state_dict = dict(ckpt_state_dict)
    else:
        if not isinstance(getattr(model, 'seg_head', None), torch.nn.Module):
            msg = (
                'init_ckpt out_chans mismatch but model has no seg_head: '
                f'ckpt_out={ckpt_out}, cur_out={cur_out}, init_ckpt={init_ckpt_path}'
            )
            raise RuntimeError(msg)
        load_state_dict = {}
        for key, value in ckpt_state_dict.items():
            if key.startswith('seg_head.'):
                removed_seg_head_count += 1
                continue
            load_state_dict[key] = value

    load_result = model.load_state_dict(load_state_dict, strict=False)
    unexpected_keys = list(load_result.unexpected_keys)
    missing_keys = list(load_result.missing_keys)

    if unexpected_keys:
        msg = (
            'init_ckpt load produced unexpected keys:\n'
            f'{_format_key_block(keys=unexpected_keys)}\n'
            f'init_ckpt={init_ckpt_path}'
        )
        raise RuntimeError(msg)

    if out_chans_match:
        if missing_keys:
            msg = (
                'init_ckpt load produced missing keys even though out_chans match:\n'
                f'{_format_key_block(keys=missing_keys)}\n'
                f'init_ckpt={init_ckpt_path}'
            )
            raise RuntimeError(msg)
    else:
        invalid_missing_keys = [
            key for key in missing_keys if not key.startswith('seg_head.')
        ]
        if invalid_missing_keys:
            msg = (
                'init_ckpt load produced non-seg_head missing keys with out_chans mismatch:\n'
                f'{_format_key_block(keys=missing_keys)}\n'
                f'init_ckpt={init_ckpt_path}'
            )
            raise RuntimeError(msg)

    print(
        '[init_ckpt] loaded '
        f'path={init_ckpt_path} '
        f'pipeline={ckpt["pipeline"]} '
        f'ckpt_in={ckpt_in} ckpt_out={ckpt_out} '
        f'cur_in={cur_in} cur_out={cur_out}'
    )
    if not out_chans_match:
        print(
            '[init_ckpt] out_chans mismatch: '
            f'excluded seg_head keys={removed_seg_head_count}'
        )
