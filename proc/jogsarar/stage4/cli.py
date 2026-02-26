from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from stage4.cfg import DEFAULT_STAGE4_CFG, _build_stage4_parser, _parse_segy_exts, load_stage4_cfg_yaml
from stage4.runner import run_stage4


def main(argv: list[str] | None = None) -> None:
    parser = _build_stage4_parser()
    args = parser.parse_args(argv)

    if args.config is None:
        cfg = DEFAULT_STAGE4_CFG
    else:
        cfg = load_stage4_cfg_yaml(args.config)

    if args.in_raw_segy_root is not None:
        cfg = replace(
            cfg, in_raw_segy_root=Path(args.in_raw_segy_root).expanduser().resolve()
        )
    if args.in_win512_segy_root is not None:
        cfg = replace(
            cfg,
            in_win512_segy_root=Path(args.in_win512_segy_root).expanduser().resolve(),
        )
    if args.out_pred_root is not None:
        cfg = replace(
            cfg, out_pred_root=Path(args.out_pred_root).expanduser().resolve()
        )
    if args.cfg_yaml is not None:
        cfg = replace(cfg, cfg_yaml=Path(args.cfg_yaml).expanduser().resolve())
    if args.ckpt_path is not None:
        cfg = replace(cfg, ckpt_path=Path(args.ckpt_path).expanduser().resolve())
    if args.device is not None:
        cfg = replace(cfg, device=str(args.device))
    if args.segy_exts is not None:
        cfg = replace(cfg, segy_exts=_parse_segy_exts(args.segy_exts))
    if args.post_trough_peak_search is not None:
        cfg = replace(cfg, post_trough_peak_search=str(args.post_trough_peak_search))
    if args.viz_every_n_shots is not None:
        cfg = replace(cfg, viz_every_n_shots=int(args.viz_every_n_shots))

    run_stage4(cfg=cfg)


__all__ = ['main']


if __name__ == '__main__':
    main()
