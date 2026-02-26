from __future__ import annotations

import argparse

from stage2.cfg import DEFAULT_STAGE2_CFG
from stage2.runner import run_stage2


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description='Run stage2 make_psn512 windows.')


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    parser.parse_args(argv)
    run_stage2(cfg=DEFAULT_STAGE2_CFG)


__all__ = ['main']


if __name__ == '__main__':
    main()
