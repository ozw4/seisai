"""Thin entrypoint for PSN SEG-Y inference."""

from __future__ import annotations

from seisai_engine.pipelines.psn.infer_segy2segy import main as pipeline_main

__all__ = ['main']


def main(argv: list[str] | None = None) -> None:
    pipeline_main(argv=argv)


if __name__ == '__main__':
    main()
