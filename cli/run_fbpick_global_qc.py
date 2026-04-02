"""Thin entrypoint for fbpick global QC."""

from __future__ import annotations

from seisai_engine.pipelines.fbpick.global_qc.run import main as pipeline_main

__all__ = ['main']


def main(argv: list[str] | None = None) -> None:
    pipeline_main(argv=argv)


if __name__ == '__main__':
    main()
