"""Thin entrypoint for fbpick coarse inference."""

from __future__ import annotations

from seisai_engine.pipelines.fbpick.coarse.infer_segy2npz import main as pipeline_main

__all__ = ['main']


def main(argv: list[str] | None = None) -> None:
    pipeline_main(argv=argv)


if __name__ == '__main__':
    main()
