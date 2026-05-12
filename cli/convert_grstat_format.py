"""Convert grstat first-break CRD files between supported text formats.

This CLI is primarily intended to convert the historical ``* rec.no.=`` layout
into the newer ``fb: recno start_ch end_ch values...`` layout while preserving
record numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seisai_pick.pickio.io_grstat import load_grstat_matrix, numpy2fbcrd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Convert grstat CRD files while preserving record numbers. Both input '
            'formats supported by seisai_pick.pickio.io_grstat are accepted.'
        )
    )
    parser.add_argument(
        '--input-crd',
        required=True,
        help='Input grstat CRD file. Legacy and recno-channel-range formats are supported.',
    )
    parser.add_argument(
        '--output-crd',
        required=True,
        help='Output grstat CRD file.',
    )
    parser.add_argument(
        '--dt-ms',
        type=float,
        default=2.0,
        help=(
            'Sampling interval multiplier used by the CRD values. Use 2.0 when '
            'CRD values are in ms and data sampling is 2 ms. Use 1.0 when values '
            'are already sample indices. Default: 2.0.'
        ),
    )
    parser.add_argument(
        '--output-format',
        choices=('recno_channel_range', 'legacy'),
        default='recno_channel_range',
        help='Output format. Default: recno_channel_range.',
    )
    parser.add_argument(
        '--values-per-line',
        type=int,
        default=None,
        help='Number of channel values per fb line. Defaults to format-specific value.',
    )
    parser.add_argument(
        '--header-comment',
        default=None,
        help='Optional header comment. Defaults to "converted from <input>".',
    )
    parser.add_argument(
        '--no-strict-blocks',
        action='store_true',
        help='Disable strict contiguous/duplicate block validation while reading.',
    )
    parser.add_argument(
        '--allow-variable-channel-count',
        action='store_true',
        help=(
            'Allow records with different max channel counts or missing channels. '
            'Missing cells are written as no-pick values in the output matrix.'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    input_path = Path(args.input_crd).expanduser().resolve()
    output_path = Path(args.output_crd).expanduser().resolve()
    if args.dt_ms <= 0:
        raise ValueError('--dt-ms must be > 0')
    if args.values_per_line is not None and args.values_per_line <= 0:
        raise ValueError('--values-per-line must be > 0 when provided')

    parsed = load_grstat_matrix(
        input_path,
        dt_multiplier=float(args.dt_ms),
        strict_blocks=not args.no_strict_blocks,
        strict_channel_count=not args.allow_variable_channel_count,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_comment = args.header_comment or f'converted from {input_path.name}'
    written = numpy2fbcrd(
        dt=float(args.dt_ms),
        fbnum=parsed.samples,
        gather_range=parsed.record_numbers.tolist(),
        output_name=str(output_path),
        output_format=args.output_format,
        values_per_line=args.values_per_line,
        header_comment=header_comment,
    )

    print('converted:')
    print(f'  input : {input_path}')
    print(f'  output: {output_path}')
    print(f'  output_format: {args.output_format}')
    print(f'  records: {len(parsed.record_numbers)}')
    print(f'  channels: {parsed.samples.shape[1]}')
    print(f'  dt_ms: {float(args.dt_ms)}')
    print(f'  strict_blocks: {not args.no_strict_blocks}')
    print(f'  strict_channel_count: {not args.allow_variable_channel_count}')
    print(f'  written_shape: {tuple(written.shape)}')


if __name__ == '__main__':
    main()
