"""Standalone matrix-free Betti curve command-line program."""
import argparse
from pathlib import Path
from ._betti import betti_curves
from ._io import load_image


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compute cubical Betti curves using connectivity and Euler characteristic.')
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--accelerator', choices=('python', 'numba'), default='python')
    parser.add_argument('--max-voxels', type=int, default=4_000_000)
    args = parser.parse_args(argv)
    output = args.output or args.input.with_name(args.input.stem + '_betti_curves.csv')
    curves = betti_curves(load_image(args.input), accelerator=args.accelerator,
                          max_voxels=args.max_voxels)
    curves.to_csv(output)
    print(f'Wrote {len(curves.levels)} Betti curve samples to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
