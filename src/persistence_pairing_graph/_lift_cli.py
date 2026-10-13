"""Command-line export of verified filtered original-grid cycle supports."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from ._io import load_image
from ._lift import representative_cycles_streaming


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='ppg-lift',
        description='Lift streaming PPG birth cycles into original doubled-grid cells.')
    parser.add_argument('input', type=Path, help='Grayscale 2D or 3D image')
    parser.add_argument('--output', type=Path, help='Compressed .npz output')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--max-features', type=int, default=10)
    parser.add_argument('--min-persistence', type=float, default=0.0)
    parser.add_argument('--max-basis-columns', type=int, default=200000)
    parser.add_argument('--max-basis-entries', type=int, default=5000000)
    parser.add_argument('--max-chain-terms', type=int, default=500000)
    parser.add_argument('--temp-dir', type=Path)
    parser.add_argument('--accelerator', choices=['python', 'numba'], default='python')
    args = parser.parse_args(argv)
    image = load_image(str(args.input))
    cycles = representative_cycles_streaming(
        image, dimensions=tuple(args.dimensions), max_features=args.max_features,
        min_persistence=args.min_persistence,
        max_basis_columns=args.max_basis_columns,
        max_basis_entries=args.max_basis_entries,
        max_chain_terms=args.max_chain_terms,
        temp_dir=args.temp_dir, accelerator=args.accelerator)
    output = args.output if args.output is not None else args.input.with_name(
        args.input.stem + '_lifted_cycles.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(cycles)+1, dtype=np.int64)
    for index, cycle in enumerate(cycles):
        offsets[index+1] = offsets[index] + len(cycle.original_grid_indices)
    support = (np.concatenate([c.original_grid_indices for c in cycles])
               if cycles else np.empty(0, dtype=np.int64))
    np.savez_compressed(
        output,
        image_shape=np.asarray(image.shape, dtype=np.int64),
        doubled_grid_shape=np.asarray(tuple(2*s+1 for s in image.shape), dtype=np.int64),
        dimensions=np.asarray([c.dimension for c in cycles], dtype=np.int16),
        compressed_birth_cells=np.asarray([c.birth_cell for c in cycles], dtype=np.int64),
        compressed_death_cells=np.asarray([c.death_cell if c.death_cell is not None else -1
                                           for c in cycles], dtype=np.int64),
        birth_values=np.asarray([c.birth_value for c in cycles], dtype=image.dtype),
        death_values=np.asarray([c.death_value if c.death_cell is not None else np.inf
                                  for c in cycles], dtype=np.float64),
        support_offsets=offsets,
        original_grid_cell_ids=support,
    )
    print(f'Exported {len(cycles)} verified cycles to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
