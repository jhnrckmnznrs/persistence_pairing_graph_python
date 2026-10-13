"""CLI for exporting algebraic PPG edges with verified geometric supports."""
import argparse
from pathlib import Path

from ._io import load_image
from ._edge_geometry import explain_ppg_edges


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ppg-explain', description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-prefix', type=Path, help='Output filename without .csv/.npz')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--max-edges', type=int, default=10)
    parser.add_argument('--edge-index', type=int, action='append', dest='edge_indices')
    parser.add_argument('--min-persistence', type=float, default=0)
    parser.add_argument('--spacing', type=float, nargs='+')
    parser.add_argument('--max-geometry-pairs', type=int, default=1_000_000)
    parser.add_argument('--max-basis-columns', type=int, default=200_000)
    parser.add_argument('--max-basis-entries', type=int, default=5_000_000)
    parser.add_argument('--max-chain-terms', type=int, default=500_000)
    parser.add_argument('--temp-dir', type=Path)
    parser.add_argument('--accelerator', choices=['python', 'numba'], default='python')
    args = parser.parse_args(argv)
    image = load_image(str(args.input))
    report = explain_ppg_edges(
        image, dimensions=args.dimensions, max_edges=args.max_edges,
        edge_indices=args.edge_indices, min_persistence=args.min_persistence,
        spacing=args.spacing, max_geometry_pairs=args.max_geometry_pairs,
        max_basis_columns=args.max_basis_columns, max_basis_entries=args.max_basis_entries,
        max_chain_terms=args.max_chain_terms, temp_dir=args.temp_dir,
        accelerator=args.accelerator)
    prefix = args.output_prefix or args.input.with_name(
        args.input.stem + '_ppg_edge_geometry')
    csv_path = report.write_csv(str(prefix) + '.csv')
    npz_path = report.write_npz(str(prefix) + '.npz')
    print(f'Exported {len(report.edges)} geometric edge explanations: {csv_path}, {npz_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
