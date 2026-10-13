"""Independent-process Rust/Python differential PPG validation.

Usage:
  python tests/compare_rust.py --rust-bin /path/to/persistence_pairing_graph --image image.tif

This tool requires the Rust binary to be built separately. Both tools execute
on private image copies so they cannot overwrite the user's outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter


def _rows(path: Path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def compare_exports(rust_dir: Path, python_dir: Path, stem: str):
    """Check exact cell-pair signatures and graph transitions with multiplicity."""
    suffix = '_persistence_pairing_graph_'
    rnodes = _rows(rust_dir / (stem + suffix + 'nodes.csv'))
    pnodes = _rows(python_dir / (stem + suffix + 'nodes.csv'))
    redges = _rows(rust_dir / (stem + suffix + 'edges.csv'))
    pedges = _rows(python_dir / (stem + suffix + 'edges.csv'))

    def normalize_nodes(rows):
        return Counter((r['dimension'], r['birth_value'], r['death_value'], r['essential'])
                       for r in rows)

    def normalize_edges(rows):
        return Counter((r['dimension'], r['from_birth_value'], r['from_death_value'],
                        r['to_birth_value'], r['to_death_value'], r['coefficient'])
                       for r in rows)

    def compare_counters(label, rust, python):
        if rust == python:
            return
        only_rust = list((rust - python).items())[:5]
        only_python = list((python - rust).items())[:5]
        raise AssertionError(f'{label} differ; only in Rust: {only_rust}; '
                             f'only in Python: {only_python}')

    compare_counters('Persistence intervals', normalize_nodes(rnodes), normalize_nodes(pnodes))
    compare_counters('Directed PPG transitions', normalize_edges(redges), normalize_edges(pedges))

    # Stronger check: canonical compressed generator numbering and exact edge ids.
    node_ids = ('node_id', 'dimension', 'birth_cell', 'death_cell',
                'birth_value', 'death_value', 'essential')
    edge_ids = ('dimension', 'from_node', 'to_node', 'coefficient')
    compare_counters('Canonical persistence-cell pairings',
                     Counter(tuple(r[k] for k in node_ids) for r in rnodes),
                     Counter(tuple(r[k] for k in node_ids) for r in pnodes))
    compare_counters('Graph node IDs and directed edges',
                     Counter(tuple(r[k] for k in edge_ids) for r in redges),
                     Counter(tuple(r[k] for k in edge_ids) for r in pedges))

    provenance_name = stem + '_compressed_cell_provenance.csv'
    rprov = rust_dir / provenance_name
    pprov = python_dir / provenance_name
    if rprov.exists() and pprov.exists():
        if _rows(rprov) != _rows(pprov):
            raise AssertionError('Python/Rust compressed cell provenance differs')
    return {'nodes': len(rnodes), 'edges': len(redges),
            'provenance_checked': rprov.exists() and pprov.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rust-bin', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--algorithm', choices=['stream-block', 'block', 'lazy'],
                        default='stream-block')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if not args.rust_bin.is_file() or not args.image.is_file():
        parser.error('Provide an existing Rust binary and an image')
    if args.algorithm == 'lazy':
        # The original uncompressed reduction has the same cell order.
        rust_name = 'lazy'
    else:
        rust_name = args.algorithm
    with tempfile.TemporaryDirectory(prefix='ppg_diff_') as temporary:
        root = Path(temporary)
        timings = {}
        for implementation in ('rust', 'python'):
            work = root / implementation
            work.mkdir()
            source = work / args.image.name
            shutil.copyfile(args.image, source)
            if implementation == 'rust':
                command = [str(args.rust_bin.resolve()), str(source),
                           '--algorithm', rust_name]
                if args.algorithm == 'stream-block':
                    command.append('--export-cell-provenance')
            else:
                command = [sys.executable, '-m', 'persistence_pairing_graph', str(source),
                           '--algorithm', args.algorithm]
                if args.algorithm == 'stream-block':
                    command.append('--export-cell-provenance')
            t = perf_counter()
            subprocess.run(command, check=True, timeout=args.timeout, capture_output=True, text=True)
            timings[implementation] = round(perf_counter() - t, 4)
        checks = compare_exports(root / 'rust', root / 'python', args.image.stem)
    print({'equivalent': True, 'checked': checks, 'wall_seconds': timings})


if __name__ == '__main__':
    main()
