"""Reproducible selected-witness timing; no bundled datasets.

Run: PYTHONPATH=src python benchmarks/benchmark_selected_witnesses.py --size 10
Use --size 14 or larger only when adequate time/memory are available.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from persistence_pairing_graph import PPGAnalysis, transition_filling_witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.size < 5 or args.repeats < 1:
        parser.error('--size must be >=5 and --repeats must be >=1')
    vol = np.zeros((args.size,) * 3, dtype=np.uint8)
    vol[args.size // 2, args.size // 2, args.size // 2] = 90
    records = []
    for repeat in range(args.repeats):
        t0 = perf_counter()
        with PPGAnalysis(vol, max_original_cells=(2 * args.size + 1) ** 3,
                         max_basis_columns=100_000) as analysis:
            build_seconds = perf_counter() - t0
            sources = [i for i, node in enumerate(analysis.nodes)
                       if node.dimension == 2 and not node.essential]
            if len(sources) != 1:
                raise RuntimeError('Expected one enclosed-cavity H2 class')
            source = sources[0]
            t1 = perf_counter()
            first = analysis.witness(source, solver='component')
            first_seconds = perf_counter() - t1
            first_stats = analysis.last_component_stats
            t2 = perf_counter()
            second = analysis.witness(source, solver='component')
            repeated_seconds = perf_counter() - t2
            if not np.array_equal(first.filling_grid_indices, second.filling_grid_indices):
                raise RuntimeError('Repeated analysis changed witness')
            records.append({
                'repeat': repeat, 'size': args.size, 'build_seconds': build_seconds,
                'first_query_seconds': first_seconds,
                'repeated_query_seconds': repeated_seconds,
                'canceled_death_candidates': first_stats.eligible_deaths,
                'selected_death_candidates': first_stats.selected_deaths,
                'visited_faces': first_stats.visited_faces,
                'filling_cells': first.filling_cell_count,
                'cached_cycles': analysis.cached_cycle_count,
                'event_solves': analysis.event_solves,
            })
    print(json.dumps(records, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
