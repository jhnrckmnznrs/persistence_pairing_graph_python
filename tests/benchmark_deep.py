"""CLI benchmark with peak resident set size (run each variant in a new process)."""
import argparse
import json
import resource
from time import perf_counter

import numpy as np

from persistence_pairing_graph import compute_ppg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--algorithm', default='stream-block')
    parser.add_argument('--order-format', choices=['delta', 'u32'], default='u32')
    parser.add_argument('--reduction-storage', choices=['packed', 'disk', 'python'], default='packed')
    parser.add_argument('--log-mode', choices=['full', 'selective'], default='full')
    parser.add_argument('--accelerator', choices=['python', 'numba'], default='python')
    parser.add_argument('--max-relation-cache-entries', type=int, default=100_000)
    args = parser.parse_args()
    array = np.random.default_rng(args.seed).integers(0, 40, (args.size,) * 3, dtype=np.uint8)
    start = perf_counter()
    graph = compute_ppg(array, algorithm=args.algorithm,
                        order_format=args.order_format,
                        reduction_storage=args.reduction_storage,
                        log_mode=args.log_mode,
                        accelerator=args.accelerator,
                        max_relation_cache_entries=args.max_relation_cache_entries)
    elapsed = perf_counter() - start
    print(json.dumps(dict(size=args.size, algorithm=args.algorithm,
                          order_format=args.order_format,
                          reduction_storage=args.reduction_storage,
                          log_mode=args.log_mode, accelerator=args.accelerator,
                          elapsed_seconds=round(elapsed, 4),
                          peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
                          generators=graph.reduced_cell_count, nodes=len(graph.nodes),
                          edges=graph.edge_count)))


if __name__ == '__main__':
    main()
