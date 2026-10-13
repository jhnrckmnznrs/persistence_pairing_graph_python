"""Phase-level profile for compiled versus pure-Python PPG backends.

Run via ``python benchmarks/profile_compiled.py --case random --size 40``.
Use separate processes for independent peak-RSS measurements. Numba is warmed
before the timed sections; the optional JIT compilation startup is excluded.
"""
from __future__ import annotations

import argparse
import json
import os
from time import perf_counter
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_structured import make_volume
import numpy as np

from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from persistence_pairing_graph._reduction import reduce_boundary
from persistence_pairing_graph._graph import build_nodes, iter_edges


def run(case, size, accelerator):
    image = make_volume(case, size)
    if accelerator == 'numba':
        warm = np.array([[[4, 1], [2, 3]], [[0, 5], [3, 1]]], dtype=np.uint8)
        small = compress_discrete_tensor_streaming(warm, accelerator='numba')
        small_reduction = reduce_boundary(small, accelerator='numba')
        small_reduction.close()
    t = perf_counter()
    complex_ = compress_discrete_tensor_streaming(image, accelerator=accelerator)
    compress = perf_counter() - t
    t = perf_counter()
    reduction = reduce_boundary(complex_, accelerator=accelerator)
    reduce = perf_counter() - t
    try:
        t = perf_counter()
        nodes = build_nodes(complex_.values, complex_.dimensions, reduction)
        node = perf_counter() - t
        t = perf_counter()
        edges = sum(1 for _ in iter_edges(complex_.values, reduction, nodes))
        graph = perf_counter() - t
        return dict(case=case,size=size,accelerator=accelerator,
                    original_cells=complex_.original_size, generators=len(complex_),
                    nodes=len(nodes), edges=edges,
                    contraction_s=round(compress,4), reduction_s=round(reduce,4),
                    nodes_s=round(node,4), graph_s=round(graph,4),
                    total_s=round(compress+reduce+node+graph,4),
                    peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2))
    finally:
        reduction.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['shell','torus','multi-cavity','lattice','checkerboard','smooth','random'], required=True)
    parser.add_argument('--size', type=int, default=40)
    parser.add_argument('--accelerator', choices=['python','numba'], required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.case,args.size,args.accelerator),sort_keys=True))

if __name__=='__main__':
    main()
