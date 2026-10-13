"""Reproducible, CPU-only structural 3-D PPG profile with phase timings.

Usage:
  python benchmarks/profile_structured.py --case torus --size 32

Every invocation profiles one case in a separate process; see
``benchmarks/run_structured_suite.py`` for comparative runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import resource
from time import perf_counter

import numpy as np

from persistence_pairing_graph._graph import build_nodes, iter_edges
from persistence_pairing_graph._reduction import reduce_boundary
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming


def make_volume(case: str, size: int, *, seed: int = 271828) -> np.ndarray:
    """An explicit scalar field; no external datasets or dependencies."""
    a = np.indices((size, size, size), dtype=np.float32)
    x, y, z = (a[i] - (size - 1) / 2 for i in range(3))
    d = np.sqrt(x*x + y*y + z*z)
    out = np.full((size, size, size), 140, dtype=np.uint8)
    if case == 'shell':
        out[(d >= size * .20) & (d <= size * .36)] = 15
        out[d < size * .20] = 110
    elif case == 'torus':
        rho = np.sqrt(x*x + y*y)
        out[np.sqrt((rho - size * .27)**2 + z*z) <= size * .105] = 10
    elif case == 'multi-cavity':
        centers = [(-.23,-.23,-.23),(.23,-.23,.23),(-.23,.23,.23),(.23,.23,-.23)]
        for cx,cy,cz in centers:
            local = np.sqrt((x-cx*size)**2+(y-cy*size)**2+(z-cz*size)**2)
            out[(local > size*.095) & (local <= size*.155)] = 15
            out[local <= size*.095] = 95
    elif case == 'lattice':
        # Thick connected orthogonal struts, yielding numerous rectangular tunnels.
        spacing = max(5,size//5)
        def near_line(u):
            return np.abs(np.remainder(u+(size-1)/2,spacing)-spacing/2) <= .8
        out[(near_line(x)&near_line(y)) | (near_line(y)&near_line(z)) | (near_line(x)&near_line(z))] = 10
    elif case == 'checkerboard':
        out[:] = np.where(((a[0]+a[1]+a[2]).astype(np.int32) % 2)==0,12,110).astype(np.uint8)
    elif case == 'smooth':
        # Correlated noise via a small sequence of separable 3-neighbor averages
        # instead of an external SciPy dependency.
        rng = np.random.default_rng(seed)
        field = rng.normal(size=out.shape).astype(np.float32)
        for _ in range(6):
            for axis in range(3):
                field = (field + np.roll(field,1,axis=axis)+np.roll(field,-1,axis=axis))/3
        field = (field-field.min())/(field.max()-field.min())
        out = np.rint(field*150).astype(np.uint8)
    elif case == 'random':
        out = np.random.default_rng(seed).integers(0, 40, out.shape, dtype=np.uint8)
    else:
        raise ValueError(f'Unknown case: {case}')
    return out


def run(case: str, size: int, storage: str, order_format: str, cache: int):
    image = make_volume(case,size)
    tic = perf_counter()
    compressed = compress_discrete_tensor_streaming(image, order_format=order_format)
    t_compress = perf_counter() - tic
    tic = perf_counter()
    reduction = reduce_boundary(compressed, storage=storage)
    t_reduce = perf_counter() - tic
    try:
        tic = perf_counter()
        nodes = build_nodes(compressed.values, compressed.dimensions, reduction)
        t_nodes = perf_counter() - tic
        tic = perf_counter()
        n_edges = 0
        edges_by_dim = Counter()
        for edge in iter_edges(compressed.values, reduction, nodes, max_relation_cache_entries=cache):
            n_edges += 1
            edges_by_dim[edge.dimension] += 1
        t_edges = perf_counter() - tic
        events = Counter((node.dimension,node.death_value) for node in nodes if not node.essential)
        n_by_dim = Counter(node.dimension for node in nodes)
        return dict(case=case, shape=[size]*3, storage=storage, order_format=order_format,
                    original_cells=compressed.original_size, reduced_cells=len(compressed),
                    nodes=len(nodes), edges=n_edges,
                    nodes_by_dimension=dict(n_by_dim),edges_by_dimension=dict(edges_by_dim),
                    finite_events=len(events), max_simultaneous_deaths=max(events.values(),default=0),
                    compress_seconds=round(t_compress,5),reduce_seconds=round(t_reduce,5),
                    nodes_seconds=round(t_nodes,5), edges_seconds=round(t_edges,5),
                    total_seconds=round(t_compress+t_reduce+t_nodes+t_edges,5),
                    peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
                    packed_reduction_bytes=reduction.packed_bytes,
                    disk_reduction_bytes=reduction.disk_bytes)
    finally:
        reduction.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['shell','torus','multi-cavity','lattice','checkerboard','smooth','random'], required=True)
    parser.add_argument('--size',type=int,default=32)
    parser.add_argument('--storage',choices=['python','packed','disk'],default='packed')
    parser.add_argument('--order-format',choices=['u32','delta'],default='u32')
    parser.add_argument('--cache',type=int,default=100_000)
    args=parser.parse_args()
    print(json.dumps(run(args.case,args.size,args.storage,args.order_format,args.cache),sort_keys=True))

if __name__=='__main__': main()
