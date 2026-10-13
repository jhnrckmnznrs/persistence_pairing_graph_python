"""Run independent-process optimization benchmarks via --module-root ROOT.

Example:
  PYTHONPATH=src python benchmarks/compare_optimizations.py --size 48 --case random
"""
import argparse
import gc
import json
import resource
import time

import numpy as np
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from persistence_pairing_graph._reduction import reduce_boundary
from persistence_pairing_graph._graph import build_nodes, iter_edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=48)
    p.add_argument('--case', choices=('random','checkerboard','correlated'), default='random')
    p.add_argument('--node-storage', choices=('compact','python'), default='compact')
    p.add_argument('--reduction-storage', choices=('packed','python'), default='packed')
    args = p.parse_args()
    n = args.size
    if args.case == 'checkerboard':
        a = (np.indices((n,n,n)).sum(axis=0) % 2 * 100).astype(np.uint8)
    elif args.case == 'correlated':
        rng = np.random.default_rng(117)
        a = rng.integers(0, 256, (n,n,n), dtype=np.uint8)
        for axis in range(3):
            a = ((a.astype(np.uint16) + np.roll(a, 1, axis).astype(np.uint16)) // 2).astype(np.uint8)
    else:
        a = np.random.default_rng(117).integers(0, 256, (n,n,n), dtype=np.uint8)
    tic = time.perf_counter()
    c = compress_discrete_tensor_streaming(a)
    t1 = time.perf_counter()
    r = reduce_boundary(c, storage=args.reduction_storage)
    t2 = time.perf_counter()
    try:
        nodes = build_nodes(c.values,c.dimensions,r,storage=args.node_storage)
    except TypeError as exc:
        if 'storage' not in str(exc) or args.node_storage != 'python':
            raise
        nodes = build_nodes(c.values,c.dimensions,r)
    t3 = time.perf_counter()
    edges = sum(1 for edge in iter_edges(c.values,r,nodes))
    t4 = time.perf_counter()
    node_payload = getattr(nodes,'nbytes', None)
    print(json.dumps({'case':args.case,'size':n,'node_storage':args.node_storage,
           'reduction_storage':args.reduction_storage,
           'cells':c.original_size,'generators':len(c), 'nodes':len(nodes),'edges':edges,
           'compression_s':t1-tic,'reduction_s':t2-t1,'nodes_s':t3-t2,
           'graph_s':t4-t3,'total_s':t4-tic,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           'packed_node_bytes':node_payload}))
    r.close()

if __name__=='__main__': main()
