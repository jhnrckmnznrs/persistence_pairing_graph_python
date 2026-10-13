"""Standalone optional benchmark: python tests/benchmark_streaming.py [side] [algorithm]."""
try:
    import resource
except ImportError:  # Windows
    resource = None
import sys
from time import perf_counter

import numpy as np
from persistence_pairing_graph import compute_ppg

side = int(sys.argv[1]) if len(sys.argv) > 1 else 32
algo = sys.argv[2] if len(sys.argv) > 2 else 'stream-block'
image = np.random.default_rng(42).integers(0, 256, (side,) * 3, dtype=np.uint8)
start = perf_counter()
ppg = compute_ppg(image, algorithm=algo)
elapsed = perf_counter() - start
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else 'unavailable'
print(f'{algo} side={side} original={ppg.original_cell_count} compressed={ppg.reduced_cell_count} '
      f'nodes={len(ppg.nodes)} edges={len(ppg.edges)} '
      f'seconds={elapsed:.3f} maxrss_kib={rss}', flush=True)
