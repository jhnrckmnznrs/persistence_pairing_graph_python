"""Optional stable counting-scatter for the streaming cubical filtration.

The source is visited in doubled-grid row-major order. One serial pass
through each plane places indices at their bucket cursors, giving exactly the
canonical (value, dimension, row-major index) order without a comparison sort.
"""
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

if njit is not None:
    @njit(cache=True)
    def _scatter_plane(keys, base, cursors, output):
        for i in range(len(keys)):
            bucket = int(keys[i])
            pos = cursors[bucket]
            output[pos] = np.uint32(base + i)
            cursors[bucket] = pos + 1


def scatter_plane(keys, base, cursors, output):
    """Stable O(number of plane cells) bucket scatter; updates cursors in place."""
    if njit is None:
        raise ImportError("Counting scatter requires the optional Numba dependency")
    _scatter_plane(np.asarray(keys, dtype=np.uint32), int(base), cursors, output)
