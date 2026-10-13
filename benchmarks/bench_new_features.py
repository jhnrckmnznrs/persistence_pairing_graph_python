"""Reproducible benchmarks for direct bucket placement and independent Betti curves."""
import argparse
import json
from time import perf_counter

import numpy as np

from persistence_pairing_graph import betti_curves, compute_ppg
from persistence_pairing_graph._compiled_order import scatter_plane
from persistence_pairing_graph._streaming import _count_buckets, _iter_keys


def order_benchmark(image, builder):
    grid_shape = tuple(2 * n + 1 for n in image.shape)
    start = perf_counter()
    counts = _count_buckets(image, grid_shape, (256 if image.dtype == np.uint8 else 65536) * (image.ndim + 1))
    cursors = np.zeros(len(counts), dtype=np.int64)
    cursors[1:] = np.cumsum(counts[:-1])
    output = np.empty(int(counts.sum()), dtype=np.uint32)
    initial = cursors.copy()
    for base, keys in _iter_keys(image, grid_shape):
        if builder == 'counting':
            scatter_plane(keys, base, cursors, output)
        else:
            order = np.argsort(keys, kind='stable')
            sorted_keys = keys[order]
            bounds = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])
            for a, b in zip(bounds[:-1], bounds[1:]):
                bucket = int(sorted_keys[a])
                pos = int(cursors[bucket])
                output[pos:pos + b - a] = base + order[a:b]
                cursors[bucket] = pos + b - a
    assert np.array_equal(cursors - initial, counts)
    return round(perf_counter() - start, 5), output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=40)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    scatter_plane(np.array([0, 1], dtype=np.uint32), 0,
                  np.zeros(2, dtype=np.int64), np.empty(2, dtype=np.uint32))
    output = []
    for dtype in (np.uint8, np.uint16):
        image = np.random.default_rng(12).integers(0, 128, (args.size,) * 3, dtype=dtype)
        result = {'dtype': np.dtype(dtype).name, 'size': args.size, 'order': {}}
        expected = None
        for builder in ('sort', 'counting'):
            elapsed = []
            for _ in range(args.repeat):
                sec, ids = order_benchmark(image, builder)
                elapsed.append(sec)
                if expected is None:
                    expected = ids.copy()
                else:
                    assert np.array_equal(ids, expected)
            result['order'][builder] = elapsed
        times = []
        for accelerator in ('python', 'numba'):
            start = perf_counter()
            curves = betti_curves(image, accelerator=accelerator)
            times.append((accelerator, round(perf_counter() - start, 4), len(curves.levels)))
        result['betti'] = times
        output.append(result)
        print(result)
    if args.output:
        with open(args.output, 'w') as fp:
            json.dump(output, fp, indent=2)


if __name__ == '__main__':
    main()
