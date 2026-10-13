"""Independent cubical Betti curves using digital connectivity and Euler counts.

Foreground cubes are CLOSED, so 8/26-connectivity yields H0. The complement
of this union of closed cubes has 4/6-connectivity and a single exterior
component. Alexander duality then yields H1 (2D) or H2 (3D). Euler--Poincare
recovers the remaining H1 in 3D. No persistence pairing or graph edges are
constructed by this shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import csv
from pathlib import Path
from functools import lru_cache

import numpy as np

from ._streaming import _iter_keys, _validate


@dataclass(slots=True)
class BettiCurves:
    levels: np.ndarray
    beta0: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray
    euler: np.ndarray

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(('filtration_value', 'beta0', 'beta1', 'beta2', 'euler'))
            for row in zip(self.levels, self.beta0, self.beta1, self.beta2, self.euler):
                writer.writerow(tuple(int(x) for x in row))
        return path


def _component_series(values, shape, order, thresholds, neighbor_offsets, foreground):
    """Incremental union-find; includes a virtual exterior component for complement."""
    n = len(values)
    ndim = len(shape)
    parent = np.arange(n + 1, dtype=np.int64)
    size = np.ones(n + 1, dtype=np.int64)
    active = np.zeros(n + 1, dtype=np.uint8)
    outside = n
    active[outside] = 1
    strides = np.empty(ndim, dtype=np.int64)
    stride = 1
    for axis in range(ndim - 1, -1, -1):
        strides[axis] = stride
        stride *= shape[axis]
    result = np.empty(len(thresholds), dtype=np.int64)
    count = 0 if foreground else 1
    cursor = 0
    for k in range(len(thresholds)):
        index = k if foreground else len(thresholds) - 1 - k
        level = thresholds[index]
        if not foreground:
            # Before adding this level: exactly the voxels with value > level
            # are in the open complement of the closed cubical sublevel set.
            result[index] = count - 1
        while cursor < n and values[order[cursor]] == level:
            cell = int(order[cursor])
            cursor += 1
            active[cell] = 1
            count += 1
            if not foreground:
                boundary = False
                for axis in range(ndim):
                    coordinate = (cell // strides[axis]) % shape[axis]
                    if coordinate == 0 or coordinate == shape[axis] - 1:
                        boundary = True
                if boundary:
                    ra = cell
                    while parent[ra] != ra:
                        parent[ra] = parent[parent[ra]]
                        ra = parent[ra]
                    rb = outside
                    while parent[rb] != rb:
                        parent[rb] = parent[parent[rb]]
                        rb = parent[rb]
                    if ra != rb:
                        if size[ra] < size[rb]:
                            ra, rb = rb, ra
                        parent[rb] = ra
                        size[ra] += size[rb]
                        count -= 1
            for delta in neighbor_offsets:
                other = cell
                valid = True
                for axis in range(ndim):
                    coord = (cell // strides[axis]) % shape[axis]
                    new_coord = coord + int(delta[axis])
                    if new_coord < 0 or new_coord >= shape[axis]:
                        valid = False
                        break
                    other += int(delta[axis]) * strides[axis]
                if not valid or not active[other]:
                    continue
                ra = cell
                while parent[ra] != ra:
                    parent[ra] = parent[parent[ra]]
                    ra = parent[ra]
                rb = other
                while parent[rb] != rb:
                    parent[rb] = parent[parent[rb]]
                    rb = parent[rb]
                if ra != rb:
                    if size[ra] < size[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    size[ra] += size[rb]
                    count -= 1
        if foreground:
            result[index] = count
    return result


@lru_cache(maxsize=1)
def _compiled_components():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError("The Numba accelerator requires 'pip install numba'") from exc
    return njit(cache=True)(_component_series)


def betti_curves(image: np.ndarray, *, accelerator: str = 'python',
                 max_voxels: int = 4_000_000) -> BettiCurves:
    """Calculate exact sublevel Betti curves for finite 2D/3D uint8/uint16 images.

    The underlying top-dimensional image pixels/voxels are closed cubes.
    The result uses no persistent matrix reduction, so it does not recover
    birth/death pairs, representatives, or directed PPG edges. Boundary is
    non-periodic. Curves are returned at every DISTINCT image intensity.
    """
    image = np.asarray(image)
    grid_shape, _ = _validate(image, None)
    if accelerator not in ('python', 'numba'):
        raise ValueError("accelerator must be 'python' or 'numba'")
    if image.size > max_voxels:
        raise MemoryError(f'{image.size:,} voxels exceeds max_voxels={max_voxels:,}')
    shape = np.asarray(image.shape, dtype=np.int64)
    flattened = np.ascontiguousarray(image).ravel()
    levels = np.unique(flattened)
    order = np.argsort(flattened, kind='stable')
    foreground_neighbors = np.asarray(
        [p for p in product((-1, 0, 1), repeat=image.ndim) if any(p)], dtype=np.int8)
    background_neighbors = np.asarray(
        [p for p in product((-1, 0, 1), repeat=image.ndim)
         if sum(x != 0 for x in p) == 1], dtype=np.int8)
    component_counts = _component_series
    if accelerator == 'numba':
        component_counts = _compiled_components()
    beta0 = component_counts(flattened, shape, order, levels,
                             foreground_neighbors, True)
    # Reverse the stable foreground order to add complement from high to low.
    # All same-value pixels are added before the next threshold is sampled.
    cavities = component_counts(flattened, shape, order[::-1].copy(), levels,
                                background_neighbors, False)
    # Cubical cell histograms: a lower-dimensional face enters at the minimum
    # value of incident image cubes. The alternating cell count is Euler chi.
    dim_count = image.ndim + 1
    n_buckets = (256 if image.dtype == np.uint8 else 65536) * dim_count
    counts = np.zeros(n_buckets, dtype=np.int64)
    for _, keys in _iter_keys(image, grid_shape):
        counts += np.bincount(keys, minlength=n_buckets)
    counts = counts.reshape((-1, dim_count))
    signs = np.array([1 if d % 2 == 0 else -1 for d in range(dim_count)], dtype=np.int64)
    cumulative_euler = np.cumsum(counts @ signs)
    euler = cumulative_euler[levels.astype(np.int64)]
    if image.ndim == 2:
        beta1 = cavities
        beta2 = np.zeros_like(beta0)
        if not np.array_equal(beta0 - beta1, euler):
            raise RuntimeError('Euler--Alexander consistency check failed (2D)')
    else:
        beta2 = cavities
        beta1 = beta0 + beta2 - euler
        if np.any(beta1 < 0):
            raise RuntimeError('Euler--Alexander consistency check failed (3D)')
    return BettiCurves(levels, beta0, beta1, beta2, euler)


def validate_ppg_betti(image: np.ndarray, graph, *, accelerator: str = 'python',
                       max_voxels: int = 4_000_000) -> BettiCurves:
    """Check every PPG barcode Betti count against independent cubical topology.

    Counts bars alive at each DISTINCT input intensity (birth <= t < death).
    Raises AssertionError on the first mismatch. This checks the barcode, not
    the PPG transition matrix, so exact edge tests remain necessary.
    """
    curves = betti_curves(image, accelerator=accelerator, max_voxels=max_voxels)
    delta = np.zeros((3, len(curves.levels) + 1), dtype=np.int64)
    for node in graph.nodes:
        q = node.dimension
        if q >= 3:
            continue
        start = int(np.searchsorted(curves.levels, node.birth_value, side='left'))
        delta[q, start] += 1
        if not node.essential:
            end = int(np.searchsorted(curves.levels, node.death_value, side='left'))
            delta[q, end] -= 1
    observed = np.cumsum(delta[:, :-1], axis=1)
    for q, expected in enumerate((curves.beta0, curves.beta1, curves.beta2)):
        errors = np.flatnonzero(observed[q] != expected)
        if errors.size:
            first = int(errors[0])
            raise AssertionError(
                f'H{q} Betti mismatch at filtration {curves.levels[first]}: '
                f'barcode={observed[q, first]}, independent={expected[first]}')
    return curves
