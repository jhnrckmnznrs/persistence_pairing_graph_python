"""Explicit spatial coarsening and matrix-free multiscale topology.

Coarsening does NOT preserve persistent homology, and a coarse persistence
point is not automatically the same feature as a fine-scale point.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ._betti import BettiCurves, betti_curves


def pool_scalar_field(image: np.ndarray, factor: int = 2, *, reducer: str = 'min') -> np.ndarray:
    """Pool aligned nonoverlapping scalar blocks, including partial edge blocks.

    'min' uses the smallest intensity in a block (existential pooling); 'max'
    uses the largest (universal pooling). Neither is a deformation retraction
    or persistence-preserving transformation in general.
    """
    image = np.asarray(image)
    if image.ndim not in (2, 3) or not all(image.shape):
        raise ValueError('Expected a nonempty 2D or 3D scalar image')
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError('factor must be a positive integer')
    if reducer not in ('min', 'max'):
        raise ValueError("reducer must be 'min' or 'max'")
    if factor == 1:
        return image.copy()
    padding = [(0, (-int(n)) % factor) for n in image.shape]
    # Edge replication is equivalent to reducing each incomplete last block
    # over its observed entries: replicated values do not affect min/max.
    padded = np.pad(image, padding, mode='edge')
    dimensions = tuple(dim for n in image.shape for dim in ((n + factor - 1) // factor, factor))
    reshaped = padded.reshape(dimensions)
    axes = tuple(2 * i + 1 for i in range(image.ndim))
    return (np.min(reshaped, axis=axes) if reducer == 'min'
            else np.max(reshaped, axis=axes))


@dataclass(slots=True)
class MultiscaleBetti:
    factor: int
    shape: tuple[int, ...]
    reducer: str
    curves: BettiCurves


def multiscale_betti_curves(image: np.ndarray, *, factors=(1, 2, 4),
                            reducer: str = 'min', accelerator: str = 'python',
                            max_voxels: int = 4_000_000) -> list[MultiscaleBetti]:
    """Compute independent Betti curves on explicitly pooled images.

    The results describe DIFFERENT discretizations. They are not vineyard
    trajectories or matched persistence classes across spatial resolutions.
    """
    results = []
    if len(set(factors)) != len(factors):
        raise ValueError('factors must be distinct')
    for factor in factors:
        pooled = pool_scalar_field(image, factor, reducer=reducer)
        curves = betti_curves(pooled, accelerator=accelerator, max_voxels=max_voxels)
        results.append(MultiscaleBetti(int(factor), pooled.shape, reducer, curves))
    return results
