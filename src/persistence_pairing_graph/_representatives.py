"""Canonical original-grid homology cycles for SMALL uncompressed filtrations.

Replays the exact V-column operations of the standard left-to-right reduction.
This intentionally does not claim to invert the filtered block contraction:
a compressed generator's one-cell provenance is not its full chain lift.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np

from ._cubical import build_cubical_complex
from ._reduction import reduce_boundary
from ._graph import build_nodes


@dataclass(slots=True)
class CycleRepresentative:
    dimension: int
    birth_cell: int
    death_cell: int | None
    birth_value: int | float
    death_value: int | float
    cells: np.ndarray
    original_grid_indices: np.ndarray
    doubled_grid_coordinates: np.ndarray

    @property
    def cell_centers(self) -> np.ndarray:
        """Centers in image-grid coordinates (one image voxel spans one unit)."""
        return self.doubled_grid_coordinates.astype(np.float64) * 0.5


def representative_cycles(image: np.ndarray, *, dimensions: Sequence[int] = (1, 2),
                          max_features: int = 10, min_persistence: float = 0.0,
                          max_cells: int = 250_000,
                          max_chain_terms: int = 500_000,
                          reduction_storage: str = 'packed') -> list[CycleRepresentative]:
    """Reconstruct canonical cycle chains from the UNCOMPRESSED lazy reducer.

    Every returned positive-persistence or essential birth has chain V_birth,
    satisfying boundary(V_birth)=0 over F2. Chains use original doubled-grid
    cells. Supports dimension 0 too, if explicitly requested. This operation
    is deliberately restricted to small inputs: dense cycles can be huge.
    This does not construct geometrically minimal cycles/cocycles, and does
    not lift the stream-block contraction (which needs its own chain map).
    """
    if max_features < 0 or max_chain_terms <= 0 or max_cells <= 0:
        raise ValueError('max_features must be nonnegative; capacity limits must be positive')
    if min_persistence < 0:
        raise ValueError('min_persistence must be nonnegative')
    if reduction_storage not in ('packed', 'python'):
        raise ValueError("Choose reduction_storage='packed' or 'python'")
    complex_ = build_cubical_complex(np.asarray(image), max_cells=max_cells)
    reduction = reduce_boundary(complex_, storage=reduction_storage, log_mode='full')
    try:
        nodes = build_nodes(complex_.values, complex_.dimensions, reduction, storage='compact')
        choices = [node for node in nodes if node.dimension in dimensions and
                   node.persistence >= min_persistence]
        # Match the node order of the canonical persistence reduction.
        choices = choices[:max_features]
        memo: dict[int, set[int]] = {}
        visited_work = 0

        def expand(column: int) -> set[int]:
            nonlocal visited_work
            if column in memo:
                return memo[column]
            # Iterative DFS prevents recursion-depth errors for long V chains.
            stack = [(column, False)]
            while stack:
                col, resolved = stack.pop()
                if col in memo:
                    continue
                ops = reduction.operations.get(col, ())
                if not resolved:
                    stack.append((col, True))
                    for source in ops:
                        if source >= col:
                            raise RuntimeError('Nontriangular canonical reduction log')
                        if source not in memo:
                            stack.append((source, False))
                    continue
                result = {col}
                for source in ops:
                    result.symmetric_difference_update(memo[source])
                    visited_work += len(memo[source])
                    if visited_work > max_chain_terms or len(result) > max_chain_terms:
                        raise MemoryError('Representative cycle expansion exceeds max_chain_terms')
                memo[col] = result
            return memo[column]

        out = []
        for node in choices:
            indices = np.fromiter(sorted(expand(node.birth_cell)), dtype=np.int64)
            if not len(indices) or np.any(complex_.dimensions[indices] != node.dimension):
                raise RuntimeError('Cycle representative has incorrect cell dimension')
            # Independently check that the reconstructed chain is a cycle.
            boundary = set()
            for j in indices:
                for face in complex_.boundary(int(j)):
                    if face in boundary:
                        boundary.remove(face)
                    else:
                        boundary.add(face)
            if boundary:
                raise RuntimeError('Canonical representative is not a cycle')
            ids = complex_.grid_indices[indices].copy()
            coordinates = np.column_stack(np.unravel_index(ids, complex_.grid_shape))
            out.append(CycleRepresentative(node.dimension, node.birth_cell, node.death_cell,
                                           node.birth_value, node.death_value, indices,
                                           ids, coordinates))
        return out
    finally:
        reduction.close()
