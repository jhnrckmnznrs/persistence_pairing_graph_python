"""Exact, bounded death-event filling witnesses on the original doubled grid.

RESEARCH ALGORITHM: the selective solver uses a compressed filling and sparse
canceled-death correction; the independent full-grid reference remains bounded.
Both verify the whole death-event relation, not an individual PPG edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from itertools import product
from typing import Sequence

import numpy as np

from ._cubical import _expanded_values
from ._graph import build_nodes, iter_edges
from ._lift import (boundary_old_chain, lift_compressed_cycles, LiftedCycleRepresentative,
                    _compressed_v_column, _BoundarySolver)
from ._reduction import reduce_boundary
from ._streaming import compress_discrete_tensor_streaming


def _boundary_faces(cell_id: int, grid_shape: Sequence[int], strides: Sequence[int]) -> tuple[int, ...]:
    faces = []
    for extent, stride in zip(grid_shape, strides):
        if (cell_id // stride) % extent & 1:
            faces.extend((cell_id - stride, cell_id + stride))
    return tuple(faces)


class _BoundedBoundarySpan:
    """Exact GF(2) elimination with sparse output and bounded input size."""
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.strides = tuple(prod(shape[i + 1:]) for i in range(len(shape)))
        self.columns: list[int] = []  # original grid IDs, appended in filtration order
        self.pivots: dict[int, tuple[int, int]] = {}  # row -> (boundary bits, chosen columns bits)

    def add(self, original_cell: int):
        index = len(self.columns)
        self.columns.append(original_cell)
        boundary = 0
        for face in _boundary_faces(original_cell, self.shape, self.strides):
            boundary ^= 1 << face
        combo = 1 << index
        while boundary:
            low = boundary.bit_length() - 1
            reducer = self.pivots.get(low)
            if reducer is None:
                self.pivots[low] = (boundary, combo)
                break
            boundary ^= reducer[0]
            combo ^= reducer[1]

    def solve(self, target: Sequence[int]) -> np.ndarray | None:
        boundary = 0
        for face in target:
            boundary ^= 1 << int(face)
        combo = 0
        while boundary:
            low = boundary.bit_length() - 1
            reducer = self.pivots.get(low)
            if reducer is None:
                return None
            boundary ^= reducer[0]
            combo ^= reducer[1]
        result = []
        while combo:
            bit = combo & -combo
            combo ^= bit
            result.append(self.columns[bit.bit_length() - 1])
        return np.asarray(sorted(result), dtype=np.int64)


@dataclass(slots=True)
class TransitionFillingWitness:
    """A verified original-grid certificate for an entire PPG death relation.

    Over F2: boundary(filling) == source_cycle XOR all target_cycles.
    This certificate is not unique and is not a canonical minimal filling.
    """
    source_node: int
    target_nodes: tuple[int, ...]
    dimension: int
    event_value: int
    image_shape: tuple[int, ...]
    source_cycle: LiftedCycleRepresentative
    target_cycles: dict[int, LiftedCycleRepresentative]
    relation_support: np.ndarray
    filling_grid_indices: np.ndarray
    filling_doubled_coordinates: np.ndarray
    predeath_obstruction_verified: bool
    compressed_filling_cells: np.ndarray
    cancelled_death_correction_cells: np.ndarray
    solver: str

    @property
    def filling_cell_count(self) -> int:
        return len(self.filling_grid_indices)



def _compressed_filling(compressed, reduction, compressed_relation: set[int], t,
                        *, max_work: int) -> np.ndarray:
    """Solve d'(A')=relation from R'=D'V'; never form the full boundary matrix."""
    death_of_pivot = {int(low): int(j) for j, low in enumerate(reduction.low) if low >= 0}
    remaining = set(compressed_relation)
    reduced_deaths: list[int] = []
    while remaining:
        pivot = max(remaining)
        death = death_of_pivot.get(pivot)
        if death is None or compressed.values[death] > t:
            raise RuntimeError('Compressed transition is not a boundary at the death value')
        reduced_deaths.append(death)
        remaining.symmetric_difference_update(reduction.reduced_columns[death])
    filling: set[int] = set()
    for death in reduced_deaths:
        filling.symmetric_difference_update(_compressed_v_column(reduction, death, max_work=max_work))
    if not filling:
        raise RuntimeError('Expected a nonempty compressed filling')
    if any(int(compressed.dimensions[cell]) != int(compressed.dimensions[next(iter(compressed_relation))]) + 1
           for cell in filling):
        raise RuntimeError('Compressed filling has inconsistent cell dimensions')
    if max(compressed.values[cell] for cell in filling) != t:
        raise RuntimeError('Compressed filling does not use an event-level generator')
    boundary: set[int] = set()
    for cell in filling:
        boundary.symmetric_difference_update(compressed.boundary(cell))
    if boundary != compressed_relation:
        raise RuntimeError('Compressed filling fails its boundary equation')
    return np.asarray(sorted(filling), dtype=np.int64)



def _boundary_before_value(compressed, reduction, relation: set[int], t: int) -> bool:
    """Whether a compressed cycle bounds using columns born STRICTLY before t.

    Canonically reduced columns R_j span exactly the same filtered boundary
    space as the input columns D_j.  This check is equivalent to the original
    grid check under the equal-value filtered chain contraction.
    """
    deaths = {int(pivot): int(j) for j, pivot in enumerate(reduction.low) if pivot >= 0}
    remaining = set(relation)
    while remaining:
        pivot = max(remaining)
        j = deaths.get(pivot)
        if j is None or int(compressed.values[j]) >= int(t):
            return False
        remaining.symmetric_difference_update(reduction.reduced_columns[j])
    return True


def _original_cell_birth(image: np.ndarray, old_id: int, grid_shape: tuple[int, ...]) -> int:
    """Evaluate one cubical cell's level without expanding the entire grid."""
    coordinates = np.unravel_index(int(old_id), grid_shape)
    incident = []
    for coord, extent in zip(coordinates, image.shape):
        if coord & 1:
            incident.append((coord // 2,))
        else:
            k = coord // 2
            incident.append(tuple(i for i in (k - 1, k) if 0 <= i < extent))
    return min(int(image[index]) for index in product(*incident))


def transition_filling_witness(
    image: np.ndarray, source_node: int, *,
    max_original_cells: int = 20_000,
    max_boundary_columns: int = 6_000,
    temp_dir: str | None = None,
    reduction_storage: str = 'packed',
    accelerator: str = 'python',
    solver: str = 'pair-trace',
    max_basis_columns: int = 200_000,
    max_basis_entries: int = 5_000_000,
    max_chain_terms: int = 500_000,
    max_work: int = 5_000_000,
) -> TransitionFillingWitness:
    """Construct one original-grid (q+1)-chain that fills the entire relation.

    Both modes compute the canonical outgoing relation and lift its cycles.
    The 'pair-trace' mode performs a selective sparse correction and verifies
    its original-grid boundary without materializing a full-grid bitset;
    'full-reference' retains a complete bounded independent original-grid solve.
    Both remain resource-bounded and are not fully out-of-core implementations.

    Verifies that the requested relation was NOT a boundary before the death
    level, but becomes a boundary at it. Original-grid cell IDs are row-major
    in the (2*n+1) doubled cubical grid.
    """
    image = np.asarray(image)
    if image.ndim not in (2, 3) or image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError('Expected a 2D/3D uint8 or uint16 grayscale image')
    if not np.all(np.isfinite(image)):
        raise ValueError('Scalar field must contain finite values')
    if solver not in ('pair-trace', 'full-reference'):
        raise ValueError("solver must be 'pair-trace' or 'full-reference'")
    if source_node < 0 or max_original_cells <= 0 or max_boundary_columns <= 0:
        raise ValueError('Invalid source node or resource limit')
    grid_shape = tuple(2 * int(s) + 1 for s in image.shape)
    total = prod(grid_shape)
    if total > max_original_cells:
        raise MemoryError(f'Original grid has {total} cells; reference limit is {max_original_cells}')

    compressed = compress_discrete_tensor_streaming(
        image, max_cells=max_original_cells, temp_dir=temp_dir,
        retain_chain_lifts=True, retain_cell_provenance=True,
        accelerator=accelerator)
    try:
        reduction = reduce_boundary(compressed, storage=reduction_storage,
                                    log_mode='full', accelerator=accelerator, temp_dir=temp_dir)
        try:
            nodes = build_nodes(compressed.values, compressed.dimensions, reduction, storage='compact')
            if source_node >= len(nodes):
                raise IndexError('source_node is not a persistence node')
            node = nodes[source_node]
            if node.essential or node.dimension >= image.ndim:
                raise ValueError('Source must be a finite interval with filling dimension in the image')
            t = node.death_value
            if not np.isfinite(t):
                raise ValueError('A finite event value is required')
            targets = sorted({e.to_node for e in iter_edges(compressed.values, reduction, nodes)
                              if e.from_node == source_node})
            if any(not (nodes[target].dimension == node.dimension and
                        nodes[target].birth_value < t < nodes[target].death_value)
                   for target in targets):
                raise RuntimeError('PPG row contains invalid surviving target')
            ids = [source_node, *targets]
            cycles = lift_compressed_cycles(
                compressed, reduction, nodes,
                dimensions=(node.dimension,), max_features=len(ids), node_indices=ids,
                max_basis_columns=max_basis_columns, max_basis_entries=max_basis_entries,
                max_chain_terms=max_chain_terms, max_work=max_work)
            reps = dict(zip(ids, cycles))
            relation = set()
            for idx in ids:
                relation.symmetric_difference_update(map(int, reps[idx].original_grid_indices))
            relation_support = np.asarray(sorted(relation), dtype=np.int64)
            if len(boundary_old_chain(relation_support, grid_shape)):
                raise RuntimeError('Whole PPG relation is not an original-grid cycle')
            compressed_relation: set[int] = set()
            for rep in reps.values():
                compressed_relation.symmetric_difference_update(map(int, rep.compressed_cells))
            compressed_filling = _compressed_filling(compressed, reduction,
                    compressed_relation, t, max_work=max_work)
            if _boundary_before_value(compressed, reduction, compressed_relation, t):
                raise RuntimeError('The PPG relation was already a boundary before the event')
            filling_seed: set[int] = set(map(int, compressed.original_grid_indices[compressed_filling]))
            corrected_boundary = relation ^ set(boundary_old_chain(filling_seed, grid_shape))
            # Sparse original-cell-ID boundary elimination: no full-grid bitset.
            # The theorem documented in README proves this subspace suffices.
            pair_span = _BoundarySolver(grid_shape, max_basis_columns=max_boundary_columns,
                                        max_basis_entries=max_basis_entries)
            for pair in compressed.lift_trace.iter_pairs():
                if int(pair['dimension']) == node.dimension + 1 and int(pair['value']) <= t:
                    pair_span.add(int(pair['death']))
            correction = pair_span.solve_boundary(sorted(corrected_boundary),
                                                   max_chain_terms=max_chain_terms)
            lifted_filling = np.asarray(sorted(filling_seed ^ set(map(int, correction))), dtype=np.int64)
            if not np.array_equal(boundary_old_chain(lifted_filling, grid_shape), relation_support):
                raise RuntimeError('Pair-trace filling fails the original boundary equation')
        finally:
            reduction.close()
    finally:
        if compressed.lift_trace is not None:
            compressed.lift_trace.close()

    if solver == 'full-reference':
        # Independent, intentionally bounded oracle: complete original-grid
        # filtration, including a separate strict-predeath span check.
        expanded = _expanded_values(image)
        levels = expanded.ravel()
        original_indices = np.arange(total, dtype=np.int64)
        axes = np.unravel_index(original_indices, grid_shape)
        dimensions = sum((axis & 1).astype(np.uint8) for axis in axes)
        qualified = np.flatnonzero((dimensions == node.dimension + 1) & (levels <= t))
        if len(qualified) > max_boundary_columns:
            raise MemoryError(f'Potential filling cells {len(qualified)} exceed {max_boundary_columns}')
        order = qualified[np.lexsort((qualified, levels[qualified]))]
        span = _BoundedBoundarySpan(grid_shape)
        for cell in order:
            if levels[cell] >= t:
                break
            span.add(int(cell))
        if span.solve(relation_support) is not None:
            raise RuntimeError('The PPG relation was already a boundary before the event')
        for cell in order:
            if levels[cell] == t:
                span.add(int(cell))
        filling = span.solve(relation_support)
        if filling is None:
            raise RuntimeError('Could not fill the canonical death relation at the event value')
    else:
        filling = lifted_filling
    if not len(filling) or not any(_original_cell_birth(image, c, grid_shape) == t for c in filling):
        raise RuntimeError('Filling lacks any event-level cell')
    if any(_original_cell_birth(image, c, grid_shape) > t for c in filling):
        raise RuntimeError('Filling contains cells born after the event')
    if not np.array_equal(boundary_old_chain(filling, grid_shape), relation_support):
        raise RuntimeError('Filling has the wrong cubical boundary')
    if any(sum(coord & 1 for coord in np.unravel_index(int(c), grid_shape)) != node.dimension + 1
           for c in filling):
        raise RuntimeError('Filling has wrong cell dimensions')
    coords = np.column_stack(np.unravel_index(filling, grid_shape))
    return TransitionFillingWitness(int(source_node), tuple(targets), node.dimension, int(t),
                                    tuple(image.shape), reps[source_node],
                                    {k: reps[k] for k in targets},
                                    relation_support, filling, coords, True,
                                    compressed_filling, correction, solver)
