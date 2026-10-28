"""Exact support-constrained fillings and dual obstructions for PPG relations.

This is a bounded local oracle, not a scalable reduction or an optimizer.
All chains/cochains are over GF(2) on the original doubled cubical grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Iterable, Sequence

import numpy as np

from ._lift import boundary_old_chain
from ._transition_witness import _original_cell_birth


@dataclass(frozen=True, slots=True)
class FillingLocalityCertificate:
    """Either an exact filling, or a verified cochain obstruction.

    A feasible result may have an empty filling if the relation is zero.
    On failure, the obstruction pairs to 1 with the relation and to 0 with
    every boundary of an allowed filling cell. It proves impossibility only
    within the supplied allowed set, not in the full complex.
    """
    feasible: bool
    dimension: int
    grid_shape: tuple[int, ...]
    relation_cells: np.ndarray
    allowed_cell_count: int
    boundary_row_count: int
    boundary_rank: int
    filling_cells: np.ndarray
    obstruction_cells: np.ndarray
    xor_steps: int


def _support(cells: Iterable[int], total: int, *, toggle: bool) -> list[int]:
    result: set[int] = set()
    for value in cells:
        index = int(value)
        if index < 0 or index >= total:
            raise ValueError(f'Cell ID {index} is outside the doubled grid')
        if toggle and index in result:
            result.remove(index)
        else:
            result.add(index)
    return sorted(result)


def _dimension(cell: int, shape: Sequence[int], strides: Sequence[int]) -> int:
    return sum((cell // stride) % extent & 1
               for extent, stride in zip(shape, strides))


def _faces(cell: int, shape: Sequence[int], strides: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for extent, stride in zip(shape, strides):
        if (cell // stride) % extent & 1:
            result.extend((cell - stride, cell + stride))
    return tuple(result)


def certify_local_filling(
    relation_cells: Iterable[int],
    allowed_filling_cells: Iterable[int],
    image_shape: Sequence[int],
    *,
    dimension: int,
    max_allowed_cells: int = 10_000,
    max_boundary_rows: int = 30_000,
    max_xor_steps: int = 2_000_000,
) -> FillingLocalityCertificate:
    """Decide if a q-cycle bounds using specified original-grid (q+1)-cells.

    Duplicate relation IDs cancel modulo 2, while duplicate allowed IDs
    are ignored. Both feasibility and the dual obstruction are verified.
    Bounded Gaussian elimination can still have quadratic storage or worse
    time in the configured problem size; no optimal-support claim is made.
    """
    shape = tuple(int(x) for x in image_shape)
    if len(shape) not in (2, 3) or any(x <= 0 for x in shape):
        raise ValueError('image_shape must be a nonempty 2D or 3D shape')
    if not 0 <= dimension < len(shape):
        raise ValueError('dimension must be between zero and image.ndim-1')
    if min(max_allowed_cells, max_boundary_rows, max_xor_steps) <= 0:
        raise ValueError('All limits must be positive')
    grid_shape = tuple(2 * x + 1 for x in shape)
    strides = tuple(prod(grid_shape[i + 1:]) for i in range(len(shape)))
    total = prod(grid_shape)
    relation = _support(relation_cells, total, toggle=True)
    allowed = _support(allowed_filling_cells, total, toggle=False)
    if len(allowed) > max_allowed_cells:
        raise MemoryError('Allowed filling-cell count exceeds max_allowed_cells')
    for cell in relation:
        if _dimension(cell, grid_shape, strides) != dimension:
            raise ValueError('Relation contains a cell of the wrong dimension')
    for cell in allowed:
        if _dimension(cell, grid_shape, strides) != dimension + 1:
            raise ValueError('Allowed filling set contains a cell of the wrong dimension')
    if len(boundary_old_chain(relation, grid_shape)):
        raise ValueError('The relation is not a cycle')

    columns = [_faces(cell, grid_shape, strides) for cell in allowed]
    row_set = set(relation)
    for column in columns:
        row_set.update(column)
    rows = sorted(row_set)
    if len(rows) > max_boundary_rows:
        raise MemoryError('Local boundary-row count exceeds max_boundary_rows')
    row_index = {cell: i for i, cell in enumerate(rows)}
    pivots: dict[int, tuple[int, int]] = {}
    xor_steps = 0
    for j, faces in enumerate(columns):
        bits = 0
        for face in faces:
            bits ^= 1 << row_index[face]
        combo = 1 << j
        while bits:
            pivot = bits.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = (bits, combo)
                break
            bits ^= pivots[pivot][0]
            combo ^= pivots[pivot][1]
            xor_steps += 1
            if xor_steps > max_xor_steps:
                raise MemoryError('Local elimination exceeds max_xor_steps')

    target = 0
    for cell in relation:
        target ^= 1 << row_index[cell]
    remainder = target
    combo = 0
    # Reduce every pivot row, including those below a free row. This leaves
    # a remainder supported entirely on free rows for the dual certificate.
    for pivot in sorted(pivots, reverse=True):
        if (remainder >> pivot) & 1:
            remainder ^= pivots[pivot][0]
            combo ^= pivots[pivot][1]
            xor_steps += 1
            if xor_steps > max_xor_steps:
                raise MemoryError('Local elimination exceeds max_xor_steps')

    if remainder == 0:
        filling = [allowed[j] for j in range(len(allowed)) if (combo >> j) & 1]
        if list(boundary_old_chain(filling, grid_shape)) != relation:
            raise RuntimeError('Internal local filling verification failed')
        return FillingLocalityCertificate(
            True, dimension, grid_shape, np.asarray(relation, dtype=np.int64),
            len(allowed), len(rows), len(pivots),
            np.asarray(filling, dtype=np.int64), np.empty(0, dtype=np.int64),
            xor_steps)

    # Choose a free row in the nonzero normal form. Extend its coordinate
    # functional so it annihilates each pivot column, in ascending pivot order.
    free_row = remainder.bit_length() - 1
    functional = 1 << free_row
    for pivot in sorted(pivots):
        reduced = pivots[pivot][0]
        if ((reduced ^ (1 << pivot)) & functional).bit_count() & 1:
            functional |= 1 << pivot
    obstruction = [cell for j, cell in enumerate(rows)
                   if (functional >> j) & 1]
    obstruction_set = set(obstruction)
    if (functional & target).bit_count() % 2 != 1:
        raise RuntimeError('Internal dual evaluation failed')
    if any(sum(face in obstruction_set for face in col) % 2 for col in columns):
        raise RuntimeError('Internal dual annihilation failed')
    return FillingLocalityCertificate(
        False, dimension, grid_shape, np.asarray(relation, dtype=np.int64),
        len(allowed), len(rows), len(pivots),
        np.empty(0, dtype=np.int64),
        np.asarray(obstruction, dtype=np.int64), xor_steps)


def eligible_filling_cells_in_box(
    image: np.ndarray,
    *,
    dimension: int,
    event_value: int | float,
    lower: Sequence[int],
    upper: Sequence[int],
    max_candidates: int = 10_000,
    max_examined_cells: int = 1_000_000,
) -> np.ndarray:
    """Find eligible (q+1)-cells in inclusive doubled-grid coordinate bounds.

    Eligible means filtration value <= event_value, using the repo's
    lower-star cubical convention. This helper supports uint8/uint16 images;
    other filtration conventions should pass explicit eligible cell IDs
    directly to certify_local_filling.
    """
    image = np.asarray(image)
    if (image.ndim not in (2, 3) or any(x <= 0 for x in image.shape)
            or image.dtype not in (np.dtype('uint8'), np.dtype('uint16'))):
        raise ValueError('Expected nonempty 2D/3D uint8 or uint16 image')
    if not 0 <= dimension < image.ndim:
        raise ValueError('Invalid homology dimension')
    if not np.isfinite(event_value) or min(max_candidates, max_examined_cells) <= 0:
        raise ValueError('Invalid event value or resource limit')
    grid_shape = tuple(2 * x + 1 for x in image.shape)
    low, high = tuple(map(int, lower)), tuple(map(int, upper))
    if (len(low) != image.ndim or len(high) != image.ndim
            or any(a < 0 or b < a or b >= g for a, b, g in zip(low, high, grid_shape))):
        raise ValueError('Bounds must be ordered and inside the doubled grid')
    if prod(b-a+1 for a,b in zip(low, high)) > max_examined_cells:
        raise MemoryError('Box exceeds max_examined_cells')
    result: list[int] = []
    for coords in product(*(range(a, b + 1) for a, b in zip(low, high))):
        if sum(c & 1 for c in coords) != dimension + 1:
            continue
        cell = int(np.ravel_multi_index(coords, grid_shape))
        if _original_cell_birth(image, cell, grid_shape) <= event_value:
            result.append(cell)
            if len(result) > max_candidates:
                raise MemoryError('Box exceeds max_candidates')
    return np.asarray(result, dtype=np.int64)


def certify_transition_filling_locality(
    image: np.ndarray,
    witness,
    *,
    lower: Sequence[int],
    upper: Sequence[int],
    max_allowed_cells: int = 10_000,
    max_boundary_rows: int = 30_000,
    max_xor_steps: int = 2_000_000,
    max_examined_cells: int = 1_000_000,
) -> FillingLocalityCertificate:
    """Check whether a complete PPG death relation fills inside a given box.

    The box constrains filling cells, not necessarily the relation's cycles.
    Witness should be a TransitionFillingWitness from the existing API.
    """
    image = np.asarray(image)
    if tuple(image.shape) != tuple(witness.image_shape):
        raise ValueError('Witness shape does not match the image')
    allowed = eligible_filling_cells_in_box(
        image, dimension=witness.dimension, event_value=witness.event_value,
        lower=lower, upper=upper, max_candidates=max_allowed_cells,
        max_examined_cells=max_examined_cells)
    return certify_local_filling(
        witness.relation_support, allowed, image.shape,
        dimension=witness.dimension, max_allowed_cells=max_allowed_cells,
        max_boundary_rows=max_boundary_rows, max_xor_steps=max_xor_steps)
