"""Relative locality oracle: restrict new cells, permit all old fillings.

For A=K_{<t} and X=K_{<=t}, solve
    d_A a + F z = c,  D z = 0,
where c is an old q-cycle, a uses every eligible old (q+1)-cell, and
z uses only the newly appearing (q+1)-cells chosen by the caller.

This bounded reference implementation does not perform a full persistence
reduction and does not claim minimum support or global scalability.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from itertools import product
from math import prod
from typing import Iterable, Sequence

import numpy as np

from ._filling_locality import (
    FillingLocalityCertificate, _dimension, _support,
    certify_local_filling,
)
from ._lift import boundary_old_chain
from ._transition_witness import _original_cell_birth


@dataclass(frozen=True, slots=True)
class RelativeLocalityCertificate:
    """A complete primal filling or a dual obstruction, relative to a new-cell region.

    The old correction is unrestricted within the full sublevel complex K_{<t}.
    The returned obstruction annihilates every eligible old boundary and every
    allowed new boundary, but pairs nontrivially with the requested old cycle.
    """
    feasible: bool
    dimension: int
    event_value: int | float
    grid_shape: tuple[int, ...]
    relation_cells: np.ndarray
    old_candidate_count: int
    new_candidate_count: int
    old_filling_cells: np.ndarray
    new_filling_cells: np.ndarray
    obstruction_cells: np.ndarray
    boundary_rank: int
    xor_steps: int
    # Diagnostics for the exact old-boundary incidence closure (or full scan).
    old_cells_examined: int = 0
    old_faces_visited: int = 0
    old_solver: str = 'full'


def certify_relative_filling(
    image: np.ndarray,
    relation_cells: Iterable[int],
    allowed_new_cells: Iterable[int],
    *,
    dimension: int,
    event_value: int | float,
    max_grid_cells: int = 100_000,
    max_old_cells: int = 20_000,
    max_new_cells: int = 10_000,
    max_boundary_rows: int = 30_000,
    max_xor_steps: int = 2_000_000,
    old_solver: str = 'component',
    max_examined_cofaces: int = 500_000,
    max_component_faces: int = 200_000,
) -> RelativeLocalityCertificate:
    """Certify relative solvability while allowing all old (q+1)-cells.

    The filtration uses the repository's scalar lower-star cubical convention.
    A=K_{<event_value} and X=K_{<=event_value}; ties enter together.
    The supplied relation must already lie in C_q(A) and be a cycle.
    Allowed new cells must have filtration value *equal* to event_value.

    The algorithm enumerates all old filling cells and reduces their boundaries
    alongside the allowed new columns. Exhaustive global old-cell enumeration is
    why this is a bounded reference oracle rather than an out-of-core backend.
    """
    image = np.asarray(image)
    if (image.ndim not in (2, 3) or not all(image.shape)
            or image.dtype not in (np.dtype('uint8'), np.dtype('uint16'))):
        raise ValueError('Expected nonempty 2D/3D uint8 or uint16 image')
    if not 0 <= dimension < image.ndim or not np.isfinite(event_value):
        raise ValueError('Invalid homology dimension or event value')
    if min(max_grid_cells, max_old_cells, max_new_cells,
           max_boundary_rows, max_xor_steps, max_examined_cofaces,
           max_component_faces) <= 0:
        raise ValueError('Resource limits must be positive')
    if old_solver not in ('component', 'full'):
        raise ValueError("old_solver must be 'component' or 'full'")
    grid_shape = tuple(2 * int(n) + 1 for n in image.shape)
    total = prod(grid_shape)
    if total > max_grid_cells:
        raise MemoryError('Original grid exceeds max_grid_cells')
    strides = tuple(prod(grid_shape[i+1:]) for i in range(image.ndim))
    relation = _support(relation_cells, total, toggle=True)
    new = _support(allowed_new_cells, total, toggle=False)
    if len(new) > max_new_cells:
        raise MemoryError('New candidate count exceeds max_new_cells')

    # Validate filtration membership of the requested old cycle, not just its
    # dimension. This is essential to interpreting the dual as a relative test.
    for cell in relation:
        if _dimension(cell, grid_shape, strides) != dimension:
            raise ValueError('Relation contains a cell of the wrong dimension')
        if _original_cell_birth(image, cell, grid_shape) >= event_value:
            raise ValueError('Relation must be contained in K_{<t}')
    if len(boundary_old_chain(relation, grid_shape)):
        raise ValueError('Relation is not a cycle')
    for cell in new:
        if _dimension(cell, grid_shape, strides) != dimension + 1:
            raise ValueError('Allowed new cell has the wrong dimension')
        if _original_cell_birth(image, cell, grid_shape) != event_value:
            raise ValueError('Allowed new cells must enter exactly at the event')

    old: list[int] = []
    examined = 0
    visited_faces = 0
    if old_solver == 'full':
        # Independent bounded reference, retained for algebraic comparison.
        for cell in range(total):
            if _dimension(cell, grid_shape, strides) != dimension + 1:
                continue
            examined += 1
            if _original_cell_birth(image, cell, grid_shape) < event_value:
                old.append(cell)
                if len(old) > max_old_cells:
                    raise MemoryError('Old filling-cell count exceeds max_old_cells')
    else:
        # Every q-row of a new column is seeded, even when c has no support
        # there. A solution supported on old columns outside this incidence
        # closure would have zero boundary in its disconnected component
        # and can be discarded. Thus this restriction preserves feasibility.
        from ._filling_locality import _faces
        seeds = set(relation)
        for cell in new:
            seeds.update(_faces(cell, grid_shape, strides))
        if len(seeds) > max_component_faces:
            raise MemoryError('Relative incidence closure exceeds max_component_faces')
        pending = deque(sorted(seeds))
        faces_seen = set(seeds)
        cells_seen: set[int] = set()
        old_set: set[int] = set()
        while pending:
            face = pending.popleft()
            visited_faces += 1
            for extent, stride in zip(grid_shape, strides):
                coordinate = (face // stride) % extent
                if coordinate & 1:
                    continue
                for step in (-1, 1):
                    candidate_coordinate = coordinate + step
                    if candidate_coordinate < 0 or candidate_coordinate >= extent:
                        continue
                    candidate = face + step * stride
                    if candidate in cells_seen:
                        continue
                    cells_seen.add(candidate)
                    examined += 1
                    if examined > max_examined_cofaces:
                        raise MemoryError('Relative incidence closure exceeds max_examined_cofaces')
                    if _original_cell_birth(image, candidate, grid_shape) >= event_value:
                        continue
                    old_set.add(candidate)
                    if len(old_set) > max_old_cells:
                        raise MemoryError('Old filling-cell count exceeds max_old_cells')
                    for other in _faces(candidate, grid_shape, strides):
                        if other not in faces_seen:
                            faces_seen.add(other)
                            if len(faces_seen) > max_component_faces:
                                raise MemoryError('Relative incidence closure exceeds max_component_faces')
                            pending.append(other)
        old = sorted(old_set)
    columns = old + new
    local: FillingLocalityCertificate = certify_local_filling(
        relation, columns, image.shape, dimension=dimension,
        max_allowed_cells=max_old_cells+max_new_cells,
        max_boundary_rows=max_boundary_rows, max_xor_steps=max_xor_steps)

    if local.feasible:
        filling = set(map(int, local.filling_cells))
        old_part = np.asarray([c for c in old if c in filling], dtype=np.int64)
        new_part = np.asarray([c for c in new if c in filling], dtype=np.int64)
        if not np.array_equal(
            boundary_old_chain([*old_part, *new_part], grid_shape),
            np.asarray(relation, dtype=np.int64)):
            raise RuntimeError('Invalid reconstructed original-grid boundary')
        # Since c is old and d_A(old_part) is old, the new part must be a
        # relative cycle: all newly entering q-faces cancel in its boundary.
        for face in boundary_old_chain(new_part, grid_shape):
            if _original_cell_birth(image, int(face), grid_shape) >= event_value:
                raise RuntimeError('New portion is not a relative cycle')
        obstruction = np.empty(0, dtype=np.int64)
    else:
        old_part = np.empty(0, dtype=np.int64)
        new_part = np.empty(0, dtype=np.int64)
        obstruction = local.obstruction_cells
    return RelativeLocalityCertificate(
        local.feasible, dimension, event_value, grid_shape,
        np.asarray(relation, dtype=np.int64), len(old), len(new),
        old_part, new_part, obstruction, local.boundary_rank, local.xor_steps,
        examined, visited_faces, old_solver)


def certify_transition_relative_locality(
    image: np.ndarray,
    witness,
    *,
    lower: Sequence[int],
    upper: Sequence[int],
    max_grid_cells: int = 100_000,
    max_old_cells: int = 20_000,
    max_new_cells: int = 10_000,
    max_boundary_rows: int = 30_000,
    max_xor_steps: int = 2_000_000,
    max_examined_cells: int = 1_000_000,
    old_solver: str = 'component',
    max_examined_cofaces: int = 500_000,
    max_component_faces: int = 200_000,
    use_existing_witness: bool = True,
) -> RelativeLocalityCertificate:
    """Localize new cells for a complete PPG death relation, not a single edge.

    Bounds are inclusive *doubled-grid* coordinates. Old corrections remain
    globally unrestricted in K_{<t}; only new (q+1)-cells are box-restricted.
    """
    image = np.asarray(image)
    if tuple(image.shape) != tuple(witness.image_shape):
        raise ValueError('Witness image shape does not match')
    if witness.event_value is None or not np.isfinite(witness.event_value):
        raise ValueError('Expected a finite death-event threshold')
    grid_shape = tuple(2*int(n)+1 for n in image.shape)
    low = tuple(map(int, lower))
    high = tuple(map(int, upper))
    if (len(low) != image.ndim or len(high) != image.ndim
            or any(a < 0 or b < a or b >= g
                   for a, b, g in zip(low, high, grid_shape))):
        raise ValueError('Box must lie inside the doubled grid')
    # Verified one-sided positive certificate: an already constructed exact
    # filling is a feasible relative solution whenever all its event-level
    # (q+1)-cells lie in the proposed box. If they do not, the region might
    # STILL support another filling, so continue to the complete oracle.
    if use_existing_witness and hasattr(witness, 'filling_grid_indices'):
        filling = np.asarray(witness.filling_grid_indices, dtype=np.int64)
        relation = np.asarray(witness.relation_support, dtype=np.int64)
        if not np.array_equal(boundary_old_chain(filling, grid_shape), relation):
            raise ValueError('Existing witness fails the exact boundary check')
        if any(_original_cell_birth(image,int(c),grid_shape) >= witness.event_value
               for c in relation):
            raise ValueError('Witness relation is not an old cycle')
        old_part, new_part = [], []
        for c in filling:
            cell = int(c)
            if not 0 <= cell < prod(grid_shape):
                raise ValueError('Witness contains a cell outside the doubled grid')
            coords = tuple(int(x) for x in np.unravel_index(cell,grid_shape))
            if sum(x&1 for x in coords) != witness.dimension+1:
                raise ValueError('Existing witness has incorrect cell dimension')
            birth = _original_cell_birth(image,cell,grid_shape)
            if birth > witness.event_value:
                raise ValueError('Existing witness exceeds the event filtration')
            if birth < witness.event_value:
                old_part.append(cell)
            else:
                new_part.append((cell,coords))
        if all(all(a<=x<=b for a,x,b in zip(low,coords,high))
               for cell,coords in new_part):
            return RelativeLocalityCertificate(
                True, witness.dimension, witness.event_value, grid_shape,
                relation, len(old_part), len(new_part),
                np.asarray(old_part,dtype=np.int64),
                np.asarray([cell for cell,coords in new_part],dtype=np.int64),
                np.empty(0,dtype=np.int64),
                -1, 0, 0, 0, 'witness')
    if prod(b-a+1 for a,b in zip(low,high)) > max_examined_cells:
        raise MemoryError('Box exceeds max_examined_cells')
    new = []
    for coords in product(*(range(a,b+1) for a,b in zip(low,high))):
        if sum(c & 1 for c in coords) != witness.dimension+1:
            continue
        cell = int(np.ravel_multi_index(coords, grid_shape))
        if _original_cell_birth(image, cell, grid_shape) == witness.event_value:
            new.append(cell)
            if len(new)>max_new_cells:
                raise MemoryError('New candidate count exceeds max_new_cells')
    return certify_relative_filling(
        image, witness.relation_support, new, dimension=witness.dimension,
        event_value=witness.event_value, max_grid_cells=max_grid_cells,
        max_old_cells=max_old_cells, max_new_cells=max_new_cells,
        max_boundary_rows=max_boundary_rows, max_xor_steps=max_xor_steps,
        old_solver=old_solver, max_examined_cofaces=max_examined_cofaces,
        max_component_faces=max_component_faces)
