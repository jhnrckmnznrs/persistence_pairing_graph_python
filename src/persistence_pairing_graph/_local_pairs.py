"""Selective incidence-component solver for filtered canceled-death corrections.

Only candidate canceled deaths whose original boundaries lie in incidence
components touched by the requested residual are used. This is an exact
restriction of the GF(2) boundary matrix, not a geometric approximation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import prod

import numpy as np

from ._lift import _BoundarySolver
from ._transition_witness import _boundary_faces


@dataclass(frozen=True, slots=True)
class ComponentSolveStats:
    eligible_deaths: int
    visited_faces: int
    selected_deaths: int


def solve_component_correction(trace, boundary_ids, *, dimension: int, level: int,
                               max_basis_columns: int, max_basis_entries: int,
                               max_chain_terms: int, max_component_faces: int = 2_000_000,
                               ) -> tuple[np.ndarray, ComponentSolveStats]:
    """Solve a canceled-death correction using only necessary incidence components.

    The original cubical boundary matrix is bipartite between q-faces and
    canceled (q+1)-death columns. Connected components of this incidence graph
    give independent GF(2) boundary subproblems. Starting at the residual's
    q-faces reaches *all* eligible death columns in the relevant components.
    No dense array indexed by every original-grid cell is constructed.

    This can still visit every eligible cancellation in a connected image;
    both workspace size and the elimination basis are explicitly bounded.
    """
    if min(max_basis_columns, max_basis_entries, max_chain_terms,
           max_component_faces) <= 0:
        raise ValueError('Component and basis limits must be positive')
    shape = tuple(int(x) for x in trace.grid_shape)
    if dimension < 0 or dimension >= len(shape):
        raise ValueError('Invalid boundary dimension')
    residual = set(int(i) for i in boundary_ids)
    if any(i < 0 or i >= prod(shape) for i in residual):
        raise IndexError('Original-grid boundary cell is outside the image')
    if not residual:
        return np.empty(0, dtype=np.int64), ComponentSolveStats(0, 0, 0)
    pairs = trace.pairs
    if pairs is None:
        raise RuntimeError('No canceled death pairs are available for correction')
    mask = ((pairs['dimension'] == dimension + 1) & (pairs['value'] <= level))
    eligible = np.sort(np.asarray(pairs['death'][mask], dtype=np.uint32))
    if not len(eligible):
        raise RuntimeError('No canceled death columns exist at this filtration level')
    if len(eligible) > 1 and np.any(eligible[1:] == eligible[:-1]):
        raise RuntimeError('Canceled death columns must be distinct')
    strides = tuple(prod(shape[i+1:]) for i in range(len(shape)))
    pending = deque(sorted(residual))
    seen_faces = set(residual)
    chosen: set[int] = set()
    if len(seen_faces) > max_component_faces:
        raise MemoryError('Incidence component exceeded max_component_faces')
    while pending:
        face = pending.popleft()
        for extent, stride in zip(shape, strides):
            coord = (face // stride) % extent
            # A (q+1)-coface is formed by changing exactly one even
            # coordinate into its adjacent odd coordinate.
            if coord & 1:
                continue
            for step in (-1, 1):
                if not 0 <= coord + step < extent:
                    continue
                death = face + step * stride
                position = int(np.searchsorted(eligible, death))
                if (position == len(eligible) or int(eligible[position]) != death
                        or death in chosen):
                    continue
                if len(chosen) >= max_basis_columns:
                    raise MemoryError('Incidence component exceeded max_basis_columns')
                chosen.add(death)
                for other in _boundary_faces(death, shape, strides):
                    if other not in seen_faces:
                        seen_faces.add(other)
                        if len(seen_faces) > max_component_faces:
                            raise MemoryError('Incidence component exceeded max_component_faces')
                        pending.append(other)
    solver = _BoundarySolver(shape, max_basis_columns=max_basis_columns,
                             max_basis_entries=max_basis_entries)
    for death in sorted(chosen):
        solver.add(death)
    correction = solver.solve_boundary(sorted(residual), max_chain_terms=max_chain_terms)
    return correction, ComponentSolveStats(len(eligible), len(seen_faces), len(chosen))
