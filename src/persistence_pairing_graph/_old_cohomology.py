"""Sparse old-level cohomology coordinates and selected boundary preimages.

This is a chain-level, reduction-reusing oracle for K_{<t}; it does not
rebuild the old cubical boundary matrix. The canonical quotient normal form
uses reduced old (q+1)-boundary columns from the existing PPG reduction.

Coordinates live in C_q(K_{<t}) / B_q(K_{<t}), a redundant cocycle
coordinate system: on cycles they determine homology exactly. They need
not be minimal coordinates in a selected basis of H_q.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

import numpy as np

from ._filling_locality import _dimension, _faces, _support
from ._lift import _compressed_v_column, boundary_old_chain
from ._local_pairs import solve_component_correction
from ._projection_replay import SelectiveProjectionReplay
from ._transition_witness import _original_cell_birth


@dataclass(frozen=True)
class OldLevelCoordinates:
    dimension: int
    event_value: int | float
    coordinates: tuple[int, ...]
    compressed_cycle_cells: int
    pivot_columns_used: tuple[int, ...]
    reduced_column_entries: int

    @property
    def is_boundary(self) -> bool:
        return not self.coordinates


@dataclass(frozen=True)
class SparseRelativeCertificate:
    """Exact decision for a bounded set of new (q+1) cells.

    A negative result supplies a *coordinate-functional* obstruction:
    the sum of the indicated quotient coordinates equals one on c and
    vanishes on every permitted relative birth generator. It is NOT
    materialized as an ambient original-grid cochain.
    """
    feasible: bool
    dimension: int
    event_value: int | float
    relation_cells: np.ndarray
    old_filling_cells: np.ndarray
    new_filling_cells: np.ndarray
    obstruction_coordinates: tuple[int, ...]
    relative_kernel_generators: int
    coordinate_rank: int
    projection_queries: int
    old_pivot_columns_used: int
    old_filling_reconstructed: bool = False


class SparseOldLevelOracle:
    """Query the already reduced PPG complex without old-cell enumeration.

    Input is an active PPGAnalysis context with retain_inclusion_trace=True
    and log_mode='full' (as in PPGAnalysis). All computations are over GF(2).
    Expensive trace indexing and pivot indexing are shared across queries.
    """

    def __init__(self, analysis, *,
                 max_coordinate_terms: int = 500_000,
                 max_reduced_entries: int = 5_000_000,
                 max_preimage_work: int = 5_000_000,
                 max_output_cells: int = 500_000,
                 max_component_faces: int = 2_000_000):
        if analysis.closed or not analysis.retain_inclusion_trace:
            raise ValueError('An open PPGAnalysis with retain_inclusion_trace=True is required')
        if min(max_coordinate_terms, max_reduced_entries, max_preimage_work,
               max_output_cells, max_component_faces) <= 0:
            raise ValueError('All oracle resource limits must be positive')
        self.analysis = analysis
        self.image = analysis.image
        self.compressed = analysis.compressed
        self.reduction = analysis.reduction
        self.grid_shape = analysis.grid_shape
        self.total_cells = prod(self.grid_shape)
        self.max_coordinate_terms = max_coordinate_terms
        self.max_reduced_entries = max_reduced_entries
        self.max_preimage_work = max_preimage_work
        self.max_output_cells = max_output_cells
        self.max_component_faces = max_component_faces
        trace = self.compressed.lift_trace
        self.projector = SelectiveProjectionReplay(trace)
        # Forward and reverse replay share the same canceled-generator owner
        # index and sparse influence cache rather than allocating them twice.
        self.inclusion = self.projector
        n = len(self.compressed)
        self.pivot_to_death = np.full(n, -1, dtype=np.int32)
        for j, pivot in enumerate(self.reduction.low):
            if pivot >= 0:
                if self.pivot_to_death[int(pivot)] >= 0:
                    raise RuntimeError('Two reduced columns have the same pivot')
                self.pivot_to_death[int(pivot)] = j
        # Index only compressed survivor cells. A sorted uint32/uint64
        # numeric table avoids one Python dictionary entry per generator.
        ids = np.asarray(self.compressed.original_grid_indices)
        self._sorted_order = np.argsort(ids, kind='stable')
        self._sorted_grid_ids = ids[self._sorted_order]
        self.projection_queries = 0

    def _validated_old_cycle(self, cycle: Iterable[int], dimension: int,
                             t: int | float) -> list[int]:
        if self.analysis.closed:
            raise RuntimeError('PPGAnalysis is closed')
        if not 0 <= dimension < self.image.ndim or not np.isfinite(t):
            raise ValueError('Invalid dimension or event value')
        chain = _support(cycle, self.total_cells, toggle=True)
        strides = tuple(prod(self.grid_shape[i+1:])
                        for i in range(len(self.grid_shape)))
        for cell in chain:
            if _dimension(cell, self.grid_shape, strides) != dimension:
                raise ValueError('Query has a cell of the wrong homology dimension')
            if _original_cell_birth(self.image,cell,self.grid_shape) >= t:
                raise ValueError('Query must lie strictly before event_value')
        if len(boundary_old_chain(chain, self.grid_shape)):
            raise ValueError('Query is not an old cellular cycle')
        return chain

    def _validated_old_chain(self, chain: Iterable[int], dimension: int,
                             t: int | float) -> list[int]:
        """Validate an arbitrary old q-chain, not necessarily a cycle."""
        if self.analysis.closed:
            raise RuntimeError('PPGAnalysis is closed')
        if not 0 <= dimension < self.image.ndim or not np.isfinite(t):
            raise ValueError('Invalid dimension or event value')
        support = _support(chain, self.total_cells, toggle=True)
        strides = tuple(prod(self.grid_shape[i+1:])
                        for i in range(len(self.grid_shape)))
        for cell in support:
            if _dimension(cell, self.grid_shape, strides) != dimension:
                raise ValueError('Query has a cell of the wrong homology dimension')
            if _original_cell_birth(self.image,cell,self.grid_shape) >= t:
                raise ValueError('Query must lie strictly before event_value')
        return support

    def evaluate_cocycle(self, cocycle_coordinates: Iterable[int],
                         old_chain: Iterable[int], *,
                         dimension: int, event_value: int | float) -> int:
        """Evaluate a lazily represented old q-cocycle on any old q-chain.

        The cocycle is a linear functional of canonical quotient normal-form
        coordinates, hence it annihilates *all* old (q+1)-boundaries, even
        when the input q-chain is not itself a cycle.
        """
        support = self._validated_old_chain(old_chain,dimension,event_value)
        projected=self.projector.project(
            support,max_output_cells=self.max_output_cells)
        self.projection_queries += 1
        compact=[self._compressed_id_from_old(v) for v in projected]
        normal, _, _ = self._normal_form(compact,dimension,event_value)
        chosen = set(map(int,cocycle_coordinates))
        if any(v < 0 or v >= len(self.compressed) for v in chosen):
            raise ValueError('Invalid compressed cocycle coordinate index')
        return len(chosen.intersection(normal)) & 1

    def _normal_form(self, projected_cells: Iterable[int],
                     dimension: int, t: int | float) -> tuple[tuple[int, ...],
                                                           tuple[int, ...], int]:
        remaining = set(map(int, projected_cells))
        free: set[int] = set()
        selected: set[int] = set()
        entries = 0
        while remaining:
            pivot = max(remaining)
            death = int(self.pivot_to_death[pivot])
            if death < 0 or self.compressed.values[death] >= t:
                remaining.remove(pivot)
                free.add(pivot)
                continue
            column = self.reduction.reduced_columns[death]
            if not column or int(column[-1]) != pivot:
                raise RuntimeError('Inconsistent reduced-column pivot')
            entries += len(column)
            if entries > self.max_reduced_entries:
                raise MemoryError('Coordinate reduction exceeds max_reduced_entries')
            remaining.symmetric_difference_update(map(int,column))
            if death in selected: selected.remove(death)
            else: selected.add(death)
            if len(remaining)+len(free)>self.max_coordinate_terms:
                raise MemoryError('Coordinate query exceeds max_coordinate_terms')
        return tuple(sorted(free)), tuple(sorted(selected)), entries

    def coordinates(self, cycle: Iterable[int], *, dimension: int,
                    event_value: int | float) -> OldLevelCoordinates:
        chain = self._validated_old_cycle(cycle, dimension, event_value)
        projected = self.projector.project(
            chain,max_output_cells=self.max_output_cells)
        self.projection_queries += 1
        if any(int(self.compressed.dimensions[j]) != dimension
               or self.compressed.values[j] >= event_value for j in
               (self._compressed_id_from_old(x) for x in projected)):
            raise RuntimeError('Projection violated homological grading or filtration')
        compact = [self._compressed_id_from_old(x) for x in projected]
        normal, deaths, work = self._normal_form(compact,dimension,event_value)
        return OldLevelCoordinates(
            dimension,event_value,normal,len(compact),deaths,work)

    def _compressed_id_from_old(self, old: int) -> int:
        # The original IDs are not in numerical order after compression.
        # Binary search a packed sorted index rather than a Python hash map.
        position = int(np.searchsorted(self._sorted_grid_ids, int(old)))
        if position >= len(self._sorted_grid_ids) or (
            int(self._sorted_grid_ids[position]) != int(old)):
            raise RuntimeError('Projection produced a canceled generator')
        return int(self._sorted_order[position])

    def boundary_preimage(self, cycle: Iterable[int], *, dimension: int,
                          event_value: int | float,
                          correction_method: str = 'homotopy') -> np.ndarray:
        """Find an exact original-grid OLD (q+1)-chain with boundary = cycle.

        May require a large filling even when the coordinate query is small.
        Uses compressed canonical V columns, exact reverse inclusion, and
        an old-level canceled-death incidence correction.
        """
        chain = self._validated_old_cycle(cycle,dimension,event_value)
        if correction_method not in ('homotopy','incidence'):
            raise ValueError("correction_method must be 'homotopy' or 'incidence'")
        coords = self.coordinates(chain,dimension=dimension,event_value=event_value)
        if coords.coordinates:
            raise ValueError('The selected old cycle does not bound before the event')
        compressed_fill: set[int] = set()
        for j in coords.pivot_columns_used:
            compressed_fill.symmetric_difference_update(
                _compressed_v_column(self.reduction,j,max_work=self.max_preimage_work))
            if len(compressed_fill)>self.max_output_cells:
                raise MemoryError('Compressed preimage exceeds max_output_cells')
        if compressed_fill:
            original_seed = self.inclusion.lift(
                self.compressed.original_grid_indices[sorted(compressed_fill)],
                max_output_cells=self.max_output_cells)
        else:
            original_seed = np.empty(0,dtype=np.int64)
        if correction_method == 'homotopy':
            # Filtered contraction identity on an old cycle x:
            #   x + i p(x) = d H(x).
            # During forward projection, record each activated birth a_j;
            # H(x) is the XOR of the corresponding canceled death b_j,
            # each lifted through strictly earlier inclusions.
            self.projector.project(
                chain,stop_before_value=event_value,
                max_output_cells=self.max_output_cells,
                record_activations=True)
            activated=self.projector.last_activated_cancellations
            if activated is None:
                raise RuntimeError('Missing selective homotopy activation trace')
            homotopy: set[int] = set()
            for j in activated:
                partial=self.inclusion.lift_canceled_death(
                    j,max_output_cells=self.max_output_cells)
                homotopy.symmetric_difference_update(map(int,partial))
                if len(homotopy)>self.max_output_cells:
                    raise MemoryError('Selective homotopy exceeds max_output_cells')
            correction=np.asarray(sorted(homotopy),dtype=np.int64)
        else:
            # Independent exact reference based on an incidence closure.
            discrepancy = set(chain)
            discrepancy.symmetric_difference_update(
                map(int,boundary_old_chain(original_seed,self.grid_shape)))
            if discrepancy:
                if int(event_value) != event_value:
                    raise ValueError('Expected integral event_value')
                correction, _ = solve_component_correction(
                    self.compressed.lift_trace, sorted(discrepancy),
                    dimension=dimension, level=int(event_value)-1,
                    max_basis_columns=self.analysis.max_basis_columns,
                    max_basis_entries=self.analysis.max_basis_entries,
                    max_chain_terms=self.max_output_cells,
                    max_component_faces=self.max_component_faces)
            else:
                correction = np.empty(0,dtype=np.int64)
        filled=set(map(int,original_seed))
        filled.symmetric_difference_update(map(int,correction))
        out=np.asarray(sorted(filled),dtype=np.int64)
        if len(out)>self.max_output_cells:
            raise MemoryError('Preimage exceeds max_output_cells')
        if not np.array_equal(boundary_old_chain(out,self.grid_shape),
                              np.asarray(chain,dtype=np.int64)):
            raise RuntimeError('Original-grid preimage boundary mismatch')
        if any(_original_cell_birth(self.image,int(k),self.grid_shape)>=event_value
               for k in out):
            raise RuntimeError('Boundary preimage leaves the old filtration sublevel')
        return out

    def restricted_relative(self, cycle: Iterable[int],
                            allowed_new_cells: Iterable[int], *,
                            dimension: int, event_value: int | float,
                            reconstruct: bool = True,
                            max_new_cells: int = 3_000,
                            max_relative_rows: int = 10_000,
                            max_relative_work: int = 2_000_000) -> SparseRelativeCertificate:
        """Exact relative feasibility; optional original-grid preimage.

        When reconstruct=False, a positive result returns the NEW relative
        chain only; the old correction is intentionally omitted. This makes
        repeated region searches much cheaper. New faces must cancel.
        """
        if min(max_new_cells,max_relative_rows,max_relative_work)<=0:
            raise ValueError('Relative resource limits must be positive')
        q=dimension; t=event_value
        old_cycle=self._validated_old_cycle(cycle,q,t)
        new=_support(allowed_new_cells,self.total_cells,toggle=False)
        if len(new)>max_new_cells:
            raise MemoryError('Too many permitted new cells')
        strides=tuple(prod(self.grid_shape[i+1:]) for i in range(len(self.grid_shape)))
        faces_by_cell=[]
        new_faces=set()
        for cell in new:
            if _dimension(cell,self.grid_shape,strides)!=q+1:
                raise ValueError('New filling cell has the wrong dimension')
            if _original_cell_birth(self.image,cell,self.grid_shape)!=t:
                raise ValueError('New filling cells must enter at exactly the event')
            faces=_faces(cell,self.grid_shape,strides)
            faces_by_cell.append(faces)
            for face in faces:
                if _original_cell_birth(self.image,face,self.grid_shape)==t:
                    new_faces.add(face)
        if len(new_faces)>max_relative_rows:
            raise MemoryError('Relative boundary exceeds max_relative_rows')
        row={face:i for i,face in enumerate(sorted(new_faces))}
        pivots: dict[int, tuple[int,int]]={}
        null_combos=[]
        operations=0
        for j,faces in enumerate(faces_by_cell):
            bits=0
            for face in faces:
                if face in row:
                    bits^=1<<row[face]
            combo=1<<j
            while bits:
                pivot=bits.bit_length()-1
                prior=pivots.get(pivot)
                if prior is None:
                    pivots[pivot]=(bits,combo)
                    break
                bits^=prior[0];combo^=prior[1];operations+=1
                if operations>max_relative_work:
                    raise MemoryError('Relative kernel exceeds max_relative_work')
            if bits==0:
                null_combos.append(combo)
        target=self.coordinates(old_cycle,dimension=q,event_value=t)
        pivot_coords:dict[int,tuple[int,int]]={}
        coordinate_rows=set(target.coordinates)
        kernel_coords=[]
        for k,combo in enumerate(null_combos):
            chain=[new[j] for j in range(len(new)) if (combo>>j)&1]
            boundary=boundary_old_chain(chain,self.grid_shape)
            if any(_original_cell_birth(self.image,int(face),self.grid_shape)>=t
                   for face in boundary):
                raise RuntimeError('Relative kernel generated a nonrelative chain')
            coord=self.coordinates(boundary,dimension=q,event_value=t)
            kernel_coords.append(coord.coordinates)
            coordinate_rows.update(coord.coordinates)
        rows=sorted(coordinate_rows)
        if len(rows)>max_relative_rows:
            raise MemoryError('Coordinate system exceeds max_relative_rows')
        row_index={v:i for i,v in enumerate(rows)}
        for k,coord in enumerate(kernel_coords):
            bits=sum(1<<row_index[v] for v in coord)
            combo=1<<k
            while bits:
                pivot=bits.bit_length()-1
                prior=pivot_coords.get(pivot)
                if prior is None:
                    pivot_coords[pivot]=(bits,combo)
                    break
                bits^=prior[0];combo^=prior[1];operations+=1
                if operations>max_relative_work:
                    raise MemoryError('Coordinate elimination exceeds max_relative_work')
        target_bits=sum(1<<row_index[v] for v in target.coordinates)
        remainder=target_bits; chosen=0
        for pivot in sorted(pivot_coords,reverse=True):
            if (remainder>>pivot)&1:
                remainder^=pivot_coords[pivot][0]
                chosen^=pivot_coords[pivot][1]
        if remainder:
            free=remainder.bit_length()-1
            functional=1<<free
            for pivot in sorted(pivot_coords):
                rest=pivot_coords[pivot][0]^(1<<pivot)
                if (rest&functional).bit_count()&1:
                    functional|=1<<pivot
            if (functional&target_bits).bit_count()%2!=1:
                raise RuntimeError('Invalid old cohomology obstruction')
            if any((functional & sum(1<<row_index[v] for v in coord)).bit_count()&1
                   for coord in kernel_coords):
                raise RuntimeError('Obstruction does not annihilate relative candidates')
            obstruct=tuple(v for i,v in enumerate(rows) if (functional>>i)&1)
            return SparseRelativeCertificate(
                False,q,t,np.asarray(old_cycle,dtype=np.int64),
                np.empty(0,dtype=np.int64),np.empty(0,dtype=np.int64),
                obstruct,len(null_combos),len(pivot_coords),
                self.projection_queries,len(target.pivot_columns_used),False)
        selected_new=0
        for k,combo in enumerate(null_combos):
            if (chosen>>k)&1: selected_new^=combo
        new_chain=np.asarray([new[j] for j in range(len(new))
                              if (selected_new>>j)&1],dtype=np.int64)
        delta=set(old_cycle)
        delta.symmetric_difference_update(
            map(int,boundary_old_chain(new_chain,self.grid_shape)))
        if reconstruct:
            old_fill=self.boundary_preimage(sorted(delta),dimension=q,event_value=t)
            if not np.array_equal(
                boundary_old_chain([*old_fill,*new_chain],self.grid_shape),
                np.asarray(old_cycle,dtype=np.int64)):
                raise RuntimeError('Reconstructed relative boundary is wrong')
        else:
            old_fill=np.empty(0,dtype=np.int64)
        return SparseRelativeCertificate(
            True,q,t,np.asarray(old_cycle,dtype=np.int64),
            old_fill,new_chain,(),len(null_combos),len(pivot_coords),
            self.projection_queries,len(target.pivot_columns_used),reconstruct)
