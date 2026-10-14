"""Selective, exact evaluation of the inclusion of an elementary filtered contraction.

Each record has a canceled birth a, canceled death b and the *current*, projected
boundary tail r such that d_j(b) = a + r.  Records are in cancellation order.
The code never constructs an intermediate differential or an inclusion matrix.
It does retain a temporary owner index for cells occurring in the trace.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable

import numpy as np

from ._lift import boundary_old_chain


@dataclass(frozen=True)
class ReplayStatistics:
    total_cancellations: int
    replayed_nonzero_steps: int
    queried_original_columns: int
    visited_record_instances: int
    tails_read: int
    influence_entries: int
    output_cells: int
    trace_record_bytes: int
    trace_tail_bytes: int


class SelectiveInclusionReplay:
    """Apply I=i_0...i_(m-1) to selected chains of final active generators.

    The selected chain is supplied as original IDs of compressed survivor cells.
    A reverse heap processes only nonzero cancellation indices.  Each original
    cell boundary is projected through its *preceding* cancellations lazily;
    the resulting influence sets are memoized across repeated chain queries.
    Limits bound sparse work, not the original doubled-grid size.
    """

    def __init__(self, trace, *, max_cached_columns=100_000,
                 max_visited_records=5_000_000, max_influence_entries=5_000_000,
                 owner_storage='auto'):
        if trace.replay_records is None and trace.n_pairs:
            raise ValueError('Streaming contraction requires retain_inclusion_trace=True')
        if not trace.retain_inclusion_trace:
            raise ValueError('Streaming contraction requires retain_inclusion_trace=True')
        self.trace = trace
        self.records = trace.replay_records
        self.tails = trace.replay_tails
        self.grid_shape = trace.grid_shape
        if any(x <= 0 for x in (max_cached_columns,max_visited_records,max_influence_entries)):
            raise ValueError('All selective replay limits must be positive')
        self.max_cached_columns = max_cached_columns
        self.max_visited_records = max_visited_records
        self.max_influence_entries = max_influence_entries
        self.n = trace.n_pairs
        if owner_storage not in ('auto','dict','packed','dense'):
            raise ValueError("owner_storage must be 'auto', 'dict', 'packed', or 'dense'")
        if owner_storage == 'auto':
            # An indexed uint32 grid is both smaller and faster for dense cancellations.
            from math import prod
            owner_storage = ('dense' if 4 * prod(self.grid_shape) <= 16 * self.n
                             else 'packed' if self.n >= 16_384 else 'dict')
        self.owner_storage = owner_storage
        self.owner = None
        self.owner_ids = None
        self.owner_steps = None
        self.owner_grid = None
        if owner_storage == 'dict':
            owner = {}
            for j in range(self.n):
                a = int(self.records[j]['birth'])
                b = int(self.records[j]['death'])
                if a in owner or b in owner or a == b:
                    raise RuntimeError('Cancellation generator was paired more than once')
                owner[a] = j
                owner[b] = j
            self.owner = owner
        elif owner_storage == 'dense':
            from math import prod
            # Store step+1 (zero denotes an uncanceled original cell). This
            # uses 4 bytes per original cubical cell, without sorting.
            self.owner_grid = np.zeros(prod(self.grid_shape), dtype=np.uint32)
            births = self.records['birth']
            deaths = self.records['death']
            ids = np.arange(1, self.n + 1, dtype=np.uint32)
            if np.any(births == deaths):
                raise RuntimeError('Cancellation generator was paired with itself')
            self.owner_grid[births] = ids
            self.owner_grid[deaths] = ids
            # Validate no generator is shared by different cancellation pairs.
            if (np.any(self.owner_grid[births] != ids) or
                    np.any(self.owner_grid[deaths] != ids)):
                raise RuntimeError('Cancellation generator was paired more than once')
        else:
            # At rest, the index takes 8 bytes for each canceled generator;
            # unlike a Python dict it stores no objects or pointers per cell.
            cell_ids = np.empty(2*self.n, dtype=np.uint32)
            cell_ids[::2] = self.records['birth']
            cell_ids[1::2] = self.records['death']
            order = np.argsort(cell_ids,kind='quicksort')
            self.owner_ids = cell_ids[order]
            if self.n and np.any(self.owner_ids[:-1] == self.owner_ids[1:]):
                raise RuntimeError('Cancellation generator was paired more than once')
            self.owner_steps = (order // 2).astype(np.uint32)

        self.cache: dict[int, frozenset[int]] = {}
        self.visited = 0
        self.tails_read = 0
        self.cached_entries = 0
        self.last_stats = None

    def _owner(self, cell: int):
        if self.owner_storage == 'dense':
            position = int(self.owner_grid[cell])
            return position - 1 if position else None
        if self.owner_storage == 'dict':
            return self.owner.get(cell)
        if self.n == 0:
            return None
        pos = int(np.searchsorted(self.owner_ids,cell))
        if pos < len(self.owner_ids) and int(self.owner_ids[pos]) == cell:
            return int(self.owner_steps[pos])
        return None

    @property
    def owner_index_bytes(self):
        if self.owner_storage == 'dense':
            return self.owner_grid.nbytes
        if self.owner_storage == 'packed':
            return self.owner_ids.nbytes + self.owner_steps.nbytes
        return None  # Python dictionary footprint is implementation-dependent.


    def _tail(self, j):
        record = self.records[j]
        start, length = int(record['offset']), int(record['length'])
        self.tails_read += length
        if not length:
            return ()
        return self.tails[start:start+length]

    def _influence(self, cell: int) -> frozenset[int]:
        cached = self.cache.get(cell)
        if cached is not None:
            return cached
        if len(self.cache) >= self.max_cached_columns:
            raise MemoryError('Selective replay exceeded max_cached_columns')
        end_owner = self._owner(cell)
        end = self.n if end_owner is None else end_owner
        chain = set(map(int, boundary_old_chain((cell,), self.grid_shape)))
        pending = []
        scheduled = set()

        def schedule(term):
            k = self._owner(int(term))
            if k is not None and k < end and k not in scheduled:
                scheduled.add(k)
                heappush(pending, k)

        for face in chain:
            schedule(face)
        influence = set()
        while pending:
            j = heappop(pending)
            self.visited += 1
            if self.visited > self.max_visited_records:
                raise MemoryError('Selective replay exceeded max_visited_records')
            record = self.records[j]
            a, b = int(record['birth']), int(record['death'])
            if a in chain:
                influence.add(j)
                chain.remove(a)
                tail = self._tail(j)
                for term in tail:
                    k = int(term)
                    if k in chain:
                        chain.remove(k)
                    else:
                        chain.add(k)
                    schedule(k)
            chain.discard(b)
        self.cached_entries += len(influence)
        if self.cached_entries > self.max_influence_entries:
            raise MemoryError('Selective replay exceeded max_influence_entries')
        result = frozenset(influence)
        self.cache[cell] = result
        return result

    def lift(self, surviving_original_cells: Iterable[int], *, max_output_cells=500_000):
        """Return original-grid support of I(chain), over F2.

        For general noncycles, verifies the filtered chain-map identity in the
        caller/test; the routine itself computes exact reverse substitution.
        """
        if max_output_cells <= 0:
            raise ValueError('max_output_cells must be positive')
        cells = set()
        for term in surviving_original_cells:
            k = int(term)
            if k in cells: cells.remove(k)
            else: cells.add(k)
        if any(self._owner(c) is not None for c in cells):
            raise ValueError('Input chain contains a canceled generator')
        active = set()
        pending = []

        def toggle(j):
            if j in active:
                active.remove(j)
            else:
                active.add(j)
                heappush(pending, -j)

        before_visited = self.visited
        before_tails = self.tails_read
        for cell in cells:
            for j in self._influence(cell):
                toggle(j)
        inserted = 0
        while pending:
            j = -heappop(pending)
            if j not in active:
                continue
            active.remove(j)
            b = int(self.records[j]['death'])
            if b in cells:
                raise RuntimeError('Reverse replay inserted a duplicate canceled death')
            cells.add(b)
            inserted += 1
            if len(cells) > max_output_cells:
                raise MemoryError('Selective replay exceeded max_output_cells')
            for k in self._influence(b):
                if k >= j:
                    raise RuntimeError('Invalid nontriangular inclusion dependency')
                toggle(k)
        result = np.asarray(sorted(cells), dtype=np.int64)
        self.last_stats = ReplayStatistics(
            total_cancellations=self.n, replayed_nonzero_steps=inserted,
            queried_original_columns=len(self.cache),
            visited_record_instances=self.visited-before_visited,
            tails_read=self.tails_read-before_tails,
            influence_entries=self.cached_entries, output_cells=len(result),
            trace_record_bytes=self.n*20, trace_tail_bytes=self.trace.replay_tail_entries*4)
        return result
