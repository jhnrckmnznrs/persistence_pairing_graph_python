"""Selective forward projection through recorded filtered elementary cancellations.

Companion to SelectiveInclusionReplay: this evaluates the chain projection
P=p_{m-1}...p_0 from original cubical cell coordinates into the final
compressed generators. For a cancellation d(b)=a+r, p(a)=r, p(b)=0,
and p fixes the remaining current generators (over GF(2)).

No intermediate boundary matrices or dense projection operator are formed.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable

import numpy as np

from ._reverse_replay import SelectiveInclusionReplay


@dataclass(frozen=True)
class ProjectionStatistics:
    total_cancellations: int
    eligible_cancellations: int
    processed_cancellations: int
    tail_terms_read: int
    input_cells: int
    output_cells: int


class SelectiveProjectionReplay(SelectiveInclusionReplay):
    """Evaluate the exact projection on selected original-grid chains.

    If stop_before_value is supplied, only pairs with scalar value strictly
    less than that threshold are canceled. This is the projection of the
    filtration's pre-event chain complex K_{<t}; the output can still contain
    generators canceled at time t or later.

    Work is bounded by the actually touched cancellation records and the
    lengths of tails read, provided the owner index already exists.
    """

    def __init__(self, trace, **kwargs):
        super().__init__(trace, **kwargs)
        self.last_projection_stats: ProjectionStatistics | None = None
        self.last_activated_cancellations: tuple[int, ...] | None = None

    def project(
        self, original_grid_cells: Iterable[int], *,
        stop_before_value: float | int | None = None,
        max_output_cells: int = 500_000,
        max_processed_records: int = 5_000_000,
        max_tail_terms: int = 5_000_000,
        record_activations: bool = False,
    ) -> np.ndarray:
        if min(max_output_cells,max_processed_records,max_tail_terms) <= 0:
            raise ValueError('Projection resource limits must be positive')
        if stop_before_value is None:
            cutoff = self.n
        else:
            if not np.isfinite(stop_before_value):
                raise ValueError('stop_before_value must be finite')
            if self.trace.pairs is None:
                raise RuntimeError('Pair filtration values are not available')
            values = self.trace.pairs['value']
            cutoff = int(np.searchsorted(values,stop_before_value,side='left'))
        from math import prod
        total = prod(self.grid_shape)
        chain: set[int] = set()
        for value in original_grid_cells:
            cell = int(value)
            if cell < 0 or cell >= total:
                raise IndexError('Original grid cell is out of bounds')
            if cell in chain: chain.remove(cell)
            else: chain.add(cell)
        input_size = len(chain)
        if len(chain) > max_output_cells:
            raise MemoryError('Projection exceeded max_output_cells')
        heap = []
        scheduled = set()
        def schedule(cell: int, *, after: int = -1):
            owner = self._owner(cell)
            if owner is None or owner >= cutoff:
                return
            if owner <= after:
                raise RuntimeError('Nontriangular cancellation projection tail')
            if owner not in scheduled:
                scheduled.add(owner)
                heappush(heap, owner)
        for cell in chain:
            schedule(cell)
        processed = 0
        tails = 0
        self.last_activated_cancellations = None
        activations: list[int] | None = [] if record_activations else None
        while heap:
            j = heappop(heap)
            a = int(self.records[j]['birth'])
            b = int(self.records[j]['death'])
            if a not in chain and b not in chain:
                continue
            processed += 1
            if processed > max_processed_records:
                raise MemoryError('Projection exceeded max_processed_records')
            if a in chain:
                if activations is not None:
                    activations.append(j)
                chain.remove(a)
                record = self.records[j]
                start = int(record['offset'])
                length = int(record['length'])
                tails += length
                if tails > max_tail_terms:
                    raise MemoryError('Projection exceeded max_tail_terms')
                for value in self.tails[start:start+length]:
                    term = int(value)
                    if term in chain: chain.remove(term)
                    else: chain.add(term)
                    schedule(term, after=j)
            chain.discard(b)
            if len(chain) > max_output_cells:
                raise MemoryError('Projection exceeded max_output_cells')
        if any((owner := self._owner(cell)) is not None and owner < cutoff
               for cell in chain):
            raise RuntimeError('Projection left a canceled generator')
        result = np.asarray(sorted(chain),dtype=np.int64)
        if activations is not None:
            self.last_activated_cancellations = tuple(activations)
        self.last_projection_stats = ProjectionStatistics(
            self.n,cutoff,processed,tails,input_size,len(result))
        return result
