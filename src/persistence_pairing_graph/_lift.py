"""Filtered cycle lifting through equal-value cubical chain cancellations.

The streaming compressor records canceled *death* generators (with dimension and
filtration level). For a compressed q-cycle z', start with its surviving cells
in the original grid and solve D_pair x = boundary(seed), where D_pair contains
original boundaries of canceled q-dimensional deaths at values <= birth(z').
The resulting seed + x is an original, filtration-respecting q-cycle. This is
a cycle lift, NOT yet a complete explicit chain contraction (i, p, h).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from struct import Struct
import numpy as np

_PAIR_STRUCT = Struct('<IBxH')
_PAIR_DTYPE = np.dtype([('death', '<u4'), ('dimension', 'u1'), ('reserved', 'u1'), ('value', '<u2')])


def boundary_old_chain(ids, grid_shape: tuple[int, ...]) -> np.ndarray:
    """Original doubled-grid cubical boundary over F2, with no reduction data."""
    strides = tuple(int(np.prod(grid_shape[i + 1:])) for i in range(len(grid_shape)))
    result: set[int] = set()
    for old in ids:
        old = int(old)
        for extent, stride in zip(grid_shape, strides):
            if ((old // stride) % extent) & 1:
                for face in (old - stride, old + stride):
                    if face in result:
                        result.remove(face)
                    else:
                        result.add(face)
    return np.fromiter(sorted(result), dtype=np.int64)


@dataclass(slots=True)
class LiftedCycleRepresentative:
    """An original-grid cycle lifted from a compressed persistence birth.

    ``compressed_cells`` are canonical compressed-generator indices.
    ``original_grid_indices`` are actual cells forming the F2 cycle in the
    original doubled cubical grid, not just generator provenance.
    """
    dimension: int
    birth_cell: int
    death_cell: int | None
    birth_value: int | float
    death_value: int | float
    compressed_cells: np.ndarray
    original_grid_indices: np.ndarray
    doubled_grid_coordinates: np.ndarray

    @property
    def cell_centers(self) -> np.ndarray:
        return self.doubled_grid_coordinates.astype(np.float64) * 0.5


class StreamingLiftData:
    """Optional disk-backed record of canceled death cells and scalar levels.

    The data are appended in filtration order and take eight bytes per canceled
    pair on disk. Call ``close()`` when no longer needed.
    """
    def __init__(self, grid_shape: tuple[int, ...], temp_dir=None):
        self.grid_shape = grid_shape
        self.scratch = TemporaryDirectory(prefix='ppg_chain_lift_', dir=temp_dir)
        self.pairs_path = Path(self.scratch.name) / 'pairs.bin'
        self._output = self.pairs_path.open('wb', buffering=1024 * 1024)
        self.n_pairs = 0
        self.pairs = None

    def add_pair(self, death_old: int, dimension: int, filtration_value: int):
        self._output.write(_PAIR_STRUCT.pack(death_old, dimension, filtration_value))
        self.n_pairs += 1

    def finish(self):
        if self._output is not None:
            self._output.close()
            self._output = None
            if self.n_pairs:
                self.pairs = np.memmap(self.pairs_path, mode='r', dtype=_PAIR_DTYPE)
        return self

    def iter_pairs(self):
        if self._output is not None:
            raise RuntimeError('Call finish() before iterating canceled pairs')
        if self.pairs is not None:
            yield from self.pairs

    def close(self):
        if self._output is not None:
            self._output.close()
            self._output = None
        if isinstance(self.pairs, np.memmap):
            self.pairs._mmap.close()
        self.pairs = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _BoundarySolver:
    """Incrementally reduced matrix of canceled q-cell boundaries over F2."""
    def __init__(self, grid_shape, *, max_basis_columns: int, max_basis_entries: int):
        self.shape = grid_shape
        self.pivots: dict[int, tuple[set[int], set[int]]] = {}
        self.max_columns = max_basis_columns
        self.max_entries = max_basis_entries
        self.entries = 0

    def add(self, death_old: int):
        if len(self.pivots) >= self.max_columns:
            raise MemoryError('Cycle lifting exceeded max_basis_columns')
        column = set(boundary_old_chain((death_old,), self.shape))
        basis_chain = {death_old}
        while column:
            pivot = max(column)
            reducer = self.pivots.get(pivot)
            if reducer is None:
                count = len(column) + len(basis_chain)
                if self.entries + count > self.max_entries:
                    raise MemoryError('Cycle lifting exceeded max_basis_entries')
                self.pivots[pivot] = (column, basis_chain)
                self.entries += count
                return
            column.symmetric_difference_update(reducer[0])
            basis_chain.symmetric_difference_update(reducer[1])
        raise RuntimeError('Dependent canceled-death columns: contraction invariant violated')

    def solve_boundary(self, boundary_ids, *, max_chain_terms: int) -> np.ndarray:
        """Solve ∂x=boundary_ids in the stored canceled-death column span.

        The pivot table is indexed by original boundary cell IDs, not a dense
        bitset with one bit per original-grid cell.  This supports selective
        corrections in large, sparse original grids subject to explicit limits.
        """
        correction: set[int] = set()
        remaining = set(map(int, boundary_ids))
        while remaining:
            pivot = max(remaining)
            reducer = self.pivots.get(pivot)
            if reducer is None:
                raise RuntimeError(
                    'Boundary is not in the recorded canceled-death span '
                    f'(unresolved original boundary pivot {pivot})')
            remaining.symmetric_difference_update(reducer[0])
            correction.symmetric_difference_update(reducer[1])
            if len(correction) > max_chain_terms:
                raise MemoryError('Boundary correction exceeded max_chain_terms')
        result = np.fromiter(sorted(correction), dtype=np.int64)
        if not np.array_equal(boundary_old_chain(result, self.shape),
                              np.asarray(sorted(set(map(int, boundary_ids))), dtype=np.int64)):
            raise RuntimeError('Canceled-death correction failed its boundary equation')
        return result

    def lift(self, seed_old_ids, *, max_chain_terms: int) -> np.ndarray:
        seed = set(map(int, seed_old_ids))
        correction = self.solve_boundary(boundary_old_chain(seed, self.shape),
                                         max_chain_terms=max_chain_terms)
        seed.symmetric_difference_update(map(int, correction))
        if len(seed) > max_chain_terms:
            raise MemoryError('Cycle lifting exceeded max_chain_terms')
        result = np.fromiter(sorted(seed), dtype=np.int64)
        if len(boundary_old_chain(result, self.shape)):
            raise RuntimeError('Lifted original-grid representative is not a cycle')
        return result


def _compressed_v_column(reduction, birth: int, *, max_work: int):
    """Reconstruct V_birth from the full canonical operation log, iteratively."""
    memo: dict[int, set[int]] = {}
    stack = [(birth, False)]
    work = 0
    while stack:
        j, resolved = stack.pop()
        if j in memo:
            continue
        operations = reduction.operations.get(j, ())
        if not resolved:
            stack.append((j, True))
            for source in operations:
                if source >= j:
                    raise RuntimeError('Nontriangular canonical reduction log')
                if source not in memo:
                    stack.append((int(source), False))
        else:
            result = {j}
            for source in operations:
                term = memo[int(source)]
                work += len(term)
                if work > max_work:
                    raise MemoryError('Compressed cycle expansion exceeds max_work')
                result.symmetric_difference_update(term)
            memo[j] = result
    return sorted(memo[birth])


def lift_compressed_cycles(compressed, reduction, nodes, *, dimensions=(1, 2),
                           max_features=10, min_persistence=0.0, node_indices=None,
                           max_basis_columns=200_000, max_basis_entries=5_000_000,
                           max_chain_terms=500_000, max_work=5_000_000):
    """Return verified original-grid cycles for compressed persistence births.

    Corrections are selected only from canceled death cells at filtration
    values <= feature birth, and only in the requested homological dimension.
    No full original-grid persistence matrix is constructed.
    """
    trace = compressed.lift_trace
    if trace is None:
        raise ValueError('Compute streaming compression with retain_chain_lifts=True')
    if max_features < 0 or min_persistence < 0:
        raise ValueError('max_features and min_persistence must be nonnegative')
    if any(limit <= 0 for limit in (max_basis_columns, max_basis_entries,
                                   max_chain_terms, max_work)):
        raise ValueError('All capacity limits must be positive')
    if node_indices is None:
        chosen = [(pos, node) for pos, node in enumerate(nodes)
                  if node.dimension in dimensions and node.persistence >= min_persistence]
        chosen = chosen[:max_features]
    else:
        # A PPG edge may reference nodes far beyond the first ``max_features``.
        # Lift precisely its selected endpoints, preserving caller order.
        indices = [int(index) for index in node_indices]
        if len(indices) != len(set(indices)):
            raise ValueError('node_indices must be distinct')
        if len(indices) > max_features:
            raise ValueError('Selected node count exceeds max_features')
        if any(index < 0 or index >= len(nodes) for index in indices):
            raise IndexError('Selected persistence node index is out of range')
        chosen = [(index, nodes[index]) for index in indices]
        if any(node.dimension not in dimensions or node.persistence < min_persistence
               for _, node in chosen):
            raise ValueError('Selected node fails dimensions or min_persistence filter')
    if not chosen:
        return []
    if compressed.original_grid_indices is None:
        raise ValueError('Streaming chain lifting requires original-cell provenance')
    chosen_sorted = sorted(chosen, key=lambda z: (z[1].birth_value, z[1].dimension, z[0]))
    pairs = trace.pairs if trace.pairs is not None else ()
    # Stable value sorting; the compressor has already appended pairs in order.
    if any(int(pairs[i]['value']) > int(pairs[i+1]['value']) for i in range(len(pairs)-1)):
        raise RuntimeError('Contraction death pairs are not ordered by filtration value')
    solvers = {q: _BoundarySolver(trace.grid_shape, max_basis_columns=max_basis_columns,
                                  max_basis_entries=max_basis_entries)
               for q in set(node.dimension for _,node in chosen)}
    position = 0
    results = {}
    for index,node in chosen_sorted:
        t = node.birth_value
        while position < len(pairs) and int(pairs[position]['value']) <= t:
            pair = pairs[position]
            q = int(pair['dimension'])
            if q in solvers:
                solvers[q].add(int(pair['death']))
            position += 1
        compressed_cycle = _compressed_v_column(reduction, node.birth_cell, max_work=max_work)
        if any(int(compressed.dimensions[x]) != node.dimension for x in compressed_cycle):
            raise RuntimeError('Compressed representative has inconsistent chain dimensions')
        old_ids = compressed.original_grid_indices[compressed_cycle]
        lifted = solvers[node.dimension].lift(old_ids, max_chain_terms=max_chain_terms)
        coords = np.column_stack(np.unravel_index(lifted, trace.grid_shape))
        results[index] = LiftedCycleRepresentative(node.dimension, node.birth_cell,
                    node.death_cell, node.birth_value, node.death_value,
                    np.asarray(compressed_cycle, dtype=np.int64), lifted, coords)
    return [results[i] for i,_ in chosen]


def representative_cycles_streaming(image: np.ndarray, *, dimensions=(1, 2),
                                    max_features=10, min_persistence=0.0,
                                    max_cells=100_000_000, temp_dir=None,
                                    order_format='u32', order_builder='auto',
                                    accelerator='python', metadata_storage='memory',
                                    reduction_storage='packed',
                                    max_basis_columns=200_000,
                                    max_basis_entries=5_000_000,
                                    max_chain_terms=500_000,
                                    max_work=5_000_000):
    """Recover actual original-grid cycles from streaming-compressed persistence.

    Optional disk-backed cancellation records are created only for this call.
    Full canonical V operation logs are required. The routine automatically
    releases all reduction and trace scratch files, even on exceptions.

    The lifting step solves original cubical boundary equations using only
    equal-value canceled-death columns. It does not require materializing or
    reducing the complete original cubical complex. Memory caps bound sparse
    solves; some highly complex inputs may exceed them.
    """
    from ._streaming import compress_discrete_tensor_streaming
    from ._reduction import reduce_boundary
    from ._graph import build_nodes
    compressed = compress_discrete_tensor_streaming(
        np.asarray(image), max_cells=max_cells, temp_dir=temp_dir,
        retain_chain_lifts=True, retain_cell_provenance=True,
        order_format=order_format, order_builder=order_builder,
        accelerator=accelerator, metadata_storage=metadata_storage)
    try:
        reduction = reduce_boundary(compressed, storage=reduction_storage,
                                    log_mode='full', accelerator=accelerator,
                                    temp_dir=temp_dir)
        try:
            nodes = build_nodes(compressed.values, compressed.dimensions,
                                reduction, storage='compact')
            return lift_compressed_cycles(
                compressed, reduction, nodes, dimensions=dimensions,
                max_features=max_features, min_persistence=min_persistence,
                max_basis_columns=max_basis_columns,
                max_basis_entries=max_basis_entries,
                max_chain_terms=max_chain_terms, max_work=max_work)
        finally:
            reduction.close()
    finally:
        compressed.lift_trace.close()
