"""Sparse F2 persistence reduction and equal-value filtered chain contraction."""
from dataclasses import dataclass
from array import array
from collections.abc import Mapping
import heapq
from tempfile import TemporaryDirectory
from pathlib import Path
from ._disk import DiskColumns, DiskOperations

import numpy as np

from ._cubical import FilteredComplex


def xor_sorted(a: list[int], b: list[int]) -> list[int]:
    """Symmetric difference of sorted, unique integer lists."""
    # Python's hash-table symmetric difference is implemented in C and is
    # generally faster once enough indices are present. Both inputs are
    # sorted unique F2 supports; sorting the result restores canonical pivots.
    if len(a) + len(b) >= 24:
        return sorted(set(a).symmetric_difference(b))
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def _toggle(values: set[int], index: int) -> None:
    if index in values:
        values.remove(index)
    else:
        values.add(index)


def _project_contracted(chain: list[int], state: np.ndarray,
                        replacement: dict[int, list[int]]) -> list[int]:
    """Eliminate contracted births, processing indices from largest to smallest."""
    work = set(chain)
    heap = [-int(x) for x in work]
    heapq.heapify(heap)
    active = []
    while heap:
        index = -heapq.heappop(heap)
        if index not in work:
            continue
        work.remove(index)
        if state[index] == 0:
            active.append(index)
        elif state[index] == 1:
            for lower in replacement[index]:
                if lower >= index:
                    raise RuntimeError("Invalid nontriangular contraction relation")
                if lower in work:
                    work.remove(lower)
                else:
                    work.add(lower)
                    heapq.heappush(heap, -lower)
        # state == 2: a cancelled death, which maps to zero
    active.reverse()
    return active


@dataclass(slots=True)
class CompressedComplex:
    values: np.ndarray
    dimensions: np.ndarray
    boundaries: list[list[int]]
    original_cells: np.ndarray
    original_grid_indices: np.ndarray
    original_size: int

    def __len__(self) -> int:
        return len(self.values)

    def boundary(self, j: int) -> list[int]:
        return self.boundaries[j]


def compress_value_blocks(complex_: FilteredComplex) -> CompressedComplex:
    """Deterministically cancel all equal-filtration-value algebraic pairs.

    This is the materialized value-block algorithm from the Rust implementation,
    including the original filtration ordering and largest-index pivot rule.
    """
    n = len(complex_)
    state = np.zeros(n, dtype=np.uint8)  # active / cancelled-birth / cancelled-death
    replacement: dict[int, list[int]] = {}
    old_to_new = np.full(n, -1, dtype=np.int64)
    surviving_old: list[int] = []
    boundaries: list[list[int]] = []
    birth = complex_.values
    start = 0
    while start < n:
        end = start + 1
        while end < n and birth[end] == birth[start]:
            end += 1
        provisional: list[tuple[int, list[int]]] = []
        cancelled: list[int] = []
        for j in range(start, end):
            projected = _project_contracted(complex_.boundary(j), state, replacement)
            if projected and birth[projected[-1]] == birth[j]:
                pivot = projected[-1]
                if state[pivot] != 0 or complex_.dimensions[pivot] + 1 != complex_.dimensions[j]:
                    raise RuntimeError("Invalid equal-value algebraic cancellation")
                state[pivot] = 1
                replacement[pivot] = projected[:-1]
                cancelled.append(pivot)
                state[j] = 2
            else:
                provisional.append((j, projected))
        for pivot in sorted(cancelled):
            replacement[pivot] = _project_contracted(replacement[pivot], state, replacement)
        for j, projected in provisional:
            if state[j] != 0:
                continue
            if any(state[row] or birth[row] >= birth[j] for row in projected):
                raise RuntimeError("Compressed boundary does not strictly lower the filtration")
            boundaries.append([int(old_to_new[row]) for row in projected])
            if any(row < 0 for row in boundaries[-1]):
                raise RuntimeError("Missing compressed boundary index")
            old_to_new[j] = len(surviving_old)
            surviving_old.append(j)
        start = end
    surviving = np.asarray(surviving_old, dtype=np.int64)
    return CompressedComplex(
        values=complex_.values[surviving], dimensions=complex_.dimensions[surviving],
        boundaries=boundaries, original_cells=surviving,
        original_grid_indices=complex_.grid_indices[surviving], original_size=n,
    )


class PackedColumns(Mapping[int, list[int]]):
    """Reduced nonzero columns in one compact uint32 arena with O(1) lookup.

    The original death indices are mapped to slots with a compact int32 array;
    only negative columns have payloads or two offset entries.
    """

    def __init__(self, n: int):
        if n >= 2**31:
            raise ValueError('Packed reduction requires fewer than 2**31 generators')
        self.slot = np.full(n, -1, dtype=np.int32)
        self.offsets = array('I', [0])
        self.data = array('I')

    def append(self, death: int, column: list[int]) -> None:
        self.slot[death] = len(self.offsets) - 1
        if len(self.data) + len(column) >= 2**32:
            raise MemoryError('Packed reduced-column arena exceeds uint32 capacity')
        self.data.extend(column)
        self.offsets.append(len(self.data))

    def read_array(self, death: int) -> array:
        """Return a packed copy for internal reduction, without Python int objects."""
        slot = int(self.slot[death])
        if slot < 0:
            raise KeyError(death)
        return self.data[self.offsets[slot]:self.offsets[slot + 1]]

    def __getitem__(self, death: int) -> array:
        slot = int(self.slot[death])
        if slot < 0:
            raise KeyError(death)
        return list(self.data[self.offsets[slot]:self.offsets[slot + 1]])

    def __iter__(self):
        return (int(j) for j in np.flatnonzero(self.slot >= 0))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get(self, key: int, default=None):
        slot = int(self.slot[key])
        if slot < 0:
            return default
        return list(self.data[self.offsets[slot]:self.offsets[slot + 1]])

    @property
    def nbytes(self) -> int:
        return self.slot.nbytes + 4 * (len(self.offsets) + len(self.data))


class PackedOperations(Mapping[int, list[int]]):
    """CSR operation log; empty columns consume only one uint32 offset."""

    def __init__(self):
        self.offsets = array('I', [0])
        self.data = array('I')
        self.nonempty = 0

    def append(self, ops: list[int]) -> None:
        if len(self.data) + len(ops) >= 2**32:
            raise MemoryError('Packed operation-log arena exceeds uint32 capacity')
        self.data.extend(ops)
        self.offsets.append(len(self.data))
        if ops:
            self.nonempty += 1

    def __getitem__(self, target: int) -> array:
        start, end = self.offsets[target], self.offsets[target + 1]
        if start == end:
            raise KeyError(target)
        return list(self.data[start:end])

    def get(self, target: int, default=None):
        start, end = self.offsets[target], self.offsets[target + 1]
        return list(self.data[start:end]) if start < end else default

    def __iter__(self):
        return (i for i in range(len(self.offsets)-1)
                if self.offsets[i] != self.offsets[i+1])

    def __len__(self) -> int:
        return self.nonempty

    @property
    def nbytes(self) -> int:
        return 4 * (len(self.offsets) + len(self.data))


@dataclass(slots=True)
class Reduction:
    low: np.ndarray
    reduced_columns: Mapping[int, list[int]]
    operations: Mapping[int, list[int]]
    scratch: TemporaryDirectory | None = None

    def close(self):
        if isinstance(self.reduced_columns, DiskColumns):
            self.reduced_columns.close()
        if isinstance(self.operations, DiskOperations):
            self.operations.close()
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None

    @property
    def disk_bytes(self) -> int | None:
        if isinstance(self.reduced_columns, DiskColumns):
            return self.reduced_columns.disk_bytes + self.operations.disk_bytes
        return None

    @property
    def packed_bytes(self) -> int | None:
        if isinstance(self.reduced_columns, PackedColumns) and isinstance(self.operations, PackedOperations):
            return self.low.nbytes + self.reduced_columns.nbytes + self.operations.nbytes
        return None


def _reduce_compiled_packed(complex_: CompressedComplex, *,
                            record_targets: np.ndarray | None) -> Reduction:
    """Compile the entire canonical reduction, not a per-column XOR callback.

    Works on the compressed CSR matrix, retaining canonical pivot and column
    operation order. The integer buffers are repacked without Python-level
    per-entry conversion.
    """
    from ._compiled_reduction import reduce_csr
    if hasattr(complex_, 'offsets') and hasattr(complex_, 'indices'):
        input_offsets = np.frombuffer(complex_.offsets, dtype=np.uint32)
        input_indices = np.frombuffer(complex_.indices, dtype=np.uint32)
    else:
        # Reference block compression stores materialized boundary lists.
        idx = array('I')
        off = array('I', [0])
        for j in range(len(complex_)):
            idx.extend(complex_.boundary(j))
            off.append(len(idx))
        input_offsets = np.frombuffer(off, dtype=np.uint32)
        input_indices = np.frombuffer(idx, dtype=np.uint32)
    low, slot, r_off, r_data, op_off, op_data = reduce_csr(
        input_offsets, input_indices, complex_.values,
        compressed=isinstance(complex_, CompressedComplex) or hasattr(complex_, 'offsets'),
        record_targets=record_targets,
    )
    reduced = PackedColumns(len(complex_))
    reduced.slot = slot
    reduced.offsets = array('I')
    reduced.offsets.frombytes(r_off.tobytes())
    reduced.data = array('I')
    reduced.data.frombytes(r_data.tobytes())
    operations = PackedOperations()
    operations.offsets = array('I')
    operations.offsets.frombytes(op_off.tobytes())
    operations.data = array('I')
    operations.data.frombytes(op_data.tobytes())
    operations.nonempty = int(np.count_nonzero(op_off[1:] != op_off[:-1]))
    return Reduction(low, reduced, operations)


def _reduce_once(complex_: FilteredComplex | CompressedComplex, *,
                 storage: str, record_targets: np.ndarray | None,
                 accelerator: str, temp_dir: str | Path | None) -> Reduction:
    if storage == 'packed' and accelerator == 'numba' and hasattr(complex_, 'offsets'):
        return _reduce_compiled_packed(complex_, record_targets=record_targets)
    n = len(complex_)
    low = np.full(n, -1, dtype=np.int32 if storage != 'python' else np.int64)
    scratch = None
    if storage in ('packed', 'disk'):
        pivot_to_death = np.full(n, -1, dtype=np.int32)
        if storage == 'packed':
            reduced = PackedColumns(n)
            operations = PackedOperations()
        else:
            scratch = TemporaryDirectory(prefix='ppg_reduce_', dir=temp_dir)
            reduced = DiskColumns(n, Path(scratch.name) / 'columns.u32')
            operations = DiskOperations(Path(scratch.name) / 'operations.u32')
    else:
        pivot_to_death = {}
        reduced = {}
        operations = {}
    if accelerator == 'numba':
        from ._accelerator import xor_sorted_compiled
    for j in range(n):
        col = complex_.boundary(j)
        ops: list[int] = []
        while col:
            pivot = col[-1]
            source = (int(pivot_to_death[pivot]) if storage != 'python'
                      else pivot_to_death.get(pivot, -1))
            if source < 0:
                break
            source_col = (reduced.read_array(source) if storage == 'packed'
                          else reduced[source])
            if accelerator == 'numba' and len(col) + len(source_col) >= 128:
                col = xor_sorted_compiled(col, source_col)
            else:
                col = xor_sorted(col, source_col)
            if record_targets is None or record_targets[j]:
                ops.append(source)
        if storage != 'python':
            operations.append(ops)
        elif ops:
            operations[j] = ops
        if col:
            pivot = col[-1]
            if isinstance(complex_, CompressedComplex) and not (complex_.values[pivot] < complex_.values[j]):
                raise RuntimeError('Compressed reduction produced a zero-length pair')
            pivot_to_death[pivot] = j
            low[j] = pivot
            if storage != 'python':
                reduced.append(j, col)
            else:
                reduced[j] = col
    return Reduction(low=low, reduced_columns=reduced, operations=operations,
                     scratch=scratch)


def reduce_boundary(complex_: FilteredComplex | CompressedComplex, *,
                    storage: str = 'packed', log_mode: str = 'full',
                    accelerator: str = 'python',
                    temp_dir: str | Path | None = None) -> Reduction:
    """Canonical F2 reduction with packed columns and transformation logs.

    ``log_mode='selective'`` repeats reduction to save *only* operation targets
    appearing in the support of a reduced death column. This preserves the
    canonical V-inverse coordinates queried during PPG extraction, but costs
    two reduction passes. Choose it for workloads where logs dominate memory.
    """
    if storage not in ('packed', 'python', 'disk'):
        raise ValueError("storage must be 'packed', 'disk', or 'python'")
    if log_mode not in ('full', 'selective'):
        raise ValueError("log_mode must be 'full' or 'selective'")
    if accelerator not in ('python', 'numba'):
        raise ValueError("accelerator must be 'python' or 'numba'")
    if accelerator == 'numba':
        from ._accelerator import has_numba
        if not has_numba():
            raise ImportError("Install the optional 'speed' extra for Numba acceleration")
    if log_mode == 'full':
        return _reduce_once(complex_, storage=storage, record_targets=None, accelerator=accelerator,
                            temp_dir=temp_dir)
    first = _reduce_once(complex_, storage=storage,
                         record_targets=np.zeros(len(complex_), dtype=np.bool_), accelerator=accelerator,
                         temp_dir=temp_dir)
    requested = np.zeros(len(complex_), dtype=np.bool_)
    for column in first.reduced_columns.values():
        for row in column:
            requested[row] = True
    first_low = first.low
    first.close()
    second = _reduce_once(complex_, storage=storage, record_targets=requested,
                          accelerator=accelerator, temp_dir=temp_dir)
    if not np.array_equal(first_low, second.low):
        second.close()
        raise RuntimeError('Selective replay changed canonical reduction pivots')
    return second
