"""Disk-backed, memory-conscious value-block contraction for 2D/3D uint images.

The cell order is counting-bucketed to a temporary uint32 memmap. No global
filtration-value array, sorted cell object table, or old-to-new table is built.
The doubled-grid state is a 4-byte-per-cell array; sparse replacement relations
and compressed boundaries use packed uint32 arenas rather than Python lists.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import prod
import heapq
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import numpy as np

from ._cubical import _expanded_values
from ._order import build_delta_order
from ._lift import StreamingLiftData

_MASK = (1 << 30) - 1
_CURRENT = 1 << 30
_FINAL_INLINE = 2 << 30
_FINAL_ARENA = 3 << 30
# Simple states: 0 unprocessed, 1 provisional, 2 cancelled death,
# >=3 active (compressed index + 3). Other states are distinguished by top bits.


@dataclass(slots=True)
class StreamingCompressedComplex:
    """Compact CSR-boundary complex; generator indices are compressed order."""

    values: np.ndarray
    dimensions: np.ndarray
    offsets: array
    indices: array
    original_size: int
    original_grid_indices: np.ndarray | None
    order_payload_bytes: int = 0
    lift_trace: StreamingLiftData | None = None

    def __len__(self) -> int:
        return len(self.values)

    def boundary(self, j: int) -> list[int]:
        return list(self.indices[self.offsets[j]:self.offsets[j + 1]])

    @property
    def boundary_entries(self) -> int:
        return len(self.indices)


def _validate(image: np.ndarray, max_cells: int | None) -> tuple[tuple[int, ...], int]:
    if image.ndim not in (2, 3) or not all(image.shape):
        raise ValueError('Expected a nonempty 2D or 3D scalar array')
    if image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise TypeError("'stream-block' supports uint8/uint16 only; use 'block' for floating-point data")
    grid_shape = tuple(2 * int(s) + 1 for s in image.shape)
    n = prod(grid_shape)
    if n > _MASK - 3:
        raise ValueError('Compact streaming state supports fewer than 2**30 cubical cells')
    if max_cells is not None and n > max_cells:
        raise MemoryError(f'Doubled cubical grid has {n:,} cells, exceeding max_cells={max_cells:,}')
    return grid_shape, n


def _value_planes(image: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Yield each doubled-grid first-axis plane using only two expanded slices.

    The first-axis coordinate runs in increasing row-major grid order. The
    expanded (one or two dimensional) cross sections remain bounded by the
    spatial area of a single scalar slice, not the entire 3D volume.
    """
    prev = None
    for z in range(image.shape[0]):
        current = _expanded_values(image[z]) if image.ndim == 3 else _expand_row(image[z])
        if z == 0:
            yield 0, current
        else:
            yield 2 * z, np.minimum(prev, current)
        yield 2 * z + 1, current
        prev = current
    yield 2 * image.shape[0], prev


def _expand_row(row: np.ndarray) -> np.ndarray:
    n = row.size
    out = np.empty(2 * n + 1, dtype=row.dtype)
    out[1::2] = row
    out[0] = row[0]
    out[-1] = row[-1]
    if n > 1:
        out[2:-1:2] = np.minimum(row[:-1], row[1:])
    return out


def _plane_dimensions(grid_shape: tuple[int, ...]) -> np.ndarray:
    if len(grid_shape) == 2:
        return (np.arange(grid_shape[1], dtype=np.uint8) & 1)
    return ((np.arange(grid_shape[1], dtype=np.uint8) & 1)[:, None]
            + (np.arange(grid_shape[2], dtype=np.uint8) & 1)[None, :]).ravel()


def _iter_keys(image: np.ndarray, grid_shape: tuple[int, ...]):
    plane_dims = _plane_dimensions(grid_shape)
    width = len(plane_dims)
    level_multiplier = image.ndim + 1
    for first_axis, plane in _value_planes(image):
        # All keys fit in uint32 even for uint16 scalar images.
        keys = (plane.reshape(-1).astype(np.uint32) * level_multiplier
                + plane_dims + (first_axis & 1))
        yield first_axis * width, keys


def _count_buckets(image: np.ndarray, grid_shape: tuple[int, ...],
                   n_buckets: int) -> np.ndarray:
    """Count filtration buckets with bounded-memory batching for uint16.

    A separate full-size bincount for each small plane repeats O(levels)
    zero-initialization work. Batch up to ~131k keys, reducing this overhead
    while keeping the extra work array below approximately 512 KiB.
    """
    counts = np.zeros(n_buckets, dtype=np.int64)
    if n_buckets <= 4096:
        for _, keys in _iter_keys(image, grid_shape):
            counts += np.bincount(keys, minlength=n_buckets)
        return counts
    batch: list[np.ndarray] = []
    entries = 0
    for _, keys in _iter_keys(image, grid_shape):
        batch.append(keys)
        entries += len(keys)
        if entries >= 131_072:
            counts += np.bincount(np.concatenate(batch), minlength=n_buckets)
            batch.clear()
            entries = 0
    if batch:
        counts += np.bincount(np.concatenate(batch), minlength=n_buckets)
    return counts


def _boundary_old(old: int, grid_shape: tuple[int, ...], strides: tuple[int, ...]) -> list[int]:
    faces: list[int] = []
    for shape, stride in zip(grid_shape, strides):
        if (old // stride) % shape & 1:
            faces.append(old - stride)
            faces.append(old + stride)
    faces.sort()
    return faces


def _toggle(s: set[int], v: int) -> None:
    if v in s:
        s.remove(v)
    else:
        s.add(v)


def _arena_relation(arena: array, offset: int):
    length = arena[offset]
    return arena[offset + 1:offset + 1 + length]


def _project(chain: list[int], metadata: np.ndarray | memoryview,
             current: array, finalized: array) -> list[int]:
    """Project into active old-grid generators, resolving current births descending."""
    # Most boundary columns of a well-compressed volume touch only active
    # generators or cancelled death cells. Avoid a set, heap, and nested
    # routing function entirely on this common path. Incoming chains always
    # have unique indices (cellular boundaries and contracted relations).
    if not chain:
        return []
    plain = []
    for term in chain:
        meta = int(metadata[term])
        if 0 < meta < _CURRENT:
            if meta != 2:
                plain.append(term)
        else:
            if meta == 0:
                raise RuntimeError(f'Boundary references unprocessed cell {term}')
            break
    else:
        plain.sort()
        return plain

    active: set[int] = set()
    pending: set[int] = set()
    heap: list[int] = []

    def route(term: int) -> None:
        meta = int(metadata[term])
        tag = meta & ~_MASK
        if tag == 0:
            if meta == 0:
                raise RuntimeError(f'Boundary references unprocessed cell {term}')
            if meta != 2:  # provisional or assigned active
                _toggle(active, term)
        elif tag == _CURRENT:
            if term in pending:
                pending.remove(term)
            else:
                pending.add(term)
                heapq.heappush(heap, -term)
        elif tag == _FINAL_INLINE:
            if meta & _MASK:
                _toggle(active, (meta & _MASK) - 1)
        elif tag == _FINAL_ARENA:
            for lower in _arena_relation(finalized, meta & _MASK):
                _toggle(active, lower)
        else:
            raise RuntimeError('Invalid compact contraction state')

    for term in chain:
        route(term)
    while heap:
        term = -heapq.heappop(heap)
        if term not in pending:
            continue
        pending.remove(term)
        meta = int(metadata[term])
        if (meta & ~_MASK) != _CURRENT:
            raise RuntimeError('Current-block contraction relation changed during projection')
        for lower in _arena_relation(current, meta & _MASK):
            if lower >= term and (int(metadata[lower]) & ~_MASK) == _CURRENT:
                raise RuntimeError('Nontriangular current-block contraction relation')
            route(lower)
    return sorted(active)


def _store_current(current: array, relation: list[int]) -> int:
    offset = len(current)
    if offset > _MASK:
        raise MemoryError('Current replacement arena exceeds compact pointer capacity')
    current.append(len(relation))
    current.extend(relation)
    return _CURRENT | offset


def _store_final(finalized: array, relation: list[int]) -> int:
    if len(relation) < 2:
        return _FINAL_INLINE | (relation[0] + 1 if relation else 0)
    offset = len(finalized)
    if offset > _MASK:
        raise MemoryError('Final replacement arena exceeds compact pointer capacity')
    finalized.append(len(relation))
    finalized.extend(relation)
    return _FINAL_ARENA | offset


def _axis_monotone(image: np.ndarray) -> bool:
    """Sufficient criterion for every cubical sublevel set to be contractible.

    For each axis there must be one consistent monotonic direction. The
    comparisons use bounded first-axis slabs (at most ~1 MiB per comparison)
    to avoid creating a volume-sized temporary Boolean array.
    """
    slab = max(1, 1_000_000 // max(1, prod(image.shape[1:])))
    for axis in range(image.ndim):
        if image.shape[axis] <= 1:
            continue
        up = down = True
        end = image.shape[0] - (axis == 0)
        for begin in range(0, end, slab):
            stop = min(begin + slab, end)
            left = [slice(None)] * image.ndim
            right = [slice(None)] * image.ndim
            left[0] = slice(begin, stop)
            right[0] = slice(begin, stop)
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            if axis == 0:
                left[0] = slice(begin, stop)
                right[0] = slice(begin + 1, stop + 1)
            if up:
                up = bool(np.all(image[tuple(left)] <= image[tuple(right)]))
            if down:
                down = bool(np.all(image[tuple(left)] >= image[tuple(right)]))
            if not up and not down:
                return False
    return True


def _single_essential_complex(
    image: np.ndarray, grid_shape: tuple[int, ...], total: int,
    retain_cell_provenance: bool,
) -> StreamingCompressedComplex:
    """Return the exact canonical compressed complex for a monotone field."""
    old_minimum = int(np.argmin(image))
    sample_coord = np.unravel_index(old_minimum, image.shape)
    strides = tuple(prod(grid_shape[i + 1:]) for i in range(image.ndim))
    grid_vertex = sum(2 * c * stride for c, stride in zip(sample_coord, strides))
    return StreamingCompressedComplex(
        values=np.array([image.flat[old_minimum]], dtype=image.dtype),
        dimensions=np.array([0], dtype=np.uint8),
        offsets=array('I', [0, 0]), indices=array('I'),
        original_size=total,
        original_grid_indices=(np.array([grid_vertex], dtype=np.uint32)
                               if retain_cell_provenance else None),
    )


def compress_discrete_tensor_streaming(
    image: np.ndarray, *, max_cells: int | None = 100_000_000,
    retain_cell_provenance: bool = False,
    temp_dir: str | Path | None = None,
    order_format: str = 'u32',
    accelerator: str = 'python',
    order_builder: str = 'auto',
    metadata_storage: str = 'memory',
    retain_chain_lifts: bool = False,
) -> StreamingCompressedComplex:
    """Stream-contract equal-value pairs from a uint8/uint16 scalar array.

    Ordering: (filtration value, cell dimension, doubled-grid row-major index).
    Order payload uses a temporary disk-backed uint32 array (4 bytes/cell).
    State uses a resident uint32 array (4 bytes/cell); temporary relations and
    compressed boundaries are stored in packed uint32 buffers. The original
    tensor stays in memory until counting and filling have finished.

    A true persistent-homology reduction is performed on the compressed
    complex by the caller, with the same event-based graph extraction as the
    materialized Python backend.
    """
    if order_format not in ('delta', 'u32'):
        raise ValueError("order_format must be 'delta' or 'u32'")
    if order_builder not in ('auto', 'sort', 'counting'):
        raise ValueError("order_builder must be 'auto', 'sort', or 'counting'")
    if order_builder == 'counting' and order_format != 'u32':
        raise ValueError("counting order builder requires order_format='u32'")
    if metadata_storage not in ('memory', 'disk'):
        raise ValueError("metadata_storage must be 'memory' or 'disk'")

    image = np.asarray(image)
    grid_shape, total = _validate(image, max_cells)
    # A coordinatewise monotone scalar field (possibly reversed along each
    # axis) has contractible sublevel cubical unions. Its value-compressed
    # complex contains exactly one essential 0-generator, with no PPG edges.
    if _axis_monotone(image) and not retain_chain_lifts:
        return _single_essential_complex(
            image, grid_shape, total, retain_cell_provenance)
    if retain_chain_lifts:
        retain_cell_provenance = True
    ndim = image.ndim
    dim_count = ndim + 1
    levels = 256 if image.dtype == np.uint8 else 65536
    n_buckets = levels * dim_count
    counts = _count_buckets(image, grid_shape, n_buckets)
    if int(counts.sum()) != total:
        raise RuntimeError('Bucket counting did not visit every cubical cell')
    starts = np.empty(n_buckets + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])
    del counts
    strides = tuple(prod(grid_shape[i + 1:]) for i in range(ndim))
    project = _project
    values = array('B' if image.dtype == np.uint8 else 'H')
    dimensions = array('B')
    offsets = array('I', [0])
    indices = array('I')
    provenance = array('I') if retain_cell_provenance else None
    finalized = array('I')
    trace = StreamingLiftData(grid_shape, temp_dir) if retain_chain_lifts else None

    with TemporaryDirectory(prefix='ppg_order_', dir=temp_dir) as workdir:
        if metadata_storage == 'disk':
            metadata = np.memmap(Path(workdir) / 'metadata.u32', mode='w+',
                                 dtype=np.uint32, shape=(total,))
            metadata[:] = 0
        else:
            metadata = np.zeros(total, dtype=np.uint32)
        if order_format == 'delta':
            compressed_order = build_delta_order(image, grid_shape, np.diff(starts), workdir,
                                                 accelerator=accelerator)
            order_payload_bytes = compressed_order.payload_bytes
            def bucket_ids(bucket):
                return compressed_order.bucket(bucket)
        else:
            cursors = starts[:-1].copy()
            order = np.memmap(Path(workdir) / 'order.u32', mode='w+',
                              dtype=np.uint32, shape=(total,))
            use_counting = (order_builder == 'counting' or
                            (order_builder == 'auto' and accelerator == 'numba'))
            if use_counting:
                from ._compiled_order import scatter_plane
                for base, keys in _iter_keys(image, grid_shape):
                    scatter_plane(keys, base, cursors, order)
            else:
                for base, keys in _iter_keys(image, grid_shape):
                    local_order = np.argsort(keys, kind='stable')
                    sorted_keys = keys[local_order]
                    boundaries = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])
                    for a, b in zip(boundaries[:-1], boundaries[1:]):
                        bucket = int(sorted_keys[a])
                        pos = int(cursors[bucket])
                        order[pos:pos + (b - a)] = base + local_order[a:b]
                        cursors[bucket] = pos + (b - a)
            if not np.array_equal(cursors, starts[1:]):
                raise RuntimeError('Streaming order-bucket fill mismatch')
            order_view = memoryview(order)
            order_payload_bytes = 4 * total
            def bucket_ids(bucket):
                return (int(order_view[pos]) for pos in range(
                    int(starts[bucket]), int(starts[bucket + 1])))
        metadata_view = memoryview(metadata)

        try:
            # Iterate nonempty value blocks in filtration order. We revisit bucket
            # ids to finalize contractions and emit survivors, avoiding per-cell
            # Python objects for a whole large constant-valued block.
            nonempty_levels = np.flatnonzero(starts[dim_count::dim_count] != starts[:-dim_count:dim_count])
            for level_index in nonempty_levels:
                level = int(level_index)
                current = array('I')
                block_start = level * dim_count
                block_end = block_start + dim_count

                # Pass 1: contractions, checking same-value pivots by provisional tag.
                for bucket in range(block_start, block_end):
                    for old in bucket_ids(bucket):
                        projected = project(_boundary_old(old, grid_shape, strides), metadata,
                                            current, finalized)
                        pivot = next((x for x in reversed(projected) if metadata_view[x] == 1), None)
                        if pivot is None:
                            metadata[old] = 1
                        else:
                            # Current-block boundaries are homogeneous in dimension;
                            # the greatest provisional index is the canonical pivot.
                            projected.remove(pivot)
                            metadata[pivot] = _store_current(current, projected)
                            metadata[old] = 2
                            if trace is not None:
                                trace.add_pair(old, bucket % dim_count, level)

                # Pass 2: finalize temporary relations in increasing (dim, old id).
                for bucket in range(block_start, block_end):
                    for old in bucket_ids(bucket):
                        meta = metadata_view[old]
                        if (meta & ~_MASK) == _CURRENT:
                            resolved = project(list(_arena_relation(current, meta & _MASK)),
                                               metadata, current, finalized)
                            metadata[old] = _store_final(finalized, resolved)
                del current

                # Pass 3: emit survivors with final boundaries in compressed order.
                # With Numba, fuse boundary generation, final-relation projection,
                # canonical sorting and compact-metadata updates by bounded chunks.
                if accelerator == 'numba':
                    from itertools import islice
                    from ._compiled_contraction import emit_survivors
                    # Use an owned snapshot. Keeping an ndarray view of an
                    # array('I') can prevent a later value block from resizing
                    # the replacement arena (BufferError), even after the JIT
                    # call has returned. The bounded copied snapshot avoids
                    # exporting the mutable array's buffer across Numba calls.
                    finalized_view = np.array(finalized, dtype=np.uint32, copy=True)
                    for bucket in range(block_start, block_end):
                        iterator = iter(bucket_ids(bucket))
                        while True:
                            ids = np.fromiter(islice(iterator, 32768), dtype=np.uint32)
                            if len(ids) == 0:
                                break
                            emitted_ids, local_offsets, local_data = emit_survivors(
                                ids, metadata, finalized_view, grid_shape, strides, len(values))
                            values.extend([level] * len(emitted_ids))
                            dimensions.extend([bucket % dim_count] * len(emitted_ids))
                            if provenance is not None:
                                provenance.frombytes(emitted_ids.tobytes())
                            base = len(indices)
                            indices.frombytes(local_data.tobytes())
                            offsets.extend(int(base + offset) for offset in local_offsets[1:])
                    del finalized_view
                    continue
                empty_current = array('I')
                for bucket in range(block_start, block_end):
                    for old in bucket_ids(bucket):
                        if metadata_view[old] != 1:
                            continue
                        boundary_old = project(_boundary_old(old, grid_shape, strides),
                                               metadata, empty_current, finalized)
                        boundary_new = []
                        for face in boundary_old:
                            meta = metadata_view[face]
                            if not 3 <= meta < _CURRENT:
                                raise RuntimeError('Compressed boundary references non-active cell')
                            new = meta - 3
                            if new >= len(values):
                                raise RuntimeError('Compressed boundary references future generator')
                            boundary_new.append(new)
                        # Existing compressed indices preserve original within-value
                        # ordering, so strict filtering of surviving boundaries holds.
                        boundary_new.sort()
                        metadata[old] = len(values) + 3
                        values.append(level)
                        dimensions.append(bucket % dim_count)
                        indices.extend(boundary_new)
                        offsets.append(len(indices))
                        if provenance is not None:
                            provenance.append(old)

            # Verify each cell is resolved. Metadata remains local to contraction.
            if np.any(metadata == 0) or np.any(metadata == 1):
                raise RuntimeError('Streaming compression left unresolved cells')
        finally:
            metadata_view.release()
            del metadata
            if order_format == 'u32':
                order_view.release()
                del order
            else:
                del compressed_order
    del finalized

    valarr = np.frombuffer(values, dtype=image.dtype)
    dimarr = np.frombuffer(dimensions, dtype=np.uint8)
    provenance_arr = (np.frombuffer(provenance, dtype=np.uint32) if provenance is not None else None)
    if trace is not None:
        trace.finish()
    return StreamingCompressedComplex(valarr, dimarr, offsets, indices, total,
                                      provenance_arr, order_payload_bytes, trace)
