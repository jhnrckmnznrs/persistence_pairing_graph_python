"""Optional nopython reduction for a packed CSR filtered chain complex.

The reduction order, pivot selection and operation log are identical to the
pure-Python canonical left-to-right F2 reduction. Only the storage and the
inner elimination loop change. Numba is imported lazily by callers.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

if njit is not None:
    @njit(cache=True)
    def _grow(buf, used, needed):
        if needed <= len(buf):
            return buf
        new_size = max(needed, max(1024, len(buf) * 2))
        new = np.empty(new_size, dtype=np.uint32)
        new[:used] = buf[:used]
        return new

    @njit(cache=True)
    def _symmetric_into(a, an, b, bn, dest):
        i = j = k = 0
        while i < an and j < bn:
            ai, bj = a[i], b[j]
            if ai == bj:
                i += 1
                j += 1
            elif ai < bj:
                dest[k] = ai
                i += 1
                k += 1
            else:
                dest[k] = bj
                j += 1
                k += 1
        while i < an:
            dest[k] = a[i]
            i += 1
            k += 1
        while j < bn:
            dest[k] = b[j]
            j += 1
            k += 1
        return k

    @njit(cache=True)
    def _reduce_csr_kernel(offsets, indices, birth_values, compressed, record_mask):
        n = len(offsets) - 1
        low = np.full(n, -1, dtype=np.int32)
        death_slot = np.full(n, -1, dtype=np.int32)
        pivot_slot = np.full(n, -1, dtype=np.int32)
        r_offsets = np.zeros(n + 1, dtype=np.uint32)
        r_deaths = np.empty(n, dtype=np.int32)
        r_data = np.empty(max(1024, min(len(indices), 65536)), dtype=np.uint32)
        op_offsets = np.zeros(n + 1, dtype=np.uint32)
        op_data = np.empty(1024, dtype=np.uint32)
        nr = 0
        rd = 0
        od = 0
        for column in range(n):
            start, end = np.int64(offsets[column]), np.int64(offsets[column + 1])
            size = end - start
            work = np.empty(max(8, size * 2), dtype=np.uint32)
            spare = np.empty(max(8, size * 2), dtype=np.uint32)
            for i in range(size):
                work[i] = indices[start + i]
            while size:
                pivot = np.int64(work[size - 1])
                slot = np.int64(pivot_slot[pivot])
                if slot < 0:
                    break
                rs, re = np.int64(r_offsets[slot]), np.int64(r_offsets[slot + 1])
                nsource = re - rs
                needed = size + nsource
                if needed > len(spare):
                    spare = np.empty(needed * 2, dtype=np.uint32)
                outsize = _symmetric_into(work, size, r_data[rs:re], nsource, spare)
                work, spare = spare, work
                size = outsize
                if record_mask[column]:
                    op_data = _grow(op_data, od, od + 1)
                    op_data[od] = r_deaths[slot]
                    od += 1
            op_offsets[column + 1] = od
            if size:
                pivot = np.int64(work[size - 1])
                if compressed and not birth_values[pivot] < birth_values[column]:
                    raise ValueError('Compressed reduction produced a nonpositive pair')
                low[column] = pivot
                pivot_slot[pivot] = nr
                death_slot[column] = nr
                r_deaths[nr] = column
                r_data = _grow(r_data, rd, rd + size)
                for i in range(size):
                    r_data[rd + i] = work[i]
                rd += size
                nr += 1
                r_offsets[nr] = rd
        return (low, death_slot, r_offsets[:nr + 1].copy(), r_data[:rd].copy(),
                op_offsets, op_data[:od].copy())


def reduce_csr(offsets, indices, values, *, compressed: bool, record_targets=None):
    """Reduce packed columns, returning pivot/slot and F2 operation buffers."""
    if njit is None:
        raise ImportError('Numba is required for the compiled reduction backend')
    offsets = np.asarray(offsets, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    values = np.asarray(values)
    n = len(offsets) - 1
    if n >= 2**31:
        raise ValueError('Compiled reduction requires fewer than 2**31 generators')
    if offsets[-1] != len(indices):
        raise ValueError('Invalid CSR boundary arrays')
    record_mask = (np.ones(n, dtype=np.bool_) if record_targets is None else
                   np.asarray(record_targets, dtype=np.bool_))
    if len(record_mask) != n:
        raise ValueError('Record mask length must equal number of generators')
    return _reduce_csr_kernel(offsets, indices, values, compressed, record_mask)
