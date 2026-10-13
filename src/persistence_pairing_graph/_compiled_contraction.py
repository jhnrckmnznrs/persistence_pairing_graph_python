"""Optional compiled emission of surviving streamed cubical generators.

Only the final (post-cancellation) phase is JIT-compiled here. Earlier
within-value cancellations and relation finalization retain their proven
Python code path. The kernel writes compact CSR output and reuses the original
metadata map, without building Python lists of face indices per surviving cell.
"""
from __future__ import annotations

import numpy as np
try:
    from numba import njit
except ImportError:
    njit = None

_SIMPLE_LIMIT = 1 << 30
_FINAL_INLINE = 2 << 30
_FINAL_ARENA = 3 << 30
_MASK = (1 << 30) - 1

if njit is not None:
    @njit(cache=True)
    def _toggle(entries, row):
        if row in entries:
            entries.remove(row)
        else:
            entries.add(row)

    @njit(cache=True)
    def _grow(buf, used, needed):
        if needed <= len(buf):
            return buf
        out = np.empty(max(needed, max(128, 2*len(buf))), dtype=np.uint32)
        out[:used] = buf[:used]
        return out

    @njit(cache=True)
    def _emit_batch(ids, metadata, finalized, grid_shape, grid_strides, compressed_base):
        out_ids = np.empty(len(ids), dtype=np.uint32)
        out_offsets = np.zeros(len(ids)+1, dtype=np.uint32)
        out_data = np.empty(max(128, len(ids)), dtype=np.uint32)
        emitted = 0
        used = 0
        for old in ids:
            old = np.int64(old)
            if metadata[old] != 1:
                continue
            support = set([np.uint32(0)])
            support.clear()
            for axis in range(len(grid_shape)):
                stride = np.int64(grid_strides[axis])
                if (old // stride) % grid_shape[axis] % 2 == 0:
                    continue
                for face in (old - stride, old + stride):
                    meta = np.uint32(metadata[face])
                    tag = meta & np.uint32(3 << 30)
                    if tag == 0:
                        if 3 <= meta < _SIMPLE_LIMIT:
                            _toggle(support, np.uint32(meta - 3))
                        elif meta == 2:
                            continue
                        else:
                            raise ValueError('Compressed emission face is not resolved')
                    elif tag == _FINAL_INLINE:
                        inline = np.int64(meta & _MASK)
                        if inline:
                            meta2 = np.uint32(metadata[inline-1])
                            if not 3 <= meta2 < _SIMPLE_LIMIT:
                                raise ValueError('Invalid inline final replacement')
                            _toggle(support, np.uint32(meta2-3))
                    elif tag == _FINAL_ARENA:
                        position = np.int64(meta & _MASK)
                        length = np.int64(finalized[position])
                        for q in range(position + 1, position + 1 + length):
                            lower = np.int64(finalized[q])
                            lower_meta = np.uint32(metadata[lower])
                            if not 3 <= lower_meta < _SIMPLE_LIMIT:
                                raise ValueError('Invalid arena final replacement')
                            _toggle(support, np.uint32(lower_meta-3))
                    else:
                        raise ValueError('Encountered unfinished within-block relation')
            ordered = np.empty(len(support), dtype=np.uint32)
            p = 0
            for term in support:
                if term >= compressed_base + emitted:
                    raise ValueError('Compressed boundary references future generator')
                ordered[p] = term
                p += 1
            ordered = np.sort(ordered)
            out_data = _grow(out_data, used, used+len(ordered))
            for v in ordered:
                out_data[used] = v
                used += 1
            out_ids[emitted] = old
            emitted += 1
            out_offsets[emitted] = used
            if compressed_base + emitted + 3 >= _SIMPLE_LIMIT:
                raise ValueError('Exceeded compact active-cell index capacity')
            metadata[old] = np.uint32(compressed_base + emitted + 2)
        return out_ids[:emitted].copy(), out_offsets[:emitted+1].copy(), out_data[:used].copy()


def emit_survivors(ids, metadata, finalized, grid_shape, strides, compressed_base):
    if njit is None:
        raise ImportError('Numba is required for accelerated contraction')
    return _emit_batch(np.asarray(ids, dtype=np.uint32), metadata,
                       np.asarray(finalized, dtype=np.uint32),
                       np.asarray(grid_shape, dtype=np.int64),
                       np.asarray(strides, dtype=np.int64), compressed_base)
