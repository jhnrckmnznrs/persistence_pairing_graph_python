"""Optional compiled F2 sparse XOR and delta-varint decoder.

Numba is deliberately optional. The canonical algorithm and deterministic
pivot order are the same for compiled and Python kernels.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as _numba
except ImportError:
    _numba = None


if _numba is not None:
    @_numba.njit(cache=True)
    def _xor_uint32(a, b):
        out = np.empty(len(a) + len(b), dtype=np.uint32)
        i = j = k = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
                j += 1
            elif a[i] < b[j]:
                out[k] = a[i]
                i += 1
                k += 1
            else:
                out[k] = b[j]
                j += 1
                k += 1
        while i < len(a):
            out[k] = a[i]
            i += 1
            k += 1
        while j < len(b):
            out[k] = b[j]
            j += 1
            k += 1
        return out[:k]


    @_numba.njit(cache=True)
    def _decode_varint_block(payload, position, end, previous, count):
        out = np.empty(count, dtype=np.uint32)
        for k in range(count):
            shift = 0
            gap = 0
            while True:
                if position >= end or shift > 28:
                    raise ValueError('Invalid varint order payload')
                byte = int(payload[position])
                position += 1
                gap |= (byte & 127) << shift
                if not byte & 128:
                    break
                shift += 7
            if gap <= 0:
                raise ValueError('Nonpositive encoded gap')
            previous += gap
            out[k] = previous
        return out, position, previous


def xor_sorted_compiled(a: list[int], b: list[int]) -> list[int]:
    if _numba is None:
        raise ImportError('Numba accelerator requires pip install persistence-pairing-graph[speed]')
    return _xor_uint32(np.asarray(a, dtype=np.uint32),
                       np.asarray(b, dtype=np.uint32)).tolist()


def has_numba() -> bool:
    return _numba is not None


def decode_varint_block(payload, position: int, end: int, previous: int, count: int):
    if _numba is None:
        raise ImportError('Numba accelerator requires pip install persistence-pairing-graph[speed]')
    return _decode_varint_block(payload, position, end, previous, count)

