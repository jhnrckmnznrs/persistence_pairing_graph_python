"""Exact, per-bucket delta-varint cell order with bounded scratch storage.

Order is (scalar value, cell dimension, doubled-grid row-major index).
Each nonempty bucket stores its first cell id as uint32 followed by ULEB128
encoded positive differences. Unlike the raw uint32 scratch order, this path
needs a separate measurement pass but never writes a full raw order table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class DeltaOrder:
    counts: np.ndarray
    first: np.ndarray
    starts: np.ndarray
    payload: np.memmap
    accelerator: str = 'python'

    @property
    def payload_bytes(self) -> int:
        return int(self.payload.size) + 4 * int(np.count_nonzero(self.counts))

    def bucket(self, key: int):
        """Yield ordered old-grid ids in this bucket, without allocating a list."""
        count = int(self.counts[key])
        if not count:
            return
        value = int(self.first[key])
        yield value
        begin = int(self.starts[key])
        end = int(self.starts[key + 1])
        if count == 1:
            if begin != end:
                raise RuntimeError('Singleton bucket has nonempty delta payload')
            return
        if self.accelerator == 'numba':
            from ._accelerator import decode_varint_block
            position = begin
            remaining = count - 1
            while remaining:
                take = min(8192, remaining)
                decoded, position, value = decode_varint_block(
                    self.payload, position, end, value, take)
                yield from map(int, decoded)
                remaining -= take
            if position != end:
                raise RuntimeError('Extra data in delta-order bucket')
            return
        data = self.payload
        position = begin
        for _ in range(count - 1):
            shift = 0
            delta = 0
            while True:
                if position >= end or shift > 28:
                    raise RuntimeError('Malformed or truncated varint scratch order')
                byte = int(data[position])
                position += 1
                delta |= (byte & 127) << shift
                if not byte & 128:
                    break
                shift += 7
            if not delta:
                raise RuntimeError('Order contains a nonpositive cell-id gap')
            value += delta
            yield value
        if position != end:
            raise RuntimeError('Extra data in delta-order bucket')


def _groups(image, shape):
    # This helper imports lazily to avoid a module import cycle.
    from ._streaming import _iter_keys
    for base, keys in _iter_keys(image, shape):
        order = np.argsort(keys, kind='stable')
        skeys = keys[order]
        borders = np.flatnonzero(np.r_[True, skeys[1:] != skeys[:-1], True])
        for a, b in zip(borders[:-1], borders[1:]):
            yield int(skeys[a]), base + order[a:b]


def _gaps(ids: np.ndarray, previous: int) -> np.ndarray:
    first = int(ids[0]) if previous < 0 else previous
    # First id of first group is stored separately in the bucket descriptor.
    data = ids if previous >= 0 else ids[1:]
    if not len(data):
        return np.empty(0, dtype=np.uint32)
    gaps = np.empty(len(data), dtype=np.uint32)
    gaps[0] = int(data[0]) - first
    if len(data) > 1:
        gaps[1:] = np.diff(data).astype(np.uint32)
    if np.any(gaps == 0):
        raise RuntimeError('Repeated cell in filtration bucket')
    return gaps


def _widths(gaps: np.ndarray) -> np.ndarray:
    return (1 + (gaps >= 128).astype(np.uint8) + (gaps >= 16384).astype(np.uint8)
            + (gaps >= 2097152).astype(np.uint8) + (gaps >= 268435456).astype(np.uint8))


def _encode(gaps: np.ndarray, widths: np.ndarray) -> np.ndarray:
    if not len(gaps):
        return np.empty(0, dtype=np.uint8)
    out = np.empty(int(widths.astype(np.int64).sum()), dtype=np.uint8)
    cursors = np.cumsum(widths, dtype=np.int64) - widths
    for offset in range(5):
        eligible = widths > offset
        if not np.any(eligible):
            break
        values = ((gaps[eligible] >> (7 * offset)) & 127).astype(np.uint8)
        values |= (widths[eligible] > offset + 1).astype(np.uint8) * 128
        out[cursors[eligible] + offset] = values
    return out


def build_delta_order(image: np.ndarray, grid_shape: tuple[int, ...],
                      counts: np.ndarray, scratch: str | Path, *,
                      accelerator: str = 'python') -> DeltaOrder:
    """Count/measure/fill without keeping the full cell ordering in memory."""
    n = len(counts)
    first = np.zeros(n, dtype=np.uint32)
    last = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    for key, ids in _groups(image, grid_shape):
        prior = int(last[key])
        if prior < 0:
            first[key] = ids[0]
        gaps = _gaps(ids, prior)
        sizes[key] += int(_widths(gaps).astype(np.int64).sum())
        last[key] = ids[-1]
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    payload_size = int(offsets[-1])
    if payload_size == 0:
        # np.memmap cannot map empty files; a one-byte unused sentinel is fine.
        payload_size = 1
    path = Path(scratch) / 'order.delta'
    payload = np.memmap(path, dtype=np.uint8, mode='w+', shape=(payload_size,))
    positions = offsets[:-1].copy()
    last.fill(-1)
    for key, ids in _groups(image, grid_shape):
        gaps = _gaps(ids, int(last[key]))
        data = _encode(gaps, _widths(gaps))
        start = int(positions[key])
        payload[start:start + len(data)] = data
        positions[key] += len(data)
        last[key] = ids[-1]
    if not np.array_equal(positions, offsets[1:]):
        raise RuntimeError('Delta-order sizing and filling disagree')
    return DeltaOrder(counts, first, offsets, payload, accelerator)
