"""Disk-backed sparse reduction arenas with bounded resident column storage."""
from __future__ import annotations

from array import array
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class DiskColumns(Mapping[int, list[int]]):
    """Append-only reduced columns, with per-column random reads from scratch."""

    def __init__(self, n: int, path: Path):
        self.slot = np.full(n, -1, dtype=np.int32)
        self.offsets = array('Q', [0])
        self.file = path.open('w+b', buffering=0)

    def append(self, death: int, column: list[int]) -> None:
        self.slot[death] = len(self.offsets) - 1
        payload = array('I', column)
        if len(payload) and max(payload) >= 2**32:
            raise OverflowError('Sparse column index exceeds uint32')
        self.file.seek(0, 2)
        self.file.write(payload.tobytes())
        self.offsets.append(self.offsets[-1] + len(payload))

    def __getitem__(self, death: int) -> list[int]:
        if death < 0 or death >= len(self.slot):
            raise KeyError(death)
        slot = int(self.slot[death])
        if slot < 0:
            raise KeyError(death)
        first, end = self.offsets[slot], self.offsets[slot + 1]
        self.file.seek(first * 4)
        payload = self.file.read((end-first)*4)
        if len(payload) != (end-first)*4:
            raise OSError('Truncated disk-backed reduction column')
        result = array('I')
        result.frombytes(payload)
        return result.tolist()

    def __iter__(self):
        return (int(j) for j in np.flatnonzero(self.slot >= 0))

    def __len__(self):
        return len(self.offsets) - 1

    @property
    def disk_bytes(self):
        return self.offsets[-1] * 4

    @property
    def resident_metadata_bytes(self):
        return self.slot.nbytes + 8 * len(self.offsets)

    def close(self):
        self.file.close()


class DiskOperations(Mapping[int, list[int]]):
    """Append-only full/selected operation log in a scratch binary file."""

    def __init__(self, path: Path):
        self.offsets = array('Q', [0])
        self.file = path.open('w+b', buffering=0)
        self.nonempty = 0

    def append(self, ops: list[int]) -> None:
        payload = array('I', ops)
        self.file.seek(0, 2)
        self.file.write(payload.tobytes())
        self.offsets.append(self.offsets[-1] + len(payload))
        if ops:
            self.nonempty += 1

    def __getitem__(self, target: int) -> list[int]:
        if target < 0 or target + 1 >= len(self.offsets):
            raise KeyError(target)
        first, end = self.offsets[target], self.offsets[target+1]
        if end == first:
            raise KeyError(target)
        self.file.seek(first*4)
        payload = self.file.read((end-first)*4)
        if len(payload) != (end-first)*4:
            raise OSError('Truncated disk-backed operation log')
        result = array('I')
        result.frombytes(payload)
        return result.tolist()

    def __iter__(self):
        return (j for j in range(len(self.offsets)-1)
                if self.offsets[j] != self.offsets[j+1])

    def __len__(self):
        return self.nonempty

    @property
    def disk_bytes(self):
        return self.offsets[-1] * 4

    @property
    def resident_metadata_bytes(self):
        return 8 * len(self.offsets)

    def close(self):
        self.file.close()
