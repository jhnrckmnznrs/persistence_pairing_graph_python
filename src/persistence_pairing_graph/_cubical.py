"""Implicit doubled-grid cubical filtration of 2D/3D scalar arrays."""
from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(slots=True)
class FilteredComplex:
    """Cells are sorted by (filtration value, dimension, row-major grid index)."""

    values: np.ndarray
    dimensions: np.ndarray
    grid_indices: np.ndarray
    grid_shape: tuple[int, ...]
    strides: tuple[int, ...]
    old_to_new: np.ndarray

    def __len__(self) -> int:
        return len(self.values)

    def boundary(self, j: int) -> list[int]:
        """Return sorted indices of codimension-one faces over F_2."""
        if self.dimensions[j] == 0:
            return []
        old = int(self.grid_indices[j])
        faces = []
        for axis, stride in enumerate(self.strides):
            if (old // stride) % self.grid_shape[axis] & 1:
                faces.append(int(self.old_to_new[old - stride]))
                faces.append(int(self.old_to_new[old + stride]))
        faces.sort()
        return faces

    def coordinates(self, cell: int) -> tuple[int, ...]:
        return tuple(int(x) for x in np.unravel_index(int(self.grid_indices[cell]), self.grid_shape))


def _expanded_values(image: np.ndarray) -> np.ndarray:
    """Min of the incident top cells, computed as separable axis transforms."""
    result = image
    for axis in range(image.ndim):
        n = result.shape[axis]
        shape = list(result.shape)
        shape[axis] = 2 * n + 1
        expanded = np.empty(shape, dtype=result.dtype)

        def index(slice_):
            s = [slice(None)] * result.ndim
            s[axis] = slice_
            return tuple(s)

        expanded[index(slice(1, None, 2))] = result
        expanded[index(0)] = result[index(0)]
        expanded[index(-1)] = result[index(-1)]
        if n > 1:
            expanded[index(slice(2, -1, 2))] = np.minimum(
                result[index(slice(None, -1))], result[index(slice(1, None))]
            )
        result = expanded
    return result


def build_cubical_complex(image: np.ndarray, *, max_cells: int | None = 3_000_000) -> FilteredComplex:
    """Use top-dimensional pixels/voxels as input scalar samples.

    Grid coordinates are 0..2*n, odd coordinates correspond to cell interiors,
    and even coordinates to boundaries between neighboring top-dimensional cells.
    """
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("Expected a 2D image or 3D scalar volume")
    if any(size == 0 for size in image.shape):
        raise ValueError("Scalar field dimensions must be nonzero")
    if image.dtype not in (np.dtype('uint8'), np.dtype('uint16'),
                           np.dtype('float32'), np.dtype('float64')):
        raise TypeError("Supported scalar dtypes: uint8, uint16, float32, float64")
    if not np.all(np.isfinite(image)):
        raise ValueError("Scalar field must not contain NaN or infinity")
    shape = tuple(2 * s + 1 for s in image.shape)
    n_cells = prod(shape)
    if max_cells is not None and n_cells > max_cells:
        raise MemoryError(
            f"Doubled cubical grid has {n_cells:,} cells, exceeding max_cells={max_cells:,}. "
            "This Python implementation materializes the filtration order; "
            "use a crop or explicitly increase max_cells if memory permits."
        )
    expanded = _expanded_values(image)
    values = expanded.ravel()
    dims = np.zeros(shape, dtype=np.uint8)
    for axis, size in enumerate(shape):
        axis_shape = [1] * image.ndim
        axis_shape[axis] = size
        dims += (np.arange(size, dtype=np.uint8) & 1).reshape(axis_shape)
    dim_flat = dims.ravel()
    indices = np.arange(n_cells, dtype=np.int64)
    # Rust's ordering: birth, dimension, then old regular-grid index.
    order = np.lexsort((indices, dim_flat, values))
    old_to_new = np.empty(n_cells, dtype=np.int64)
    old_to_new[order] = np.arange(n_cells, dtype=np.int64)
    strides = tuple(prod(shape[axis + 1:]) for axis in range(image.ndim))
    return FilteredComplex(values[order], dim_flat[order], order, shape, strides, old_to_new)
