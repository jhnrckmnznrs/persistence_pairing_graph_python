"""Interpret exact PPG transitions alongside verified original-grid cycle geometry.

The directed edge is an algebraic relation computed from the canonical PPG
reduction. The two lifted supports are *birth-time representatives*: geometric
intersection is not implied by, or required for, a nonzero transition.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ._graph import PersistenceEdge, PersistenceNode, build_nodes, iter_edges
from ._lift import LiftedCycleRepresentative, lift_compressed_cycles
from ._reduction import reduce_boundary
from ._streaming import compress_discrete_tensor_streaming


def _bounds(cycle: LiftedCycleRepresentative, spacing: np.ndarray):
    coords = cycle.doubled_grid_coordinates.astype(np.float64)
    half = (coords % 2) * 0.5
    lower = (coords * 0.5 - half) * spacing
    upper = (coords * 0.5 + half) * spacing
    return lower, upper


def _min_cell_distance(a: tuple[np.ndarray, np.ndarray],
                       b: tuple[np.ndarray, np.ndarray], max_pairs: int) -> float | None:
    """Exact minimum Euclidean distance between closed cubical cells.

    Return None rather than a misleading approximation if the Cartesian
    product exceeds the caller's explicit work limit. Uses O(chunk * ndim)
    temporary arrays rather than a complete pairwise distance tensor.
    """
    a0, a1 = a
    b0, b1 = b
    if not len(a0) or not len(b0):
        raise ValueError('A lifted cycle must contain at least one cubical cell')
    if len(a0) * len(b0) > max_pairs:
        return None
    best = float('inf')
    # Keep each temporary within ~32k cell pairs even when one cycle is huge.
    # The full work cap remains the separate max_pairs limit above.
    b_batch = min(len(b0), 512)
    a_batch = max(1, 32768 // b_batch)
    for ai in range(0, len(a0), a_batch):
        lo = a0[ai:ai + a_batch, None, :]
        hi = a1[ai:ai + a_batch, None, :]
        for bi in range(0, len(b0), b_batch):
            rhs_lo = b0[None, bi:bi + b_batch, :]
            rhs_hi = b1[None, bi:bi + b_batch, :]
            delta = np.maximum(np.maximum(rhs_lo - hi, lo - rhs_hi), 0.0)
            sq = np.sum(delta * delta, axis=2)
            best = min(best, float(np.min(sq)))
            if best == 0.0:
                return 0.0
    return float(np.sqrt(best))


@dataclass(frozen=True, slots=True)
class EdgeGeometry:
    """A canonical graph edge annotated with its two verified birth cycles."""
    edge_index: int
    edge: PersistenceEdge
    event_value: int | float
    source: LiftedCycleRepresentative
    target: LiftedCycleRepresentative
    shared_cell_count: int
    min_distance: float | None
    # Physical bounding boxes in image axis order (row, col) or (z, row, col).
    source_bounds: tuple[tuple[float, ...], tuple[float, ...]]
    target_bounds: tuple[tuple[float, ...], tuple[float, ...]]

    @property
    def geometrically_touching(self) -> bool | None:
        return None if self.min_distance is None else self.min_distance == 0.0


@dataclass(slots=True)
class EdgeExplanationReport:
    """Selected PPG edges and their spatial representatives.

    Invariants: edge ``coefficient`` is 1 over F2; supports are lifted using
    the same compressed chain complex and reduction as the graph extraction.
    ``min_distance=None`` indicates that the exact comparison was skipped.
    """
    image_shape: tuple[int, ...]
    spacing: tuple[float, ...]
    edges: list[EdgeGeometry]

    @property
    def cycles(self) -> dict[int, LiftedCycleRepresentative]:
        result: dict[int, LiftedCycleRepresentative] = {}
        for explanation in self.edges:
            result[explanation.edge.from_node] = explanation.source
            result[explanation.edge.to_node] = explanation.target
        return result

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(('edge_index', 'dimension', 'from_node', 'to_node',
                             'transition_value', 'from_birth_value', 'from_death_value',
                             'to_birth_value', 'to_death_value', 'coefficient',
                             'from_cycle_cells', 'to_cycle_cells', 'shared_cells',
                             'minimum_cell_distance', 'geometrically_touching'))
            for item in self.edges:
                e, a, b = item.edge, item.source, item.target
                writer.writerow((item.edge_index, e.dimension, e.from_node, e.to_node,
                                 item.event_value, a.birth_value, a.death_value,
                                 b.birth_value, b.death_value, e.coefficient,
                                 len(a.original_grid_indices), len(b.original_grid_indices),
                                 item.shared_cell_count,
                                 '' if item.min_distance is None else item.min_distance,
                                 '' if item.geometrically_touching is None
                                 else int(item.geometrically_touching)))
        return path

    def write_npz(self, path: str | Path) -> Path:
        """Non-pickle NPZ: supports stored once per endpoint node in CSR form."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cycles = self.cycles
        ids = sorted(cycles)
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        for i, idx in enumerate(ids):
            offsets[i + 1] = offsets[i] + len(cycles[idx].original_grid_indices)
        supports = (np.concatenate([cycles[idx].original_grid_indices for idx in ids])
                    if ids else np.empty(0, dtype=np.int64))
        np.savez_compressed(
            path,
            image_shape=np.asarray(self.image_shape, dtype=np.int64),
            doubled_grid_shape=np.asarray(tuple(2 * n + 1 for n in self.image_shape), dtype=np.int64),
            spacing=np.asarray(self.spacing, dtype=np.float64),
            node_ids=np.asarray(ids, dtype=np.int64),
            dimensions=np.asarray([cycles[i].dimension for i in ids], dtype=np.int16),
            compressed_birth_cells=np.asarray([cycles[i].birth_cell for i in ids], dtype=np.int64),
            compressed_death_cells=np.asarray([
                cycles[i].death_cell if cycles[i].death_cell is not None else -1 for i in ids
            ], dtype=np.int64),
            birth_values=np.asarray([cycles[i].birth_value for i in ids], dtype=np.float64),
            death_values=np.asarray([cycles[i].death_value for i in ids], dtype=np.float64),
            support_offsets=offsets,
            original_grid_cell_ids=supports,
            edge_indices=np.asarray([e.edge_index for e in self.edges], dtype=np.int64),
            from_node=np.asarray([e.edge.from_node for e in self.edges], dtype=np.int64),
            to_node=np.asarray([e.edge.to_node for e in self.edges], dtype=np.int64),
            edge_dimensions=np.asarray([e.edge.dimension for e in self.edges], dtype=np.int16),
            transition_values=np.asarray([e.event_value for e in self.edges], dtype=np.float64),
            minimum_cell_distance=np.asarray([
                np.nan if e.min_distance is None else e.min_distance for e in self.edges
            ], dtype=np.float64),
            shared_cell_count=np.asarray([e.shared_cell_count for e in self.edges], dtype=np.int64),
        )
        return path


def explain_ppg_edges(image: np.ndarray, *, dimensions: Sequence[int] = (1, 2),
                      max_edges: int = 10, edge_indices: Sequence[int] | None = None,
                      min_persistence: float = 0.0,
                      spacing: Sequence[float] | None = None,
                      max_geometry_pairs: int = 1_000_000,
                      max_cells: int = 100_000_000, temp_dir=None,
                      order_format: str = 'u32', order_builder: str = 'auto',
                      accelerator: str = 'python', metadata_storage: str = 'memory',
                      reduction_storage: str = 'packed',
                      max_relation_cache_entries: int = 100_000,
                      max_basis_columns: int = 200_000,
                      max_basis_entries: int = 5_000_000,
                      max_chain_terms: int = 500_000,
                      max_work: int = 5_000_000) -> EdgeExplanationReport:
    """Compute selected exact PPG edges and attach independently verified cycles.

    Source and target are the selected *birth-time* cycle representatives.
    Their intersection, shared cells and geometric separation are descriptive;
    they do not define an edge, which is calculated from the canonical
    simultaneous-death transition matrix over F2.

    ``edge_indices`` are zero-based indices in the full canonical edge stream.
    The first ``max_edges`` qualifying edges are selected if omitted. All
    traced cancellation pairs remain disk-backed until the selected endpoint
    cycles have been lifted. Full chain homotopy maps are not constructed.
    """
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError('Expected a 2D or 3D scalar image')
    if max_edges < 0 or max_geometry_pairs < 0 or min_persistence < 0:
        raise ValueError('Limits and min_persistence must be nonnegative')
    dims = tuple(int(d) for d in dimensions)
    if any(d < 0 or d > image.ndim for d in dims):
        raise ValueError('Requested dimension must be less than the image dimension')
    if spacing is None:
        spacing_tuple = (1.0,) * image.ndim
    else:
        spacing_tuple = tuple(float(s) for s in spacing)
    if len(spacing_tuple) != image.ndim or not all(np.isfinite(s) and s > 0 for s in spacing_tuple):
        raise ValueError('spacing must contain one finite positive number per image axis')
    indices = None if edge_indices is None else sorted(set(int(k) for k in edge_indices))
    if indices is not None and (any(k < 0 for k in indices) or len(indices) > max_edges):
        raise ValueError('edge_indices must be nonnegative and contain at most max_edges entries')
    if max_edges == 0 or indices == [] or dims == ():
        return EdgeExplanationReport(tuple(image.shape), spacing_tuple, [])

    compressed = compress_discrete_tensor_streaming(
        image, max_cells=max_cells, temp_dir=temp_dir,
        retain_chain_lifts=True, retain_cell_provenance=True,
        order_format=order_format, order_builder=order_builder,
        accelerator=accelerator, metadata_storage=metadata_storage)
    try:
        reduction = reduce_boundary(compressed, storage=reduction_storage,
                                    log_mode='full', accelerator=accelerator, temp_dir=temp_dir)
        try:
            nodes = build_nodes(compressed.values, compressed.dimensions,
                                reduction, storage='compact')
            selected: list[tuple[int, PersistenceEdge]] = []
            wanted = set(indices) if indices is not None else None
            for position, edge in enumerate(iter_edges(
                    compressed.values, reduction, nodes,
                    max_relation_cache_entries=max_relation_cache_entries)):
                if indices is not None and position > indices[-1]:
                    break
                if (wanted is None or position in wanted) and edge.dimension in dims:
                    a, b = nodes[edge.from_node], nodes[edge.to_node]
                    if a.persistence >= min_persistence and b.persistence >= min_persistence:
                        selected.append((position, PersistenceEdge(
                            int(edge.dimension), int(edge.from_node), int(edge.to_node),
                            int(edge.coefficient))))
                if len(selected) >= max_edges or (indices is not None and position >= indices[-1]):
                    break
            if indices is not None and len(selected) != len(indices):
                raise ValueError('Some edge_indices were absent or failed the filters')
            if not selected:
                return EdgeExplanationReport(tuple(image.shape), spacing_tuple, [])
            node_ids = sorted({n for _, e in selected for n in (e.from_node, e.to_node)})
            lifted = lift_compressed_cycles(
                compressed, reduction, nodes, dimensions=dims,
                max_features=len(node_ids), node_indices=node_ids,
                min_persistence=min_persistence,
                max_basis_columns=max_basis_columns,
                max_basis_entries=max_basis_entries,
                max_chain_terms=max_chain_terms, max_work=max_work)
            reps = dict(zip(node_ids, lifted))
            scale = np.asarray(spacing_tuple, dtype=np.float64)
            boxes = {i: _bounds(rep, scale) for i, rep in reps.items()}
            bounds = {i: (tuple(map(float, np.min(boxes[i][0], axis=0))),
                          tuple(map(float, np.max(boxes[i][1], axis=0))))
                      for i in node_ids}
            explained = []
            for position, edge in selected:
                source, target = reps[edge.from_node], reps[edge.to_node]
                if not (source.dimension == target.dimension == edge.dimension
                        and source.birth_value < source.death_value
                        and target.birth_value < source.death_value < target.death_value):
                    raise RuntimeError('PPG source/target lifetime invariant violated')
                shared = int(np.intersect1d(source.original_grid_indices,
                                            target.original_grid_indices, assume_unique=True).size)
                distance = (0.0 if shared else
                            _min_cell_distance(boxes[edge.from_node], boxes[edge.to_node],
                                               max_geometry_pairs))
                explained.append(EdgeGeometry(
                    position, edge, source.death_value, source, target, shared, distance,
                    bounds[edge.from_node], bounds[edge.to_node]))
            return EdgeExplanationReport(tuple(image.shape), spacing_tuple, explained)
        finally:
            reduction.close()
    finally:
        if compressed.lift_trace is not None:
            compressed.lift_trace.close()
