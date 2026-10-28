"""Persistence-pairing graphs for 2D and 3D cubical scalar filtrations."""
from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

from ._cubical import build_cubical_complex
from ._graph import PersistenceEdge, PersistenceNode, CompactNodeTable, iter_edges, build_nodes
from ._io import export_csv, load_image
from ._reduction import compress_value_blocks, reduce_boundary
from ._streaming import compress_discrete_tensor_streaming
from ._betti import BettiCurves, betti_curves, validate_ppg_betti
from ._representatives import CycleRepresentative, representative_cycles
from ._lift import (LiftedCycleRepresentative, representative_cycles_streaming,
                    lift_compressed_cycles, boundary_old_chain)
from ._summaries import EventSummary, write_event_summaries
from ._edge_geometry import EdgeExplanationReport, EdgeGeometry, explain_ppg_edges
from ._transition_witness import TransitionFillingWitness, transition_filling_witness
from ._filling_locality import (FillingLocalityCertificate, certify_local_filling,
                                eligible_filling_cells_in_box,
                                certify_transition_filling_locality)
from ._analysis import PPGAnalysis
from ._reverse_replay import SelectiveInclusionReplay, ReplayStatistics, DependencyClosureStatistics
from ._projection_replay import SelectiveProjectionReplay, ProjectionStatistics
from ._old_cohomology import (SparseOldLevelOracle, OldLevelCoordinates,
                               SparseRelativeCertificate)
from ._relative_locality import (RelativeLocalityCertificate, certify_relative_filling,
                                  certify_transition_relative_locality)
from ._multiscale import pool_scalar_field, multiscale_betti_curves, MultiscaleBetti
from ._robustness import ppg_perturbation_probe, PerturbationReport, PerturbationSample

__all__ = [
    'SelectiveInclusionReplay', 'ReplayStatistics', 'DependencyClosureStatistics',
    'SelectiveProjectionReplay', 'ProjectionStatistics',
    'SparseOldLevelOracle', 'OldLevelCoordinates', 'SparseRelativeCertificate',
    'RelativeLocalityCertificate', 'certify_relative_filling',
    'certify_transition_relative_locality',
    'compute_ppg', 'PersistencePairingGraph', 'PersistenceNode', 'PersistenceEdge', 'CompactNodeTable',
    'build_cubical_complex', 'compress_discrete_tensor_streaming', 'load_image', 'export_csv',
    'betti_curves', 'BettiCurves', 'validate_ppg_betti', 'representative_cycles', 'CycleRepresentative',
    'representative_cycles_streaming', 'LiftedCycleRepresentative', 'lift_compressed_cycles',
    'EventSummary', 'write_event_summaries', 'EdgeGeometry', 'EdgeExplanationReport', 'explain_ppg_edges',
    'TransitionFillingWitness', 'transition_filling_witness', 'PPGAnalysis',
    'FillingLocalityCertificate', 'certify_local_filling',
    'eligible_filling_cells_in_box', 'certify_transition_filling_locality',
    'pool_scalar_field', 'multiscale_betti_curves', 'MultiscaleBetti',
    'ppg_perturbation_probe', 'PerturbationReport', 'PerturbationSample',
]


@dataclass(slots=True)
class PersistencePairingGraph:
    """Graph plus the filtration metadata used to index its birth/death cells."""
    nodes: list[PersistenceNode] | CompactNodeTable
    edges: list[PersistenceEdge]
    algorithm: str
    original_cell_count: int
    reduced_cell_count: int
    cell_values: np.ndarray
    cell_dimensions: np.ndarray
    original_grid_indices: np.ndarray | None
    edge_count: int = 0
    edges_materialized: bool = True
    edge_output_path: Path | None = None
    event_summaries: list[EventSummary] | None = None
    in_degrees: np.ndarray | None = None
    out_degrees: np.ndarray | None = None

    def barcodes(self, dimension: int | None = None) -> list[tuple[int, float, float]]:
        return [(node.dimension, float(node.birth_value), float(node.death_value))
                for node in self.nodes
                if dimension is None or node.dimension == dimension]

    def write_event_summaries(self, path: str | Path) -> Path:
        if self.event_summaries is None:
            raise ValueError('Compute with collect_event_summary=True')
        return write_event_summaries(path, self.event_summaries)

    def cell_coordinates(self, cell: int, image_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Doubled-grid coordinate of a compressed generator (not a full cycle)."""
        if self.original_grid_indices is None:
            raise ValueError('Recompute with retain_cell_provenance=True')
        grid_shape = tuple(2 * int(n) + 1 for n in image_shape)
        if np.prod(grid_shape) != self.original_cell_count:
            raise ValueError('image_shape does not match the original cubical grid')
        return tuple(int(x) for x in np.unravel_index(int(self.original_grid_indices[cell]), grid_shape))

    def to_networkx(self):
        """Convert to a NetworkX DiGraph (requires separately installed networkx)."""
        if not self.edges_materialized:
            raise ValueError('Graph edges were streamed to CSV; load the CSV or compute with retained edges')
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install networkx to use to_networkx()") from exc
        graph = nx.DiGraph()
        for i, node in enumerate(self.nodes):
            graph.add_node(i, dimension=node.dimension, birth_cell=node.birth_cell,
                           death_cell=node.death_cell, birth_value=node.birth_value,
                           death_value=node.death_value, essential=node.essential)
        for edge in self.edges:
            graph.add_edge(edge.from_node, edge.to_node, dimension=edge.dimension,
                           coefficient=edge.coefficient)
        return graph


def compute_ppg(image: np.ndarray, *, algorithm: str = 'block',
                max_cells: int | None = None,
                retain_cell_provenance: bool = False,
                temp_dir: str | None = None,
                order_format: str = 'u32',
                reduction_storage: str = 'packed',
                log_mode: str = 'full',
                accelerator: str = 'python',
                max_relation_cache_entries: int = 100_000,
                edge_output_path: str | Path | None = None,
                node_storage: str = 'compact',
                order_builder: str = 'auto',
                metadata_storage: str = 'memory',
                collect_event_summary: bool = False,
                collect_degrees: bool = False) -> PersistencePairingGraph:
    """Compute persistence-pairing graphs over F2 for 2D/3D scalar arrays.

    ``stream-block`` uses disk-backed counting buckets, a uint32-per-cell
    contraction state, and packed compressed boundary storage. It supports
    uint8/uint16 images; ``block`` and ``lazy`` also support finite floats.

    ``max_cells`` defaults to 100 million for stream-block and 3 million for
    materialized algorithms. Providing an explicit value overrides either
    guard. The low-level constructors permit ``max_cells=None`` to disable
    their safety guard explicitly.

    ``retain_cell_provenance`` stores one uint32 original doubled-grid index
    for each compressed generator on the streaming path (optional to save RAM).
    ``temp_dir`` sets where the temporary order table is stored; disk space
    needed is approximately 4 bytes per original cubical cell.
    """
    if node_storage not in ('compact', 'python'):
        raise ValueError("node_storage must be 'compact' or 'python'")
    if algorithm not in ('stream-block', 'block', 'lazy'):
        raise ValueError("Choose algorithm='stream-block', 'block', or 'lazy'")
    if max_cells is None:
        max_cells = 100_000_000 if algorithm == 'stream-block' else 3_000_000
    if max_cells <= 0:
        raise ValueError('max_cells must be positive')
    if retain_cell_provenance and algorithm == 'lazy':
        raise ValueError('Cell provenance is available for block and stream-block')

    if algorithm == 'stream-block':
        working = compress_discrete_tensor_streaming(
            image, max_cells=max_cells,
            retain_cell_provenance=retain_cell_provenance, temp_dir=temp_dir,
            order_format=order_format, accelerator=accelerator,
            order_builder=order_builder, metadata_storage=metadata_storage)
        original_cell_count = working.original_size
        original_grid_indices = working.original_grid_indices
    else:
        complex_ = build_cubical_complex(image, max_cells=max_cells)
        original_cell_count = len(complex_)
        if algorithm == 'block':
            working = compress_value_blocks(complex_)
            original_grid_indices = working.original_grid_indices
        else:
            working = complex_
            original_grid_indices = None
    reduction = reduce_boundary(working, storage=reduction_storage, log_mode=log_mode,
                                accelerator=accelerator, temp_dir=temp_dir)
    try:
        return _extract_graph(working, reduction, algorithm, original_cell_count,
                              original_grid_indices, max_relation_cache_entries, edge_output_path,
                              node_storage, collect_event_summary, collect_degrees)
    finally:
        reduction.close()


def _extract_graph(working, reduction, algorithm, original_cell_count,
                   original_grid_indices, max_relation_cache_entries, edge_output_path,
                   node_storage, collect_event_summary=False, collect_degrees=False):
    from contextlib import nullcontext
    nodes = build_nodes(working.values, working.dimensions, reduction, storage=node_storage)
    # One counter per actual (degree, scalar death time) event. Skipped events
    # remain present with zero outgoing edges. Complexity is O(events), not O(E).
    event_counts = {} if collect_event_summary else None
    if event_counts is not None:
        for node in nodes:
            if not node.essential:
                key = (node.dimension, node.death_value)
                if key not in event_counts:
                    event_counts[key] = [0, 0]
                event_counts[key][0] += 1
    indeg = np.zeros(len(nodes), dtype=np.uint64) if collect_degrees else None
    outdeg = np.zeros(len(nodes), dtype=np.uint64) if collect_degrees else None
    edge_count = 0
    path = Path(edge_output_path) if edge_output_path is not None else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    edges = []
    with (path.open('w', newline='', encoding='utf-8') if path is not None
          else nullcontext(None)) as f:
        writer = csv.writer(f, lineterminator='\n') if f is not None else None
        if writer is not None:
            writer.writerow(('dimension', 'from_node', 'to_node', 'from_birth_value',
                             'from_death_value', 'to_birth_value', 'to_death_value', 'coefficient'))
        for edge in iter_edges(working.values, reduction, nodes,
                               max_relation_cache_entries=max_relation_cache_entries):
            edge_count += 1
            if writer is not None:
                a, b = nodes[edge.from_node], nodes[edge.to_node]
                writer.writerow((edge.dimension, edge.from_node, edge.to_node,
                                 a.birth_value, a.death_value, b.birth_value,
                                 b.death_value, edge.coefficient))
            else:
                edges.append(edge)
            if event_counts is not None:
                key = (edge.dimension, nodes[edge.from_node].death_value)
                event_counts[key][1] += 1
            if collect_degrees:
                outdeg[edge.from_node] += 1
                indeg[edge.to_node] += 1
    events = (None if event_counts is None else
              [EventSummary(d, t, counts[0], counts[1])
               for (d, t), counts in sorted(event_counts.items(),
                                            key=lambda item: (item[0][1], item[0][0]))])
    return PersistencePairingGraph(
        nodes=nodes, edges=edges, algorithm=algorithm,
        original_cell_count=original_cell_count,
        reduced_cell_count=len(working),
        cell_values=working.values, cell_dimensions=working.dimensions,
        original_grid_indices=original_grid_indices,
        edge_count=edge_count, edges_materialized=path is None,
        edge_output_path=path, event_summaries=events,
        in_degrees=indeg, out_degrees=outdeg,
    )
