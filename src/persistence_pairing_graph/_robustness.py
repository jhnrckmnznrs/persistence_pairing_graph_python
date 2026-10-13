"""Perturbation probes for aggregate PPG observables.

Do not match indistinguishable persistence points or claim that a specific
basis-dependent edge is stable across filtrations. This diagnostic samples
observable distributions only.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import csv

import numpy as np


@dataclass(frozen=True, slots=True)
class PerturbationSample:
    replicate: int
    nodes: int
    edges: int
    nodes_by_dimension: tuple[int, ...]
    edges_by_dimension: tuple[int, ...]
    finite_total_persistence: float


@dataclass(slots=True)
class PerturbationReport:
    amplitude: int
    seed: int
    samples: list[PerturbationSample]

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(('replicate', 'nodes', 'edges', 'nodes_by_dimension',
                             'edges_by_dimension', 'finite_total_persistence'))
            for s in self.samples:
                writer.writerow((s.replicate, s.nodes, s.edges,
                                 ';'.join(map(str, s.nodes_by_dimension)),
                                 ';'.join(map(str, s.edges_by_dimension)),
                                 s.finite_total_persistence))
        return path


def ppg_perturbation_probe(image: np.ndarray, *, repeats: int = 5,
                           amplitude: int = 1, seed: int = 0,
                           max_voxels: int = 100_000,
                           algorithm: str = 'stream-block',
                           accelerator: str = 'python') -> PerturbationReport:
    """Sample graph-level sensitivity to independent bounded intensity noise.

    Replicate 0 is the unmodified image. Other replicates add independent
    uniformly distributed integer noise, clipped to the scalar dtype limits.
    Reports node and edge counts by dimension and finite total persistence.
    No cross-run persistence-point matching is attempted; individual-edge
    reproducibility or statistical confidence intervals are NOT established.
    """
    from . import compute_ppg
    image = np.asarray(image)
    if image.ndim not in (2, 3) or not all(image.shape) or image.dtype not in (np.uint8, np.uint16):
        raise ValueError('Expected nonempty 2D/3D uint8/uint16 image')
    if repeats < 1 or amplitude < 0 or amplitude > 65535:
        raise ValueError('repeats must be >=1; amplitude must be between 0 and 65535')
    if image.size > max_voxels:
        raise MemoryError('Perturbation probe exceeds max_voxels')
    rng = np.random.default_rng(seed)
    max_value = np.iinfo(image.dtype).max
    samples = []
    with TemporaryDirectory(prefix='ppg_perturbation_') as tmp:
        for replicate in range(repeats + 1):
            if replicate:
                noise = rng.integers(-amplitude, amplitude + 1, size=image.shape, dtype=np.int32)
                perturbed = np.clip(image.astype(np.int32) + noise, 0, max_value).astype(image.dtype)
            else:
                perturbed = image
            graph = compute_ppg(
                perturbed, algorithm=algorithm, accelerator=accelerator,
                collect_event_summary=True,
                edge_output_path=Path(tmp) / 'edges.csv',
            )
            node_counts = [0] * image.ndim
            total_persistence = 0.0
            for node in graph.nodes:
                node_counts[node.dimension] += 1
                if not node.essential:
                    total_persistence += node.persistence
            edge_counts = [0] * image.ndim
            for event in graph.event_summaries:
                edge_counts[event.dimension] += event.outgoing_edges
            samples.append(PerturbationSample(
                replicate, len(graph.nodes), graph.edge_count,
                tuple(node_counts), tuple(edge_counts), total_persistence))
    return PerturbationReport(amplitude, seed, samples)
