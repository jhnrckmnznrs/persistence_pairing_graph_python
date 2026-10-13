"""Independent topology, exact original-grid chains, event summaries, counting order."""
import csv
from collections import Counter

import numpy as np
import pytest

from persistence_pairing_graph import (
    betti_curves, compute_ppg, representative_cycles,
)
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming


def _interval_betti(graph, t, dimension):
    return sum(node.dimension == dimension and node.birth_value <= t < node.death_value
               for node in graph.nodes)


@pytest.mark.parametrize('shape', [(3, 3), (4, 4), (3, 3, 3), (4, 4, 4)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_independent_betti_agrees_with_canonical_reduction(shape, dtype):
    for seed in range(3):
        image = np.random.default_rng(seed).integers(0, 6, shape).astype(dtype)
        curves = betti_curves(image)
        graph = compute_ppg(image, algorithm='block')
        for i, level in enumerate(curves.levels):
            for dimension, values in enumerate((curves.beta0, curves.beta1, curves.beta2)):
                assert _interval_betti(graph, int(level), dimension) == int(values[i])
            assert curves.beta0[i] - curves.beta1[i] + curves.beta2[i] == curves.euler[i]
        assert tuple(curves.beta0[-1:] == [1]) == (True,)
        assert int(curves.beta1[-1]) == 0
        assert int(curves.beta2[-1]) == 0


@pytest.mark.parametrize('shape', [(3, 4), (3, 3, 3)])
def test_numba_betti_agrees_with_python(shape):
    pytest.importorskip('numba')
    image = np.random.default_rng(11).integers(0, 5, shape, dtype=np.uint8)
    a = betti_curves(image, accelerator='python')
    b = betti_curves(image, accelerator='numba')
    for label in ('levels', 'beta0', 'beta1', 'beta2', 'euler'):
        assert np.array_equal(getattr(a, label), getattr(b, label))


def test_h1_loop_and_3d_cavity_examples(tmp_path):
    field2d = np.full((5, 5), 10, dtype=np.uint8)
    field2d[2, 2] = 90
    curves2 = betti_curves(field2d)
    assert curves2.beta1.tolist() == [1, 0]
    field3d = np.full((5, 5, 5), 10, dtype=np.uint8)
    field3d[2, 2, 2] = 90
    curves3 = betti_curves(field3d)
    assert curves3.beta2.tolist() == [1, 0]
    output = curves3.to_csv(tmp_path / 'betti.csv')
    with output.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]['beta2'] == '1'


@pytest.mark.parametrize('storage', ['packed', 'python'])
def test_cycle_reconstruction_exact_on_original_grid(storage):
    image = np.full((5, 5), 10, dtype=np.uint8)
    image[2, 2] = 90
    chains = representative_cycles(image, dimensions=(1,), reduction_storage=storage)
    assert len(chains) == 1
    chain = chains[0]
    assert chain.dimension == 1
    assert chain.birth_value == 10 and chain.death_value == 90
    assert chain.doubled_grid_coordinates.shape[1] == 2
    assert len(chain.cells) >= 4
    assert np.array_equal(chain.cell_centers * 2, chain.doubled_grid_coordinates)
    assert chain.cells.shape == chain.original_grid_indices.shape


def test_h2_cycle_reconstruction():
    image = np.full((3, 3, 3), 5, dtype=np.uint8)
    image[1, 1, 1] = 30
    chains = representative_cycles(image, dimensions=(2,), max_cells=50000)
    assert len(chains) == 1
    assert chains[0].death_value == 30
    assert len(chains[0].cells) >= 6
    assert chains[0].doubled_grid_coordinates.shape[1] == 3


def test_exact_event_summary_and_degrees_without_retaining_edges(tmp_path):
    image = np.random.default_rng(18).integers(0, 9, (6, 6), dtype=np.uint8)
    full = compute_ppg(image, algorithm='lazy', node_storage='python',
                       collect_event_summary=True, collect_degrees=True)
    streamed = compute_ppg(image, algorithm='lazy', node_storage='compact',
                           edge_output_path=tmp_path / 'edges.csv',
                           collect_event_summary=True, collect_degrees=True)
    assert streamed.edges == [] and not streamed.edges_materialized
    assert streamed.edge_count == full.edge_count
    assert streamed.event_summaries == full.event_summaries
    assert sum(event.outgoing_edges for event in streamed.event_summaries) == streamed.edge_count
    assert int(sum(streamed.out_degrees)) == streamed.edge_count
    assert int(sum(streamed.in_degrees)) == streamed.edge_count
    assert np.array_equal(streamed.out_degrees, full.out_degrees)
    assert np.array_equal(streamed.in_degrees, full.in_degrees)
    counts = Counter((e.dimension, full.nodes[e.from_node].death_value) for e in full.edges)
    for event in streamed.event_summaries:
        assert event.outgoing_edges == counts[(event.dimension, event.death_value)]
        assert event.dying_classes > 0
        assert event.mean_outgoing_edges == event.outgoing_edges / event.dying_classes
    output = streamed.write_event_summaries(tmp_path / 'events.csv')
    with output.open() as stream:
        assert len(list(csv.DictReader(stream))) == len(streamed.event_summaries)


def test_provenance_cell_coordinates():
    image = np.array([[0, 5], [7, 1]], dtype=np.uint8)
    graph = compute_ppg(image, algorithm='block', retain_cell_provenance=True)
    for j, old in enumerate(graph.original_grid_indices):
        assert graph.cell_coordinates(j, image.shape) == tuple(np.unravel_index(int(old), (5, 5)))
    with pytest.raises(ValueError, match='image_shape'):
        graph.cell_coordinates(0, (3, 3))


@pytest.mark.parametrize('shape', [(4, 7), (4, 5, 3)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_compiled_counting_order_matches_canonical_sort(shape, dtype):
    pytest.importorskip('numba')
    image = np.random.default_rng(12).integers(0, 16, shape).astype(dtype)
    sort = compress_discrete_tensor_streaming(image, accelerator='python', order_builder='sort',
                                              retain_cell_provenance=True)
    counted = compress_discrete_tensor_streaming(image, accelerator='python', order_builder='counting',
                                                 retain_cell_provenance=True)
    for label in ('values', 'dimensions', 'original_grid_indices'):
        assert np.array_equal(getattr(sort, label), getattr(counted, label))
    assert sort.offsets == counted.offsets
    assert sort.indices == counted.indices


def test_invalid_counting_delta_order_rejected():
    with pytest.raises(ValueError, match='requires'):
        compress_discrete_tensor_streaming(np.zeros((3, 3), dtype=np.uint8),
                                           order_builder='counting', order_format='delta')


@pytest.mark.parametrize('order_format', ['delta', 'u32'])
def test_disk_metadata_exact_and_scratch_cleanup(tmp_path, order_format):
    image = np.random.default_rng(19).integers(0, 9, (3, 4, 5), dtype=np.uint8)
    memory = compress_discrete_tensor_streaming(image, order_format=order_format,
                                                metadata_storage='memory',
                                                retain_cell_provenance=True, temp_dir=tmp_path)
    disk = compress_discrete_tensor_streaming(image, order_format=order_format,
                                              metadata_storage='disk',
                                              retain_cell_provenance=True, temp_dir=tmp_path)
    for attr in ('values', 'dimensions', 'original_grid_indices'):
        assert np.array_equal(getattr(memory, attr), getattr(disk, attr))
    assert memory.offsets == disk.offsets
    assert memory.indices == disk.indices
    assert list(tmp_path.iterdir()) == []


def test_betti_validation_detects_missing_barcode():
    from persistence_pairing_graph import validate_ppg_betti
    image = np.random.default_rng(9).integers(0, 7, (4, 4, 4), dtype=np.uint8)
    graph = compute_ppg(image, algorithm='stream-block')
    curves = validate_ppg_betti(image, graph)
    assert int(curves.beta0[-1]) == 1
    # Deliberately corrupt one barcode for a negative validation check.
    graph.nodes = []
    with pytest.raises(AssertionError, match='Betti mismatch'):
        validate_ppg_betti(image, graph)


def test_event_summary_cli_smoke(tmp_path):
    import tifffile
    from persistence_pairing_graph.__main__ import main
    image = np.random.default_rng(3).integers(0, 9, (5, 5), dtype=np.uint8)
    path = tmp_path / 'input.tif'
    tifffile.imwrite(path, image)
    assert main([str(path), '--algorithm', 'stream-block', '--event-summary',
                 '--degrees', '--stream-edges']) == 0
    assert (tmp_path / 'input_ppg_event_summary.csv').exists()
    assert (tmp_path / 'input_ppg_degrees.csv').exists()
    assert (tmp_path / 'input_persistence_pairing_graph_edges.csv').exists()


def test_betti_cli_smoke(tmp_path):
    import tifffile
    from persistence_pairing_graph._betti_cli import main
    path = tmp_path / 'b.tiff'
    tifffile.imwrite(path, np.ones((3, 3), dtype=np.uint8))
    assert main([str(path), '--output', str(tmp_path / 'b.csv')]) == 0
    with (tmp_path / 'b.csv').open() as stream:
        data = list(csv.DictReader(stream))
    assert len(data) == 1 and data[0]['beta0'] == '1'


def test_multiscale_pool_partial_boundaries_and_curve_shapes():
    from persistence_pairing_graph import pool_scalar_field, multiscale_betti_curves
    image = np.arange(15, dtype=np.uint8).reshape((3, 5))
    assert np.array_equal(pool_scalar_field(image, 2, reducer='min'),
                          [[0, 2, 4], [10, 12, 14]])
    assert np.array_equal(pool_scalar_field(image, 2, reducer='max'),
                          [[6, 8, 9], [11, 13, 14]])
    assert np.array_equal(pool_scalar_field(image, 1), image)
    profiles = multiscale_betti_curves(image, factors=(1, 2, 4))
    assert [p.shape for p in profiles] == [(3, 5), (2, 3), (1, 2)]
    assert all(int(p.curves.beta0[-1]) == 1 for p in profiles)
    with pytest.raises(ValueError):
        multiscale_betti_curves(image, factors=(1, 1))
    with pytest.raises(ValueError):
        pool_scalar_field(image, 0)


def test_ppg_perturbation_probe_reproducible_and_bounded(tmp_path):
    from persistence_pairing_graph import ppg_perturbation_probe
    image = np.random.default_rng(8).integers(1, 15, (4, 5), dtype=np.uint8)
    one = ppg_perturbation_probe(image, repeats=2, amplitude=1, seed=111)
    two = ppg_perturbation_probe(image, repeats=2, amplitude=1, seed=111)
    assert one.samples == two.samples
    assert len(one.samples) == 3
    assert all(sum(s.nodes_by_dimension) == s.nodes for s in one.samples)
    assert all(sum(s.edges_by_dimension) == s.edges for s in one.samples)
    assert all(s.finite_total_persistence >= 0 for s in one.samples)
    no_noise = ppg_perturbation_probe(image, repeats=2, amplitude=0)
    assert no_noise.samples[0].nodes == no_noise.samples[1].nodes
    assert no_noise.samples[0].edges == no_noise.samples[1].edges
    assert no_noise.to_csv(tmp_path / 'probe.csv').exists()
    with pytest.raises(MemoryError):
        ppg_perturbation_probe(image, max_voxels=2)


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2)])
def test_exhaustive_binary_topology_matches_reduction(shape):
    from persistence_pairing_graph import validate_ppg_betti
    count = int(np.prod(shape))
    for configuration in range(1 << count):
        image = np.asarray([(configuration >> i) & 1 for i in range(count)],
                           dtype=np.uint8).reshape(shape)
        graph = compute_ppg(image, algorithm='block')
        validate_ppg_betti(image, graph)
