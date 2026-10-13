"""Filtered original-grid cycle reconstruction through streaming block contraction."""
from itertools import product

import numpy as np
import pytest

from persistence_pairing_graph import representative_cycles_streaming, compute_ppg
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._cubical import _expanded_values
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from persistence_pairing_graph._reduction import reduce_boundary
from persistence_pairing_graph._graph import build_nodes
from persistence_pairing_graph._lift import lift_compressed_cycles


def check_cycle(image, rep):
    shape = tuple(2*s+1 for s in image.shape)
    assert np.array_equal(rep.doubled_grid_coordinates,
                          np.column_stack(np.unravel_index(rep.original_grid_indices, shape)))
    assert np.array_equal(rep.cell_centers * 2, rep.doubled_grid_coordinates)
    assert not len(boundary_old_chain(rep.original_grid_indices, shape))
    assert len(rep.original_grid_indices) > 0
    dims = (rep.doubled_grid_coordinates % 2).sum(axis=1)
    assert np.all(dims == rep.dimension)
    original_values = _expanded_values(image).ravel()[rep.original_grid_indices]
    assert np.max(original_values) <= rep.birth_value
    assert np.max(original_values) == rep.birth_value


@pytest.mark.parametrize('shape', [(3, 4), (4, 4), (3, 3, 3), (4, 4, 4)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_random_lifts_are_filtered_cycles(shape, dtype, tmp_path):
    for seed in range(7):
        image = np.random.default_rng(seed).integers(0, 7, shape).astype(dtype)
        reps = representative_cycles_streaming(image, max_features=20, temp_dir=tmp_path)
        for rep in reps:
            check_cycle(image, rep)
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('shape,dimension', [((5, 5), 1), ((5, 5, 5), 2)])
def test_cavity_and_loop_lift(shape, dimension):
    field = np.full(shape, 10, dtype=np.uint8)
    field[tuple(s//2 for s in shape)] = 90
    reps = representative_cycles_streaming(field, dimensions=(dimension,))
    assert len(reps) == 1
    assert reps[0].death_value == 90
    check_cycle(field, reps[0])


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2)])
def test_exhaustive_binary_fields_lift(shape):
    for flags in product((0, 1), repeat=int(np.prod(shape))):
        field = np.array(flags, dtype=np.uint8).reshape(shape)
        reps = representative_cycles_streaming(field, max_features=5)
        for rep in reps:
            check_cycle(field, rep)


def test_on_demand_trace_does_not_change_compressed_complex(tmp_path):
    field = np.random.default_rng(5).integers(0, 8, (4, 4, 4), dtype=np.uint8)
    baseline = compress_discrete_tensor_streaming(field, retain_cell_provenance=True)
    traced = compress_discrete_tensor_streaming(field, retain_chain_lifts=True,
                                                temp_dir=tmp_path)
    try:
        assert np.array_equal(traced.values, baseline.values)
        assert np.array_equal(traced.dimensions, baseline.dimensions)
        assert traced.offsets == baseline.offsets
        assert traced.indices == baseline.indices
        assert np.array_equal(traced.original_grid_indices, baseline.original_grid_indices)
        assert traced.lift_trace.n_pairs == (traced.original_size-len(traced))//2
    finally:
        traced.lift_trace.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('order', ['delta', 'u32'])
@pytest.mark.parametrize('metadata', ['memory', 'disk'])
def test_lift_storage_backends(order, metadata, tmp_path):
    field = np.random.default_rng(32).integers(0, 7, (3, 4, 5), dtype=np.uint8)
    reps = representative_cycles_streaming(field, order_format=order,
                                          metadata_storage=metadata,
                                          max_features=10, temp_dir=tmp_path)
    for rep in reps: check_cycle(field, rep)
    assert not list(tmp_path.iterdir())


def test_invalid_capacity_cleans_up(tmp_path):
    field = np.full((5,5), 10, dtype=np.uint8)
    field[2,2] = 90
    with pytest.raises(MemoryError):
        representative_cycles_streaming(field, max_basis_columns=1,
                                         temp_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_source_and_lifted_homology_barcode_agree():
    field = np.random.default_rng(57).integers(0, 7, (4,5,4), dtype=np.uint8)
    graph = compute_ppg(field, algorithm='stream-block')
    reps = representative_cycles_streaming(field, max_features=100)
    expected = [(n.dimension, n.birth_value, n.death_value)
                for n in graph.nodes if n.dimension in (1,2)]
    assert [(r.dimension, r.birth_value, r.death_value) for r in reps] == expected
    for rep in reps: check_cycle(field, rep)


def test_projection_of_lift_recovers_compressed_cycle():
    """Lift corrections consist only of canceled deaths, killed by projection."""
    field = np.random.default_rng(23).integers(0, 7, (4, 4, 4), dtype=np.uint8)
    compressed = compress_discrete_tensor_streaming(field, retain_chain_lifts=True)
    red = reduce_boundary(compressed, log_mode='full')
    try:
        nodes = build_nodes(compressed.values, compressed.dimensions, red)
        chosen = lift_compressed_cycles(compressed, red, nodes, max_features=20)
        all_deaths = set(map(int, compressed.lift_trace.pairs['death']))
        survivor_ids = set(map(int, compressed.original_grid_indices))
        assert all_deaths.isdisjoint(survivor_ids)
        for chain in chosen:
            seed = set(map(int, compressed.original_grid_indices[chain.compressed_cells]))
            original = set(map(int, chain.original_grid_indices))
            assert original & survivor_ids == seed
            assert original - seed <= all_deaths
    finally:
        red.close()
        compressed.lift_trace.close()


def test_numba_streaming_lifting_matches_python():
    pytest.importorskip('numba')
    field = np.random.default_rng(44).integers(0, 7, (4, 4, 4), dtype=np.uint8)
    pure = representative_cycles_streaming(field, max_features=15,
                                           accelerator='python')
    accelerated = representative_cycles_streaming(field, max_features=15,
                                                  accelerator='numba')
    assert len(pure) == len(accelerated)
    for a, b in zip(pure, accelerated):
        assert (a.dimension, a.birth_cell, a.death_cell) == (b.dimension, b.birth_cell, b.death_cell)
        assert np.array_equal(a.original_grid_indices, b.original_grid_indices)


def test_uint16_filtration_large_levels():
    field = np.full((5, 5, 5), 50000, dtype=np.uint16)
    field[2,2,2] = 60000
    rep = representative_cycles_streaming(field, dimensions=(2,))[0]
    assert rep.birth_value == 50000 and rep.death_value == 60000
    check_cycle(field, rep)


def test_lifting_cli_npz_export(tmp_path):
    tifffile = pytest.importorskip('tifffile')
    from persistence_pairing_graph._lift_cli import main
    image = np.full((5, 5), 10, dtype=np.uint8)
    image[2, 2] = 90
    source = tmp_path / 'image.tif'
    output = tmp_path / 'representatives.npz'
    tifffile.imwrite(source, image)
    assert main([str(source), '--dimensions', '1', '--output', str(output)]) == 0
    with np.load(output) as artifact:
        assert artifact['dimensions'].tolist() == [1]
        assert artifact['birth_values'].tolist() == [10]
        assert artifact['death_values'].tolist() == [90.0]
        assert artifact['support_offsets'][0] == 0
        assert artifact['support_offsets'][-1] == len(artifact['original_grid_cell_ids'])
        assert len(boundary_old_chain(artifact['original_grid_cell_ids'], (11, 11))) == 0
