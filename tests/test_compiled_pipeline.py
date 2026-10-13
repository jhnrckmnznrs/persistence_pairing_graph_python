"""Strict optional Numba parity: cells, canonical reduction and graph edges."""
from __future__ import annotations

from array import array
import numpy as np
import pytest

numba = pytest.importorskip('numba')

from persistence_pairing_graph import compute_ppg
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from persistence_pairing_graph._reduction import reduce_boundary
from persistence_pairing_graph._compiled_contraction import emit_survivors


@pytest.mark.parametrize('shape', [(2,3),(4,5),(3,4,5),(5,5,5)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('seed', [1,3,17])
def test_compiled_and_python_exact_streaming_and_graph(shape, dtype, seed):
    arr = np.random.default_rng(seed).integers(0, 8, shape, dtype=dtype)
    ref = compress_discrete_tensor_streaming(arr, retain_cell_provenance=True)
    compiled = compress_discrete_tensor_streaming(arr, retain_cell_provenance=True,
                                                   accelerator='numba')
    assert np.array_equal(ref.values, compiled.values)
    assert np.array_equal(ref.dimensions, compiled.dimensions)
    assert np.array_equal(ref.original_grid_indices, compiled.original_grid_indices)
    assert ref.offsets == compiled.offsets
    assert ref.indices == compiled.indices
    reduced_ref = reduce_boundary(ref)
    reduced_compiled = reduce_boundary(compiled, accelerator='numba')
    try:
        assert np.array_equal(reduced_ref.low, reduced_compiled.low)
        for j in range(len(ref)):
            assert reduced_ref.reduced_columns.get(j) == reduced_compiled.reduced_columns.get(j)
            assert reduced_ref.operations.get(j) == reduced_compiled.operations.get(j)
    finally:
        reduced_ref.close()
        reduced_compiled.close()
    graph_ref = compute_ppg(arr, algorithm='stream-block')
    graph_compiled = compute_ppg(arr, algorithm='stream-block', accelerator='numba')
    assert graph_ref.nodes == graph_compiled.nodes
    assert graph_ref.edges == graph_compiled.edges


@pytest.mark.parametrize('log_mode', ['full', 'selective'])
@pytest.mark.parametrize('case', ['checkerboard','random','torus','lattice','smooth'])
def test_structured_3d_canonical_reduction(case, log_mode):
    from benchmarks.profile_structured import make_volume
    volume = make_volume(case, 9)
    compressed = compress_discrete_tensor_streaming(volume)
    base = reduce_boundary(compressed, log_mode=log_mode)
    jit = reduce_boundary(compressed, log_mode=log_mode, accelerator='numba')
    try:
        assert np.array_equal(base.low, jit.low)
        assert base.reduced_columns == jit.reduced_columns or all(
            base.reduced_columns.get(j) == jit.reduced_columns.get(j)
            for j in range(len(compressed)))
        for j in range(len(compressed)):
            assert base.operations.get(j) == jit.operations.get(j)
    finally:
        base.close()
        jit.close()


def test_compiled_emitter_handles_final_inline_and_arena():
    # 1 x 3 row of cells in doubled coordinates, middle edge (id 3) emits
    # with a cancelled vertex replaced by a lower active vertex.
    # Synthetic metadata is deliberately minimal, bypassing image plumbing.
    meta = np.zeros(5, dtype=np.uint32)
    meta[0] = 3  # compressed 0
    meta[2] = 2 << 30 | 1 # inline to cell 0
    meta[4] = 3 # another old active generator mapped to compressed 0 (synthetic F2 test)
    meta[3] = 1 # new edge
    ids, offsets, data = emit_survivors(np.array([3], dtype=np.uint32),meta,
        np.empty(0,dtype=np.uint32), (5,), (1,), 1)
    assert ids.tolist() == [3]
    assert offsets.tolist() == [0,0] # 0 and its duplicate via inline cancel
    assert data.tolist() == []
    assert meta[3] == 4

@pytest.mark.parametrize('order_format', ['u32', 'delta'])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_compiled_delta_order_and_streaming_edges(tmp_path, order_format, dtype):
    x = np.random.default_rng(733).integers(0, 18, (5,6,7), dtype=dtype)
    common = dict(algorithm='stream-block', order_format=order_format,
                  retain_cell_provenance=True)
    graph_ref = compute_ppg(x, **common)
    path = tmp_path / 'edges.csv'
    graph_compiled = compute_ppg(x, accelerator='numba', edge_output_path=path,
                                 **common)
    assert graph_compiled.nodes == graph_ref.nodes
    assert graph_compiled.edges_materialized is False
    assert graph_compiled.edge_count == len(graph_ref.edges)
    assert np.array_equal(graph_compiled.original_grid_indices,
                          graph_ref.original_grid_indices)
    from persistence_pairing_graph._io import export_csv
    baseline_path = tmp_path / 'reference'
    export_csv(graph_ref, 'test.tif', output_dir=baseline_path)
    assert path.read_bytes() == (baseline_path / 'test_persistence_pairing_graph_edges.csv').read_bytes()


@pytest.mark.parametrize('seed', [4, 7, 22])
def test_compiled_selective_canonical_graph(seed):
    x = np.random.default_rng(seed).integers(0,10,(4,5,6),dtype=np.uint8)
    baseline = compute_ppg(x,algorithm='stream-block')
    accelerated = compute_ppg(x,algorithm='stream-block',log_mode='selective', accelerator='numba')
    assert baseline.nodes == accelerated.nodes
    assert baseline.edges == accelerated.edges

@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('seed', [16, 733, 8823])
def test_compiled_relation_arena_can_grow_between_value_blocks(dtype, seed):
    """Regression: JIT must not hold an exported buffer during array growth."""
    image = np.random.default_rng(seed).integers(0, 32, (6,7,8), dtype=dtype)
    baseline = compress_discrete_tensor_streaming(image, retain_cell_provenance=True)
    accelerated = compress_discrete_tensor_streaming(image, accelerator='numba',
                                                     retain_cell_provenance=True)
    assert np.array_equal(baseline.values, accelerated.values)
    assert baseline.offsets == accelerated.offsets
    assert baseline.indices == accelerated.indices
    assert np.array_equal(baseline.original_grid_indices,
                          accelerated.original_grid_indices)
