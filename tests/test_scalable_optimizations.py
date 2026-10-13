"""Differential tests for three memory/runtime optimizations.

All comparisons preserve canonical filtration-order indices and graph edge
ordering, not just the persistence diagram. No external Rust toolchain needed.
"""
from collections import Counter
from array import array

import numpy as np
import pytest

from persistence_pairing_graph import CompactNodeTable, compute_ppg
from persistence_pairing_graph._streaming import _project, _CURRENT, _FINAL_INLINE, _FINAL_ARENA
from persistence_pairing_graph._reduction import xor_sorted


@pytest.mark.parametrize('shape', [(2, 3), (3, 4), (2, 3, 4), (3, 3, 4)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('seed', [0, 2, 14])
def test_compact_vs_materialized_nodes_and_edges(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    # Small range forces ties and many zero-length cancellations.
    image = rng.integers(0, 6, shape, dtype=dtype)
    compressed_reference = compute_ppg(image, algorithm='block', node_storage='python')
    for storage in ('packed', 'disk'):
        graph = compute_ppg(image, algorithm='stream-block',
                            reduction_storage=storage, node_storage='compact')
        assert isinstance(graph.nodes, CompactNodeTable)
        assert graph.nodes == compressed_reference.nodes
        assert list(graph.nodes) == compressed_reference.nodes
        assert graph.edges == compressed_reference.edges
        assert graph.edge_count == len(graph.edges)
        assert graph.nodes[:] == compressed_reference.nodes
        assert graph.nodes[-1] == compressed_reference.nodes[-1]
        assert graph.nodes.nbytes == 8 * len(graph.nodes)


def test_compact_and_python_nodes_on_float_fields():
    a = np.array([[0.5, 1., 0.5, 2.], [0., 3., 1., 0.5],
                  [2., 1., 3., 2.]], dtype=np.float64)
    for algorithm in ('lazy', 'block'):
        base = compute_ppg(a, algorithm=algorithm, node_storage='python')
        packed = compute_ppg(a, algorithm=algorithm, node_storage='compact')
        assert packed.nodes == base.nodes
        assert packed.edges == base.edges


def test_compact_invalid_options_and_empty_survivors():
    a = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match='node_storage'):
        compute_ppg(a, node_storage='unknown')
    g = compute_ppg(a, algorithm='stream-block', node_storage='compact')
    assert len(g.nodes) == 1
    assert g.nodes[0].essential
    assert g.nodes[0].death_cell is None
    assert g.nodes.nbytes == 8


@pytest.mark.parametrize('seed', [1, 9, 27])
def test_sparse_xor_sorted_matches_set_reference(seed):
    rng = np.random.default_rng(seed)
    for size in [0, 1, 2, 4, 6, 12, 36, 128, 256, 1024]:
        a = sorted(map(int, rng.choice(8000, size=size, replace=False)))
        b = sorted(map(int, rng.choice(8000, size=size, replace=False)))
        assert xor_sorted(a, b) == sorted(set(a) ^ set(b))
        assert xor_sorted(b, a) == xor_sorted(a, b)


def test_simple_projection_and_fallback_paths():
    meta = np.array([3, 2, 1, _FINAL_INLINE | 1, _FINAL_ARENA | 0,
                     _CURRENT | 0], dtype=np.uint32)
    current = array('I', [1, 0])
    finalized = array('I', [1, 0])
    assert _project([0, 1, 2], meta, current, finalized) == [0, 2]
    assert _project([3, 4], meta, current, finalized) == []  # inline and arena cancel
    assert _project([5], meta, current, finalized) == [0]
    with pytest.raises(RuntimeError, match='unprocessed'):
        _project([0], np.zeros(1, dtype=np.uint32), current, finalized)


def test_event_pruning_is_identical_for_complex_3d():
    from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
    from persistence_pairing_graph._reduction import reduce_boundary
    from persistence_pairing_graph._graph import build_nodes, iter_edges
    rng = np.random.default_rng(881)
    arrays = [rng.integers(0, 32, (4, 5, 6), dtype=np.uint8),
              (np.indices((5, 5, 5)).sum(axis=0) % 2).astype(np.uint8)]
    for arr in arrays:
        c = compress_discrete_tensor_streaming(arr)
        r = reduce_boundary(c)
        try:
            for storage in ('compact', 'python'):
                nodes = build_nodes(c.values, c.dimensions, r, storage=storage)
                pruned = list(iter_edges(c.values, r, nodes, skip_empty_survivor_events=True))
                exhaustive = list(iter_edges(c.values, r, nodes, skip_empty_survivor_events=False))
                assert pruned == exhaustive
        finally:
            r.close()


def test_csv_parity_with_compact_nodes(tmp_path):
    from persistence_pairing_graph import export_csv
    a = np.random.default_rng(45).integers(0, 30, (4, 4, 4), dtype=np.uint8)
    ref = compute_ppg(a, algorithm='stream-block', node_storage='python')
    optimized = compute_ppg(a, algorithm='stream-block', node_storage='compact')
    paths_ref = export_csv(ref, 'case.tif', output_dir=tmp_path / 'ref')
    paths_opt = export_csv(optimized, 'case.tif', output_dir=tmp_path / 'compact')
    for x, y in zip(paths_ref[:2], paths_opt[:2]):
        assert x.read_bytes() == y.read_bytes()
