import csv

import numpy as np
import pytest

from persistence_pairing_graph import build_cubical_complex, compute_ppg, export_csv, load_image
from persistence_pairing_graph._reduction import compress_value_blocks, reduce_boundary, xor_sorted
from persistence_pairing_graph._graph import build_nodes


@pytest.fixture
def pants():
    return np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 0],
        [0, 30, 10, 10, 10, 30, 20, 20, 20, 30, 0],
        [0, 30, 10, 90, 10, 30, 20, 80, 20, 30, 0],
        [0, 30, 10, 10, 10, 30, 20, 20, 20, 30, 0],
        [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)


def signatures(graph):
    return sorted((n.dimension, n.birth_value, n.death_value) for n in graph.nodes)


def edge_signatures(graph):
    return sorted((e.dimension, graph.nodes[e.from_node].birth_value,
                   graph.nodes[e.from_node].death_value,
                   graph.nodes[e.to_node].birth_value,
                   graph.nodes[e.to_node].death_value) for e in graph.edges)


def test_pair_of_pants_reference_lazy(pants):
    graph = compute_ppg(pants, algorithm='lazy')
    assert edge_signatures(graph) == sorted([
        (0, 10, 30, 0, float('inf')),
        (0, 20, 30, 0, float('inf')),
        (1, 20, 30, 0, 90),
        (1, 20, 30, 10, 80),
        (1, 10, 80, 0, 90),
    ])


def test_same_positive_barcode_for_both_backends(pants):
    assert signatures(compute_ppg(pants, algorithm='block')) == signatures(compute_ppg(pants, algorithm='lazy'))


@pytest.mark.parametrize('image', [
    np.array([[7]], dtype=np.uint8),
    np.array([[10, 3]], dtype=np.uint8),
    np.array([[0, 2], [1, 3]], dtype=np.uint8),
    np.array([[[0, 1], [1, 0]], [[2, 3], [1, 2]]], dtype=np.uint8),
])
def test_grid_and_double_boundary_zero(image):
    complex_ = build_cubical_complex(image)
    assert complex_.grid_shape == tuple(2 * d + 1 for d in image.shape)
    for j in range(len(complex_)):
        b = complex_.boundary(j)
        assert all(i < j for i in b)
        assert all(complex_.values[i] <= complex_.values[j] for i in b)
        assert all(complex_.dimensions[i] + 1 == complex_.dimensions[j] for i in b)
        total = []
        for i in b:
            total = xor_sorted(total, complex_.boundary(i))
        assert total == []


def test_shared_edge_minimum_filtration():
    complex_ = build_cubical_complex(np.array([[10, 3]], dtype=np.uint8))
    old = np.ravel_multi_index((1, 2), complex_.grid_shape)
    pos = int(complex_.old_to_new[old])
    assert complex_.values[pos] == 3
    assert complex_.dimensions[pos] == 1


def test_constant_fields_only_one_essential():
    for shape in [(1, 1), (2, 3), (1, 2, 2)]:
        image = np.full(shape, 7, dtype=np.uint16)
        result = compute_ppg(image, algorithm='block')
        assert [(n.dimension, n.essential, n.birth_value) for n in result.nodes] == [(0, True, 7)]
        assert not result.edges
        assert result.reduced_cell_count == 1


def test_compressed_boundaries_strictly_lower():
    image = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.uint8)
    compressed = compress_value_blocks(build_cubical_complex(image))
    for j, boundary in enumerate(compressed.boundaries):
        assert all(i < j and compressed.values[i] < compressed.values[j] for i in boundary)


@pytest.mark.parametrize('shape', [(2, 2), (3, 3), (2, 3), (2, 2, 2), (2, 2, 3)])
def test_random_small_barcode_agreement(shape):
    for seed in range(5):
        image = np.random.default_rng(seed).integers(0, 5, size=shape, dtype=np.uint8)
        left = compute_ppg(image, algorithm='lazy')
        right = compute_ppg(image, algorithm='block')
        assert signatures(left) == signatures(right)
        assert sum(n.essential for n in left.nodes) == 1


def test_reduction_operation_log_matches_explicit_v():
    image = np.array([[0, 2, 1], [3, 1, 4]], dtype=np.uint8)
    complex_ = build_cubical_complex(image)
    red = reduce_boundary(complex_)
    columns = [complex_.boundary(j) for j in range(len(complex_))]
    v = [[j] for j in range(len(complex_))]
    low_to_col = {}
    for j in range(len(complex_)):
        while columns[j] and columns[j][-1] in low_to_col:
            i = low_to_col[columns[j][-1]]
            columns[j] = xor_sorted(columns[j], columns[i])
            v[j] = xor_sorted(v[j], v[i])
        if columns[j]:
            low_to_col[columns[j][-1]] = j
        assert columns[j] == red.reduced_columns.get(j, [])
    for j, reduced_col in red.reduced_columns.items():
        coords = sorted(reduced_col)
        basis = []
        while coords:
            k = coords[-1]
            basis.append(k)
            coords = xor_sorted(coords, v[k])
        basis.reverse()
        from_log = set(reduced_col)
        for target in reduced_col:
            for source in red.operations.get(target, ()):
                if source in from_log:
                    from_log.remove(source)
                else:
                    from_log.add(source)
        assert sorted(from_log) == basis


def test_exports_and_tiff(tmp_path, pants):
    tifffile = pytest.importorskip('tifffile')
    path = tmp_path / 'image.tif'
    tifffile.imwrite(path, pants)
    loaded = load_image(path)
    assert np.array_equal(loaded, pants)
    result = compute_ppg(loaded, algorithm='block')
    n, e, p = export_csv(result, path, export_cell_provenance=True)
    assert n.exists() and e.exists() and p.exists()
    with n.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.nodes)
    with p.open() as f:
        provenance = list(csv.DictReader(f))
    assert len(provenance) == result.reduced_cell_count


def test_invalid():
    with pytest.raises(ValueError):
        compute_ppg(np.array([1, 2], dtype=np.uint8))
    with pytest.raises(ValueError):
        compute_ppg(np.array([[float('nan')]]))
    with pytest.raises(TypeError):
        compute_ppg(np.array([[1]], dtype=np.int32))
    with pytest.raises(MemoryError):
        compute_ppg(np.ones((8, 8, 8), dtype=np.uint8), max_cells=100)


def test_networkx_graph(pants):
    nx = pytest.importorskip('networkx')
    result = compute_ppg(pants)
    g = result.to_networkx()
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == len(result.nodes)
    assert g.number_of_edges() == len(result.edges)


def test_uint16_and_float_scalar_fields():
    array = np.array([[0, 4], [6, 2]], dtype=np.uint16)
    result = compute_ppg(array)
    assert any(n.essential for n in result.nodes)
    for dtype in [np.float32, np.float64]:
        f = array.astype(dtype) / 3
        result = compute_ppg(f)
        assert len(result.nodes) > 0


def test_multipage_tiff(tmp_path):
    tifffile = pytest.importorskip('tifffile')
    stack = np.arange(3 * 5 * 7, dtype=np.uint16).reshape((3, 5, 7))
    path = tmp_path / 'stack.tif'
    tifffile.imwrite(path, stack, photometric='minisblack')
    assert np.array_equal(load_image(path), stack)


def test_graph_edges_match_between_algorithms_on_small_examples():
    for shape in [(3, 3), (3, 4), (2, 2, 2)]:
        for seed in range(6):
            a = np.random.default_rng(seed).integers(0, 7, size=shape, dtype=np.uint8)
            assert edge_signatures(compute_ppg(a, algorithm='lazy')) == edge_signatures(compute_ppg(a, algorithm='block'))

# The streaming constructor must reproduce the deterministic value-compressed
# chain complex, not merely the persistence barcode.
@pytest.mark.parametrize('shape', [(1, 1), (2, 3), (4, 5), (3, 3, 3), (2, 3, 4)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_streaming_compressed_complex_identical_to_materialized(shape, dtype, tmp_path):
    from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
    for seed in (0, 2, 5):
        volume = np.random.default_rng(seed).integers(0, 20, shape, dtype=dtype)
        materialized = compress_value_blocks(build_cubical_complex(volume))
        streaming = compress_discrete_tensor_streaming(
            volume, retain_cell_provenance=True, temp_dir=tmp_path)
        assert np.array_equal(streaming.values, materialized.values)
        assert np.array_equal(streaming.dimensions, materialized.dimensions)
        assert np.array_equal(streaming.original_grid_indices, materialized.original_grid_indices)
        assert [streaming.boundary(j) for j in range(len(streaming))] == materialized.boundaries
        assert not list(tmp_path.iterdir()), 'Streaming order scratch file was not removed'


def test_streaming_pair_of_pants_edges_and_provenance(pants):
    from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
    from persistence_pairing_graph import compute_ppg
    streaming = compute_ppg(pants, algorithm='stream-block', retain_cell_provenance=True)
    materialized = compute_ppg(pants, algorithm='block')
    assert streaming.nodes == materialized.nodes
    assert streaming.edges == materialized.edges
    assert np.array_equal(streaming.original_grid_indices, materialized.original_grid_indices)
    assert streaming.reduced_cell_count < streaming.original_cell_count
    assert [
        (edge.dimension,
         streaming.nodes[edge.from_node].birth_value,
         streaming.nodes[edge.from_node].death_value,
         streaming.nodes[edge.to_node].birth_value,
         streaming.nodes[edge.to_node].death_value)
        for edge in streaming.edges
    ] == [
        (0, 10, 30, 0, float('inf')),
        (0, 20, 30, 0, float('inf')),
        (1, 20, 30, 0, 90),
        (1, 20, 30, 10, 80),
        (1, 10, 80, 0, 90),
    ]


def test_streaming_provenance_optional_and_export(pants, tmp_path):
    from persistence_pairing_graph import compute_ppg, export_csv
    graph = compute_ppg(pants, algorithm='stream-block')
    assert graph.original_grid_indices is None
    with pytest.raises(ValueError, match='provenance'):
        export_csv(graph, tmp_path / 'pants.tif', export_cell_provenance=True)
    graph = compute_ppg(pants, algorithm='stream-block', retain_cell_provenance=True)
    nodes_path, edges_path, provenance_path = export_csv(
        graph, tmp_path / 'pants.tif', export_cell_provenance=True)
    assert all(p.exists() for p in (nodes_path, edges_path, provenance_path))
    with provenance_path.open() as file:
        assert len(list(csv.DictReader(file))) == len(graph.cell_values)


def test_streaming_uint16_sparse_value_levels():
    from persistence_pairing_graph import compute_ppg
    a = np.array([[[0, 65535], [1, 128]], [[32768, 1], [128, 65535]]], dtype=np.uint16)
    stream = compute_ppg(a, algorithm='stream-block')
    block = compute_ppg(a, algorithm='block')
    assert stream.nodes == block.nodes
    assert stream.edges == block.edges


def test_streaming_constant_volume_and_guard():
    from persistence_pairing_graph import compute_ppg
    a = np.full((6, 7, 8), 22, dtype=np.uint8)
    graph = compute_ppg(a, algorithm='stream-block')
    assert graph.reduced_cell_count == 1
    assert [(node.dimension, node.birth_value, node.essential) for node in graph.nodes] == [(0, 22, True)]
    assert graph.edges == []
    with pytest.raises(MemoryError):
        compute_ppg(a, algorithm='stream-block', max_cells=10)
    with pytest.raises(TypeError, match='uint8/uint16'):
        compute_ppg(a.astype(np.float64), algorithm='stream-block')

@pytest.mark.parametrize('shape', [(1,1), (2,3), (3,3,3), (1,2,4), (7,11,13)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_streaming_constant_shortcut_exact(shape, dtype):
    from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
    a = np.full(shape, 17, dtype=dtype)
    compressed = compress_discrete_tensor_streaming(a, retain_cell_provenance=True)
    assert compressed.values.tolist() == [17]
    assert compressed.dimensions.tolist() == [0]
    assert compressed.original_grid_indices.tolist() == [0]
    assert compressed.boundary(0) == []
    assert len(compressed) == 1
    assert compressed.original_size == np.prod([2 * n + 1 for n in shape])
    graph = compute_ppg(a, algorithm='stream-block')
    assert len(graph.nodes) == 1
    assert graph.nodes[0].essential
    assert graph.edges == []

@pytest.mark.parametrize('shape', [(4, 5), (3, 4, 5)])
@pytest.mark.parametrize('reverse_mask', range(8))
def test_streaming_monotone_shortcut_matches_canonical_compression(shape, reverse_mask):
    from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
    rng = np.random.default_rng(12 + reverse_mask)
    sample = np.zeros(shape, dtype=np.uint16)
    for axis, length in enumerate(shape):
        steps = np.cumsum(rng.integers(0, 5, size=length, dtype=np.uint16))
        if reverse_mask & (1 << axis):
            steps = steps[::-1]
        reshape = [1] * len(shape)
        reshape[axis] = length
        sample += steps.reshape(reshape)
    streamed = compress_discrete_tensor_streaming(sample, retain_cell_provenance=True)
    materialized = compress_value_blocks(build_cubical_complex(sample))
    assert np.array_equal(streamed.values, materialized.values)
    assert np.array_equal(streamed.dimensions, materialized.dimensions)
    assert np.array_equal(streamed.original_grid_indices, materialized.original_grid_indices)
    assert [streamed.boundary(i) for i in range(len(streamed))] == materialized.boundaries

@pytest.mark.parametrize('m,r', [(2,3),(64,40),(100,300),(1100,50)])
def test_bitset_event_solver_matches_set_forward_substitution(m,r):
    from persistence_pairing_graph._graph import _solve_triangular_bitsets
    rng = np.random.default_rng(m + r)
    rows = []
    for i in range(m):
        lows = rng.integers(0, i, size=min(3,i)) if i else []
        rights = rng.integers(m, m + r, size=3) if r else []
        row = set(map(int, lows)) ^ set(map(int, rights))
        row.add(i)
        rows.append(row)
    expected = [row.copy() for row in rows]
    for i in range(m):
        for lower in sorted(x for x in expected[i] if x < i):
            expected[i].symmetric_difference_update(expected[lower])
    observed = _solve_triangular_bitsets(rows, m)
    assert observed is not None
    for i, bits in enumerate(observed):
        assert bits == sum(1 << (j-m) for j in expected[i] if j>=m)

@pytest.mark.parametrize('shape', [(17,13), (4, 9, 11), (1, 3, 17)])
def test_streaming_batched_uint16_histogram(shape):
    from persistence_pairing_graph._streaming import _count_buckets, _iter_keys
    array = np.random.default_rng(70).integers(0,65536,shape,dtype=np.uint16)
    grid_shape=tuple(2*n+1 for n in shape)
    n_buckets = 65536*(len(shape)+1)
    optimized = _count_buckets(array,grid_shape,n_buckets)
    expected = np.zeros_like(optimized)
    for _,keys in _iter_keys(array,grid_shape):
        expected += np.bincount(keys,minlength=n_buckets)
    assert np.array_equal(optimized,expected)

# Exact equivalence tests for the memory-bounded reduction / order / PPG backends.
@pytest.mark.parametrize('shape', [(2, 4), (3, 3), (2, 3, 4), (3, 4, 3)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_deep_delta_raw_packed_python_and_selective_equivalence(shape, dtype, tmp_path):
    from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
    from persistence_pairing_graph._reduction import reduce_boundary
    from persistence_pairing_graph._graph import build_edges, build_nodes
    for seed in (2, 9, 17):
        array_ = np.random.default_rng(seed).integers(0, 13, shape, dtype=dtype)
        delta = compress_discrete_tensor_streaming(
            array_, temp_dir=tmp_path, order_format='delta', retain_cell_provenance=True)
        raw = compress_discrete_tensor_streaming(
            array_, temp_dir=tmp_path, order_format='u32', retain_cell_provenance=True)
        assert np.array_equal(delta.values, raw.values)
        assert np.array_equal(delta.dimensions, raw.dimensions)
        assert np.array_equal(delta.original_grid_indices, raw.original_grid_indices)
        assert delta.indices == raw.indices
        assert delta.offsets == raw.offsets
        for storage, mode in [('python', 'full'), ('packed', 'full'), ('packed', 'selective')]:
            reduction = reduce_boundary(delta, storage=storage, log_mode=mode)
            nodes = build_nodes(delta.values, delta.dimensions, reduction)
            edges = build_edges(delta.values, reduction, nodes, max_relation_cache_entries=2)
            if storage == 'python' and mode == 'full':
                expected = (nodes, edges, reduction)
            else:
                assert (nodes, edges) == expected[:2]
                assert np.array_equal(reduction.low, expected[2].low)
                for j in range(len(delta)):
                    assert reduction.reduced_columns.get(j, []) == expected[2].reduced_columns.get(j, [])
        assert not list(tmp_path.iterdir())


def test_stream_edges_matches_regular_csv_and_is_bounded(tmp_path, pants):
    graph = compute_ppg(pants, algorithm='stream-block', max_relation_cache_entries=0)
    baseline_dir = tmp_path / 'baseline'
    export_csv(graph, 'pants.tif', output_dir=baseline_dir)
    output_dir = tmp_path / 'bounded'
    edges_path = output_dir / 'pants_persistence_pairing_graph_edges.csv'
    bounded = compute_ppg(pants, algorithm='stream-block', edge_output_path=edges_path,
                          max_relation_cache_entries=1)
    assert bounded.edges == []
    assert not bounded.edges_materialized
    assert bounded.edge_count == len(graph.edges)
    export_csv(bounded, 'pants.tif', output_dir=output_dir)
    for part in ['nodes', 'edges']:
        name = f'pants_persistence_pairing_graph_{part}.csv'
        assert (baseline_dir / name).read_bytes() == (output_dir / name).read_bytes()
    with pytest.raises(ValueError, match='streamed'):
        bounded.to_networkx()
    with pytest.raises(ValueError, match='differs'):
        export_csv(bounded, 'pants.tif', output_dir=tmp_path / 'other')


def test_delta_storage_reduction_and_compiled_reference(tmp_path):
    from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
    from persistence_pairing_graph._reduction import reduce_boundary
    from persistence_pairing_graph._accelerator import has_numba
    rng = np.random.default_rng(123)
    a = rng.integers(0, 64, (14, 11, 9), dtype=np.uint8)
    raw = compress_discrete_tensor_streaming(a, order_format='u32', temp_dir=tmp_path)
    delta = compress_discrete_tensor_streaming(a, order_format='delta', temp_dir=tmp_path)
    assert delta.order_payload_bytes < raw.order_payload_bytes
    assert delta.indices == raw.indices
    assert delta.offsets == raw.offsets
    assert delta.order_payload_bytes > 0
    base = reduce_boundary(delta)
    assert base.packed_bytes is not None
    if has_numba():
        accelerated = compress_discrete_tensor_streaming(a, order_format='delta',
                                                         accelerator='numba', temp_dir=tmp_path)
        assert accelerated.indices == delta.indices
        r = reduce_boundary(accelerated, accelerator='numba')
        assert np.array_equal(base.low, r.low)
        assert list(base.reduced_columns.items()) == list(r.reduced_columns.items())


def test_invalid_deep_parameters():
    a = np.array([[1, 9], [4, 2]], dtype=np.uint8)
    with pytest.raises(ValueError, match='order_format'):
        compute_ppg(a, algorithm='stream-block', order_format='rle')
    with pytest.raises(ValueError, match='storage'):
        compute_ppg(a, reduction_storage='bits')
    with pytest.raises(ValueError, match='log_mode'):
        compute_ppg(a, log_mode='sketch')
    with pytest.raises(ValueError, match='cache'):
        compute_ppg(a, max_relation_cache_entries=-1)

@pytest.mark.parametrize('shape,dimension', [((3,3),1), ((3,3,3),2)])
def test_closed_cubical_shell_has_predicted_homology(shape,dimension):
    a = np.zeros(shape,dtype=np.uint8)
    a[(1,)*len(shape)] = 10
    expected = (dimension, 0, 10)
    for algorithm in ('lazy','block','stream-block'):
        graph = compute_ppg(a,algorithm=algorithm)
        assert expected in [(n.dimension,n.birth_value,n.death_value) for n in graph.nodes]
        assert sum(n.essential for n in graph.nodes) == 1
        assert graph.edges == []


def test_cli_streams_csv_without_retaining_edges(tmp_path, pants):
    from persistence_pairing_graph.__main__ import main
    tifffile=pytest.importorskip('tifffile')
    source=tmp_path/'pants.tif'
    tifffile.imwrite(source,pants)
    assert main([str(source),'--algorithm','stream-block','--stream-edges',
                 '--order-format','delta','--max-relation-cache-entries','0'])==0
    name=tmp_path/'pants_persistence_pairing_graph_edges.csv'
    with name.open() as fp:
        assert len(list(csv.DictReader(fp)))==5

@pytest.mark.parametrize('shape', [(3,4), (3,3,3), (6,7,8)])
def test_disk_reduction_and_scratch_cleanup(shape, tmp_path):
    from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
    from persistence_pairing_graph._reduction import reduce_boundary
    from persistence_pairing_graph._graph import build_nodes, build_edges
    a=np.random.default_rng(84).integers(0,20,shape,dtype=np.uint8)
    complex_=compress_discrete_tensor_streaming(a,temp_dir=tmp_path)
    expected=reduce_boundary(complex_,storage='packed')
    for log_mode in ('full','selective'):
        actual=reduce_boundary(complex_,storage='disk', log_mode=log_mode, temp_dir=tmp_path)
        try:
            assert actual.disk_bytes is not None
            assert np.array_equal(actual.low,expected.low)
            for j in range(len(complex_)):
                assert actual.reduced_columns.get(j,[])==expected.reduced_columns.get(j,[])
            nodes=build_nodes(complex_.values,complex_.dimensions,actual)
            edges=build_edges(complex_.values,actual,nodes,max_relation_cache_entries=0)
            assert nodes==build_nodes(complex_.values,complex_.dimensions,expected)
            assert edges==build_edges(complex_.values,expected,nodes)
        finally:
            actual.close()
        assert not list(tmp_path.iterdir()), 'Disk reduction scratch should be deleted'
    graph=compute_ppg(a,algorithm='stream-block',reduction_storage='disk',temp_dir=tmp_path)
    assert graph.edge_count==len(graph.edges)
    assert not list(tmp_path.iterdir())


def test_rust_python_comparison_csv_signatures(tmp_path, pants):
    from compare_rust import compare_exports
    from persistence_pairing_graph import export_csv
    a, b = tmp_path / 'rust', tmp_path / 'python'
    export_csv(compute_ppg(pants, algorithm='stream-block'), 'pants.tif', output_dir=a)
    export_csv(compute_ppg(pants, algorithm='stream-block', reduction_storage='disk'),
               'pants.tif', output_dir=b)
    assert compare_exports(a, b, 'pants')['edges'] == 5
    edge_file = b / 'pants_persistence_pairing_graph_edges.csv'
    original = edge_file.read_text()
    edge_file.write_text(original.replace(',20,30,0,90,1', ',20,30,0,10,1'))
    with pytest.raises(AssertionError, match='transitions'):
        compare_exports(a, b, 'pants')
