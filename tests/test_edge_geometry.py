"""Canonical PPG edge-to-geometry linkage and exact cubical-support checks."""

import numpy as np
import pytest

from persistence_pairing_graph import (compute_ppg, explain_ppg_edges)
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._edge_geometry import _bounds, _min_cell_distance


@pytest.fixture
def pants():
    return np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0,30,30,30,30,30,30,30,30,30, 0],
        [0,30,10,10,10,30,20,20,20,30, 0],
        [0,30,10,90,10,30,20,80,20,30, 0],
        [0,30,10,10,10,30,20,20,20,30, 0],
        [0,30,30,30,30,30,30,30,30,30, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)


def _check_report(image, report):
    grid_shape = tuple(2 * s + 1 for s in image.shape)
    actual = compute_ppg(image, algorithm='stream-block')
    for link in report.edges:
        edge = actual.edges[link.edge_index]
        assert (link.edge.dimension, link.edge.from_node, link.edge.to_node) == (
            edge.dimension, edge.from_node, edge.to_node)
        assert link.edge.coefficient == 1
        assert link.source.birth_value < link.event_value
        assert link.event_value < link.target.death_value
        for rep in (link.source, link.target):
            assert not len(boundary_old_chain(rep.original_grid_indices, grid_shape))
            assert np.all(np.sum(rep.doubled_grid_coordinates % 2, axis=1) == rep.dimension)
        assert link.shared_cell_count == len(set(link.source.original_grid_indices)
                                             & set(link.target.original_grid_indices))
        if link.min_distance is not None:
            assert link.min_distance >= 0
            assert link.geometrically_touching == (link.min_distance == 0)


def test_pair_of_pants_lifted_h1_edges_match_canonical_ppg(pants):
    report = explain_ppg_edges(pants, dimensions=(1,), max_edges=10)
    assert [r.edge_index for r in report.edges] == [2, 3, 4]
    assert [(r.edge.from_node, r.edge.to_node) for r in report.edges] == [(2,4),(2,3),(3,4)]
    assert len(report.cycles) == 3
    _check_report(pants, report)
    assert all(link.min_distance is not None for link in report.edges)
    # PPG edges need not denote physical contact or common supporting cells.
    assert all(link.shared_cell_count == 0 for link in report.edges)
    assert all(link.min_distance > 0 for link in report.edges)


def test_h0_edges_are_supported_too(pants):
    report = explain_ppg_edges(pants, dimensions=(0,), max_edges=10)
    assert [r.edge_index for r in report.edges] == [0,1]
    _check_report(pants, report)
    assert all(len(rep.original_grid_indices) == 1 for rep in report.cycles.values())


def test_specific_edge_selection_uses_full_ppg_stream_indices(pants):
    report = explain_ppg_edges(pants, edge_indices=[4, 2], dimensions=(1,), max_edges=2)
    assert [r.edge_index for r in report.edges] == [2, 4]
    _check_report(pants, report)
    with pytest.raises(ValueError, match='absent or failed'):
        explain_ppg_edges(pants, edge_indices=[1], dimensions=(1,))
    with pytest.raises(ValueError, match='edge_indices'):
        explain_ppg_edges(pants, edge_indices=[1,2], max_edges=1)
    with pytest.raises(ValueError, match='edge_indices'):
        explain_ppg_edges(pants, edge_indices=[-1])


def test_no_approximation_on_pair_cap(pants):
    report = explain_ppg_edges(pants, dimensions=(1,), max_edges=1,
                               max_geometry_pairs=0, spacing=(2, 0.5))
    assert report.edges[0].min_distance is None
    assert report.edges[0].geometrically_touching is None
    assert report.spacing == (2.0, 0.5)
    assert report.edges[0].source_bounds[0][0] >= 0


def test_cubical_cells_can_touch_without_sharing_cell_ids():
    from persistence_pairing_graph._lift import LiftedCycleRepresentative
    # Doubled 2D grid: horizontal and vertical 1-cells meet at a vertex.
    def fake(cell, coord):
        return LiftedCycleRepresentative(1, 0, None, 0, float('inf'),
                                         np.array([0]), np.array([cell]),
                                         np.array([coord], dtype=np.int64))
    a = _bounds(fake(5, (1, 0)), np.ones(2))
    b = _bounds(fake(11, (2, 1)), np.ones(2))
    assert _min_cell_distance(a, b, 2) == 0
    c = _bounds(fake(7, (1, 2)), np.ones(2))
    assert _min_cell_distance(a, c, 2) == 1
    assert _min_cell_distance(a, c, 0) is None


def test_three_dimensional_edges_and_cycle_verification(pants):
    image = np.repeat(pants[None,:,:], 2, axis=0)
    report = explain_ppg_edges(image, dimensions=(1,2), max_edges=8)
    assert report.image_shape == image.shape
    _check_report(image, report)


def test_uint16_and_storage_backends(pants, tmp_path):
    image = (pants.astype(np.uint16) * 1000)
    reference = explain_ppg_edges(image, dimensions=(1,), max_edges=3,
                                  temp_dir=tmp_path)
    alternative = explain_ppg_edges(image, dimensions=(1,), max_edges=3,
                                    order_format='delta', metadata_storage='disk',
                                    reduction_storage='disk', temp_dir=tmp_path)
    assert [x.edge for x in reference.edges] == [x.edge for x in alternative.edges]
    for a,b in zip(reference.edges,alternative.edges):
        assert np.array_equal(a.source.original_grid_indices,b.source.original_grid_indices)
        assert np.array_equal(a.target.original_grid_indices,b.target.original_grid_indices)
        assert a.min_distance == b.min_distance
    assert not list(tmp_path.iterdir())


def test_export_csv_and_npz_round_trip(pants, tmp_path):
    report = explain_ppg_edges(pants, dimensions=(1,), max_edges=5)
    csv_path = report.write_csv(tmp_path / 'edges.csv')
    npz_path = report.write_npz(tmp_path / 'edges.npz')
    assert len(csv_path.read_text().splitlines()) == len(report.edges)+1
    with np.load(npz_path, allow_pickle=False) as data:
        assert data['edge_indices'].tolist() == [2,3,4]
        assert data['node_ids'].tolist() == [2,3,4]
        assert data['support_offsets'][-1] == len(data['original_grid_cell_ids'])
        node_ids = data['node_ids'].tolist()
        for i,node_id in enumerate(node_ids):
            start, end = data['support_offsets'][i:i+2]
            chain = data['original_grid_cell_ids'][start:end]
            assert not len(boundary_old_chain(chain, (15, 23)))
            assert np.array_equal(chain, report.cycles[node_id].original_grid_indices)
        assert data['from_node'].tolist() == [2,2,3]
        assert data['to_node'].tolist() == [4,3,4]


def test_cli_works_and_releases_scratch(pants, tmp_path):
    tiff = pytest.importorskip('tifffile')
    from persistence_pairing_graph._edge_geometry_cli import main
    src = tmp_path / 'pants.tif'
    prefix = tmp_path / 'result'
    tiff.imwrite(src, pants)
    assert main([str(src), '--dimensions', '1', '--max-edges', '5',
                 '--output-prefix', str(prefix), '--temp-dir', str(tmp_path)]) == 0
    assert (tmp_path / 'result.csv').is_file()
    with np.load(tmp_path / 'result.npz', allow_pickle=False) as d:
        assert d['edge_indices'].tolist() == [2,3,4]
    assert len(list(tmp_path.iterdir())) == 3


@pytest.mark.parametrize('bad_spacing', [(0,1), (1,float('nan')), (1,), (1,2,3)])
def test_bad_spacing_rejected(pants,bad_spacing):
    with pytest.raises(ValueError, match='spacing'):
        explain_ppg_edges(pants, spacing=bad_spacing)
