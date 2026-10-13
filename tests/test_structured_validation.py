"""Topology-rich 3D regression cases and exact safe-event skipping tests."""
from __future__ import annotations

from collections import Counter
import csv

import numpy as np
import pytest

from persistence_pairing_graph import compute_ppg, export_csv
from persistence_pairing_graph._graph import iter_edges, build_nodes
from persistence_pairing_graph._reduction import reduce_boundary
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.profile_structured import make_volume


@pytest.mark.parametrize('case', ['shell','torus','multi-cavity','lattice','checkerboard','smooth','random'])
@pytest.mark.parametrize('size', [8,12])
def test_structured_exact_compression_and_graph_equivalence(case, size):
    a=make_volume(case,size)
    full=compute_ppg(a,algorithm='block',retain_cell_provenance=True)
    streamed=compute_ppg(a,algorithm='stream-block',retain_cell_provenance=True)
    assert streamed.reduced_cell_count == full.reduced_cell_count
    assert [(n.dimension,n.birth_cell,n.death_cell,n.birth_value,n.death_value) for n in streamed.nodes] == [
        (n.dimension,n.birth_cell,n.death_cell,n.birth_value,n.death_value) for n in full.nodes]
    assert [(e.dimension,e.from_node,e.to_node) for e in streamed.edges] == [
        (e.dimension,e.from_node,e.to_node) for e in full.edges]
    np.testing.assert_array_equal(streamed.original_grid_indices, full.original_grid_indices)


@pytest.mark.parametrize('case', ['checkerboard','random','multi-cavity','lattice','smooth'])
def test_empty_survivor_event_skip_preserves_exact_graph(case):
    # Test both the optimized and the original complete event solver on an
    # identical canonical reduction; this does not merely compare barcodes.
    a=make_volume(case,12)
    complex_=compress_discrete_tensor_streaming(a)
    reduction=reduce_boundary(complex_,storage='packed')
    try:
        nodes=build_nodes(complex_.values,complex_.dimensions,reduction)
        full=list(iter_edges(complex_.values,reduction,nodes,
                             skip_empty_survivor_events=False))
        fast=list(iter_edges(complex_.values,reduction,nodes,
                             skip_empty_survivor_events=True))
        assert fast == full
    finally:
        reduction.close()


def test_pair_of_pants_exact_nonzero_transition_survives_shortcut():
    pants=np.array([
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,30,30,30,30,30,30,30,30,30,0],
        [0,30,10,10,10,30,20,20,20,30,0],
        [0,30,10,90,10,30,20,80,20,30,0],
        [0,30,10,10,10,30,20,20,20,30,0],
        [0,30,30,30,30,30,30,30,30,30,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ],dtype=np.uint8)
    result=compute_ppg(pants,algorithm='stream-block')
    assert len(result.edges)==5


def test_3d_canonical_h1_h2_events():
    torus=compute_ppg(make_volume('torus',24),algorithm='stream-block')
    shell=compute_ppg(make_volume('shell',24),algorithm='stream-block')
    assert Counter(n.dimension for n in torus.nodes) == {0:1,1:1}
    assert Counter(n.dimension for n in shell.nodes) == {0:1,2:1}
    assert not torus.edges and not shell.edges


def test_differential_csv_comparator_catches_bad_edges(tmp_path):
    from compare_rust import compare_exports
    a=make_volume('multi-cavity',12)
    graph=compute_ppg(a,algorithm='stream-block',retain_cell_provenance=True)
    rust=tmp_path/'rust'; python=tmp_path/'python'
    export_csv(graph,'synthetic.tif',output_dir=rust,export_cell_provenance=True)
    export_csv(graph,'synthetic.tif',output_dir=python,export_cell_provenance=True)
    assert compare_exports(rust,python,'synthetic')['provenance_checked']
    f=python/'synthetic_persistence_pairing_graph_edges.csv'
    with f.open(newline='',encoding='utf-8') as h: rows=list(csv.DictReader(h))
    assert rows
    rows[0]['to_node']=str(int(rows[0]['to_node'])+1)
    with f.open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    with pytest.raises(AssertionError,match='Graph node IDs'):
        compare_exports(rust,python,'synthetic')


def test_tiff_fixture_pages_compatible_with_rust_reader(tmp_path):
    tifffile=pytest.importorskip('tifffile')
    from validate_rust_suite import fixtures
    for name,volume in fixtures(8,271828,True):
        p=tmp_path/f'{name}.tif'
        tifffile.imwrite(p,volume,photometric='minisblack',metadata=None)
        with tifffile.TiffFile(p) as tif:
            assert len(tif.pages)==(volume.shape[0] if volume.ndim==3 else 1)
            assert all(page.samplesperpixel==1 for page in tif.pages)
            assert all(page.dtype==volume.dtype for page in tif.pages)
            assert np.array_equal(tif.asarray(),volume)
