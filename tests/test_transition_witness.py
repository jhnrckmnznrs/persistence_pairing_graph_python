"""Independent finite-grid certificate tests for complete PPG death relations."""
import numpy as np
import pytest
from math import prod

from persistence_pairing_graph import compute_ppg, transition_filling_witness
from persistence_pairing_graph._cubical import _expanded_values
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._transition_witness import _BoundedBoundarySpan


PANTS = np.asarray([
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,30,30,30,30,30,30,30,30,30,0],
    [0,30,10,10,10,30,20,20,20,30,0],
    [0,30,10,90,10,30,20,80,20,30,0],
    [0,30,10,10,10,30,20,20,20,30,0],
    [0,30,30,30,30,30,30,30,30,30,0],
    [0,0,0,0,0,0,0,0,0,0,0]
], dtype=np.uint8)


def verify(image, witness):
    shape = tuple(2*n+1 for n in image.shape)
    births = _expanded_values(image).ravel()
    assert not len(boundary_old_chain(witness.relation_support, shape))
    assert np.array_equal(boundary_old_chain(witness.filling_grid_indices,shape),
                          witness.relation_support)
    assert set(witness.target_nodes) == set(witness.target_cycles)
    assert witness.predeath_obstruction_verified
    assert len(witness.filling_grid_indices)
    assert all(births[c] <= witness.event_value for c in witness.filling_grid_indices)
    assert any(births[c] == witness.event_value for c in witness.filling_grid_indices)
    assert np.array_equal(witness.filling_doubled_coordinates,
                          np.column_stack(np.unravel_index(witness.filling_grid_indices,shape)))
    xor = set(witness.source_cycle.original_grid_indices)
    for cycle in witness.target_cycles.values():
        xor.symmetric_difference_update(cycle.original_grid_indices)
    assert np.array_equal(witness.relation_support,np.asarray(sorted(xor),dtype=np.int64))


def test_pair_of_pants_three_nontrivial_h1_edges_and_two_complete_fillings():
    w1 = transition_filling_witness(PANTS,2)
    assert w1.dimension == 1
    assert w1.event_value == 30
    assert w1.target_nodes == (3,4)
    verify(PANTS,w1)
    w2 = transition_filling_witness(PANTS,3)
    assert w2.dimension == 1
    assert w2.event_value == 80
    assert w2.target_nodes == (4,)
    verify(PANTS,w2)
    assert set(w1.filling_grid_indices) != set(w2.filling_grid_indices)


def test_individual_edge_is_not_independently_a_filling_relation():
    w = transition_filling_witness(PANTS,2)
    image = PANTS
    shape = (15,23)
    levels = _expanded_values(image).ravel()
    solver = _BoundedBoundarySpan(shape)
    for old in np.argsort(levels,kind='stable'):
        coords = np.unravel_index(int(old),shape)
        if sum(x&1 for x in coords) == 2 and levels[old] <= 30:
            solver.add(int(old))
    source = set(w.source_cycle.original_grid_indices)
    assert solver.solve(source) is None
    for v in w.target_nodes:
        alone = source ^ set(w.target_cycles[v].original_grid_indices)
        assert solver.solve(sorted(alone)) is None
    assert solver.solve(w.relation_support) is not None


def test_h0_merge_witness():
    w = transition_filling_witness(PANTS,0)
    assert w.dimension == 0
    verify(PANTS,w)


def test_3d_h1_witness():
    vol=np.repeat(PANTS[None,:,:],2,axis=0)
    w=transition_filling_witness(vol,2)
    assert w.dimension == 1
    verify(vol,w)


@pytest.mark.parametrize('shape',[(2,3),(3,3),(2,2,2),(2,3,2)])
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_randomized_full_rows_and_zero_outdegree(shape,dtype):
    rng = np.random.default_rng(731+sum(shape))
    for trial in range(4):
        image=rng.integers(0,5,shape,dtype=dtype)
        g=compute_ppg(image, algorithm='stream-block')
        finite=[k for k,node in enumerate(g.nodes) if not node.essential and node.dimension<image.ndim]
        for k in finite[:3]:
            w=transition_filling_witness(image,k)
            verify(image,w)
            assert w.target_nodes==tuple(sorted(e.to_node for e in g.edges if e.from_node==k))


def test_bounded_reference_not_mistaken_for_out_of_core():
    with pytest.raises(MemoryError):
        transition_filling_witness(PANTS,2,max_original_cells=10)
    with pytest.raises(MemoryError):
        transition_filling_witness(PANTS,2,max_boundary_columns=1)
    with pytest.raises(ValueError):
        transition_filling_witness(PANTS,-1)


def test_full_reference_and_canceled_pair_methods_agree_on_the_equation():
    for image in (PANTS, np.repeat(PANTS[None,:,:],2,axis=0)):
        for source in (2,3):
            pair=transition_filling_witness(image,source,solver='pair-trace')
            full=transition_filling_witness(image,source,solver='full-reference')
            verify(image,pair)
            verify(image,full)
            assert np.array_equal(pair.relation_support,full.relation_support)
            assert pair.target_nodes==full.target_nodes
            assert pair.event_value==full.event_value
            assert len(pair.compressed_filling_cells)>0
            assert pair.cancelled_death_correction_cells.ndim==1


def test_exhaustive_binary_2d_and_3d_small_events():
    for shape in ((2,3),(2,2,2)):
        for bits in range(1<<prod(shape)):
            image=np.asarray([(bits>>i)&1 for i in range(prod(shape))],dtype=np.uint8).reshape(shape)
            graph=compute_ppg(image,algorithm='stream-block')
            for source,node in enumerate(graph.nodes):
                if node.essential or node.dimension>=image.ndim:
                    continue
                cert=transition_filling_witness(image,source,solver='pair-trace')
                verify(image,cert)
                assert cert.target_nodes==tuple(sorted(e.to_node for e in graph.edges if e.from_node==source))


def test_closed_3d_cavity_is_filled_by_voxels_at_its_death_event():
    image=np.zeros((5,5,5),dtype=np.uint8)
    image[2,2,2]=90
    graph=compute_ppg(image,algorithm='stream-block')
    sources=[i for i,n in enumerate(graph.nodes) if n.dimension==2 and not n.essential]
    assert len(sources)==1
    for solver in ('pair-trace','full-reference'):
        w=transition_filling_witness(image,sources[0],solver=solver)
        assert w.target_nodes==()
        assert w.dimension==2
        assert w.event_value==90
        assert len(w.filling_grid_indices)==3
        verify(image,w)


def test_pair_trace_selective_solver_does_not_expand_original_grid(monkeypatch):
    from persistence_pairing_graph import _transition_witness as module
    def fail_if_materialized(_image):
        raise AssertionError('pair-trace unexpectedly materialized a full grid')
    monkeypatch.setattr(module, '_expanded_values', fail_if_materialized)
    w = transition_filling_witness(PANTS, 2, solver='pair-trace')
    assert w.target_nodes == (3, 4)
    assert w.predeath_obstruction_verified
    assert np.array_equal(boundary_old_chain(w.filling_grid_indices, (15, 23)), w.relation_support)


def test_selective_sparse_solver_on_grid_above_default_limit():
    image = np.zeros((14, 14, 14), dtype=np.uint8)
    image[7, 7, 7] = 120
    graph = compute_ppg(image, algorithm='stream-block')
    source, = (i for i, node in enumerate(graph.nodes)
               if node.dimension == 2 and not node.essential)
    witness = transition_filling_witness(image, source, solver='pair-trace',
                                         max_original_cells=30_000,
                                         max_boundary_columns=30_000)
    assert prod(2 * n + 1 for n in image.shape) > 20_000
    assert witness.event_value == 120
    assert witness.target_nodes == ()
    verify(image, witness)


def test_sparse_cancelled_death_basis_solves_original_boundary():
    from persistence_pairing_graph._lift import _BoundarySolver
    shape = (5, 5)
    solver = _BoundarySolver(shape, max_basis_columns=10, max_basis_entries=1000)
    for death in (6, 8, 16, 18):
        solver.add(death)
    target = boundary_old_chain((6, 18), shape)
    correction = solver.solve_boundary(target, max_chain_terms=10)
    assert np.array_equal(correction, np.array([6, 18]))
    assert np.array_equal(boundary_old_chain(correction, shape), target)
