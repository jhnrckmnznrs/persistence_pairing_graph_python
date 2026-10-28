"""Independent exactness tests for sparse old cohomology coordinates.

Reference oracle: all old original-grid (q+1)-columns, not the stored
compressed reduction. Includes H0/H1/H2, 2D and 3D, and PPG death events.
"""
from math import prod

import numpy as np
import pytest

from persistence_pairing_graph import (
    PPGAnalysis, SparseOldLevelOracle, certify_relative_filling,
)
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._transition_witness import _original_cell_birth


def cells(image, q, match):
    grid=tuple(2*n+1 for n in image.shape)
    return [i for i in range(prod(grid)) if
            sum(x&1 for x in np.unravel_index(i,grid))==q
            and match(_original_cell_birth(image,i,grid))]


def _reference(image, c, new, q, t):
    return certify_relative_filling(image,c,new,dimension=q,event_value=t,
                                    old_solver='full',max_grid_cells=200_000,
                                    max_boundary_rows=100_000)


@pytest.mark.parametrize('shape',[(2,2),(3,3),(2,2,2),(2,3,2)])
def test_old_level_quotient_coordinates_agree_with_full_original_boundary(shape):
    rng=np.random.default_rng(1512+sum(shape))
    grid=tuple(2*n+1 for n in shape)
    for repeat in range(3):
        image=rng.integers(0,4,shape,dtype=np.uint8)
        with PPGAnalysis(image,retain_inclusion_trace=True) as analysis:
            oracle=SparseOldLevelOracle(analysis)
            for q in range(image.ndim):
                t=2
                oldtop=cells(image,q+1,lambda level:level<t)
                oldq=cells(image,q,lambda level:level<t)
                candidates=[]
                if oldq and q==0:
                    # Every 0-chain is a cycle, but a single q>0 cell
                    # generally is not; the oracle must reject noncycles.
                    candidates.extend(([oldq[0]],oldq[:2]))
                candidates.append([])
                if oldtop:
                    candidates.append(boundary_old_chain(oldtop[::2],grid))
                for c in candidates:
                    if any(_original_cell_birth(image,int(k),grid)>=t for k in c):
                        continue
                    result=oracle.coordinates(c,dimension=q,event_value=t)
                    reference=_reference(image,c,[],q,t)
                    assert result.is_boundary==reference.feasible
                    assert result.coordinates==tuple(sorted(result.coordinates))
                    if result.is_boundary:
                        pre=oracle.boundary_preimage(c,dimension=q,event_value=t)
                        assert np.array_equal(boundary_old_chain(pre,grid),
                                              np.asarray(sorted(c),dtype=np.int64))
                        assert all(_original_cell_birth(image,int(x),grid)<t for x in pre)


@pytest.mark.parametrize('shape',[(2,2),(2,2,2),(3,2,2)])
def test_sparse_relative_feasibility_and_preimages_match_full_reference(shape):
    rng=np.random.default_rng(2311+sum(shape))
    grid=tuple(2*n+1 for n in shape)
    for repeat in range(6):
        image=rng.integers(0,4,shape,dtype=np.uint8)
        with PPGAnalysis(image,retain_inclusion_trace=True) as analysis:
            oracle=SparseOldLevelOracle(analysis)
            for q in range(image.ndim):
                t=2
                oldtop=cells(image,q+1,lambda v:v<t)
                newtop=cells(image,q+1,lambda v:v==t)
                new=newtop[::2]
                # Choose old cycles from old boundaries; also include an old
                # vertex when q=0 (which can be nonbounding).
                oldcycle=boundary_old_chain(oldtop[::2],grid)
                cases=[oldcycle, []]
                if q==0:
                    oldvertices=cells(image,0,lambda v:v<t)
                    if oldvertices:cases.append([oldvertices[0]])
                for c in cases:
                    ref=_reference(image,c,new,q,t)
                    got=oracle.restricted_relative(c,new,dimension=q,event_value=t)
                    assert got.feasible==ref.feasible
                    if got.feasible:
                        assert np.array_equal(boundary_old_chain(
                            [*got.old_filling_cells,*got.new_filling_cells],grid),
                            np.asarray(sorted(c),dtype=np.int64))
                        assert all(_original_cell_birth(image,int(x),grid)<t
                                   for x in got.old_filling_cells)
                        assert set(map(int,got.new_filling_cells)).issubset(new)
                    else:
                        assert got.obstruction_coordinates


def test_ppg_pair_of_pants_complete_death_relation():
    image=np.array([
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,30,30,30,30,30,30,30,30,30,0],
        [0,30,10,10,10,30,20,20,20,30,0],
        [0,30,10,90,10,30,20,80,20,30,0],
        [0,30,10,10,10,30,20,20,20,30,0],
        [0,30,30,30,30,30,30,30,30,30,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ],dtype=np.uint8)
    grid=(15,23)
    with PPGAnalysis(image,retain_inclusion_trace=True) as analysis:
        witness=analysis.witness(2,solver='replay')
        oracle=SparseOldLevelOracle(analysis)
        old=oracle.coordinates(witness.relation_support,dimension=1,event_value=30)
        assert not old.is_boundary
        assert oracle.coordinates(witness.relation_support,dimension=1,event_value=31).is_boundary
        after=oracle.boundary_preimage(witness.relation_support,dimension=1,event_value=31)
        assert np.array_equal(boundary_old_chain(after,grid),
                              witness.relation_support)
        permitted=[int(c) for c in witness.filling_grid_indices
                   if _original_cell_birth(image,int(c),grid)==30]
        feasible=oracle.restricted_relative(
            witness.relation_support,permitted,dimension=1,event_value=30)
        assert feasible.feasible
        assert np.array_equal(boundary_old_chain(
            [*feasible.old_filling_cells,*feasible.new_filling_cells],grid),
            witness.relation_support)
        impossible=oracle.restricted_relative(
            witness.relation_support,[],dimension=1,event_value=30)
        assert not impossible.feasible
        assert impossible.obstruction_coordinates


def test_query_resource_errors_and_wrong_filtration():
    image=np.asarray([[1,2],[1,1]],dtype=np.uint8)
    with PPGAnalysis(image,retain_inclusion_trace=True) as analysis:
        oracle=SparseOldLevelOracle(analysis)
        new=cells(image,2,lambda t:t==2)
        with pytest.raises(ValueError):
            oracle.restricted_relative([],new,dimension=1,event_value=1)
        with pytest.raises(ValueError):
            oracle.coordinates([new[0]],dimension=1,event_value=2)
        with pytest.raises(ValueError):
            oracle.restricted_relative([],new,dimension=1,event_value=2,
                                        max_new_cells=0)


@pytest.mark.parametrize('shape',[(2,3),(2,2,2),(3,3,2)])
def test_filtered_chain_homotopy_identity_on_old_cycles(shape):
    rng=np.random.default_rng(2007+sum(shape))
    grid=tuple(2*n+1 for n in shape)
    for rep in range(3):
        image=rng.integers(0,4,shape,dtype=np.uint8)
        with PPGAnalysis(image,retain_inclusion_trace=True) as analysis:
            oracle=SparseOldLevelOracle(analysis)
            for q in range(image.ndim):
                oldtop=cells(image,q+1,lambda level:level<2)
                cycle=boundary_old_chain(oldtop[::2],grid)
                projected=oracle.projector.project(
                    cycle,stop_before_value=2,record_activations=True)
                activated=oracle.projector.last_activated_cancellations
                homotopy=set()
                for j in activated:
                    homotopy.symmetric_difference_update(
                        map(int,oracle.inclusion.lift_canceled_death(j)))
                old_included=oracle.inclusion.lift(projected)
                residual=set(map(int,cycle))
                residual.symmetric_difference_update(map(int,old_included))
                assert set(map(int,boundary_old_chain(sorted(homotopy),grid)))==residual
                correct=oracle.boundary_preimage(
                    cycle,dimension=q,event_value=2,
                    correction_method='homotopy')
                reference=oracle.boundary_preimage(
                    cycle,dimension=q,event_value=2,
                    correction_method='incidence')
                assert np.array_equal(boundary_old_chain(correct,grid),cycle)
                assert np.array_equal(boundary_old_chain(reference,grid),cycle)


def test_lazy_cocycle_annihilates_every_old_boundary():
    image=np.array([
        [0,0,0,0,0,0,0],
        [0,2,2,2,2,2,0],
        [0,2,1,3,1,2,0],
        [0,2,2,2,2,2,0],
        [0,0,0,0,0,0,0],
    ],dtype=np.uint8)
    with PPGAnalysis(image,retain_inclusion_trace=True) as analysis:
        oracle=SparseOldLevelOracle(analysis)
        oldedges=cells(image,1,lambda t:t<3)
        if oldedges:
            # Each component of a quotient normal form yields a cocycle,
            # even when evaluated on noncycle original q-chains.
            q=1
            cycle=oldedges[:1]
            # Collect a nonempty quotient coordinate from some old chain.
            projected=oracle.projector.project(cycle)
            old_coords=[oracle._compressed_id_from_old(x) for x in projected]
            normal,_,_=oracle._normal_form(old_coords,q,3)
            for basis_coordinate in normal[:3]:
                phi=(basis_coordinate,)
                assert oracle.evaluate_cocycle(phi,cycle,
                        dimension=q,event_value=3)==1
                for old_cell in cells(image,q+1,lambda t:t<3):
                    boundary=boundary_old_chain([old_cell],analysis.grid_shape)
                    assert oracle.evaluate_cocycle(phi,boundary,
                           dimension=q,event_value=3)==0


@pytest.mark.parametrize('side',[3,5,9])
def test_one_new_square_can_force_quadratic_original_filling_support(side):
    # The selected outer cycle has 4*side edges, but its unique filling
    # in the complete disk uses side^2 squares. The death event adds one
    # central square. This is an output-size obstruction, not a speed test.
    image=np.ones((side,side),dtype=np.uint8)
    image[side//2,side//2]=2
    grid=(2*side+1,)*2
    old=cells(image,2,lambda t:t<2)
    new=cells(image,2,lambda t:t==2)
    assert len(old)==side*side-1 and len(new)==1
    outer=boundary_old_chain([*old,*new],grid)
    assert len(outer)==4*side
    with PPGAnalysis(image,retain_inclusion_trace=True) as analysis:
        oracle=SparseOldLevelOracle(analysis)
        assert not oracle.coordinates(outer,dimension=1,event_value=2).is_boundary
        decision=oracle.restricted_relative(
            outer,new,dimension=1,event_value=2,reconstruct=False)
        assert decision.feasible and not decision.old_filling_reconstructed
        complete=oracle.restricted_relative(
            outer,new,dimension=1,event_value=2,reconstruct=True)
        assert complete.feasible and complete.old_filling_reconstructed
        assert len(complete.new_filling_cells)==1
        assert len(complete.old_filling_cells)==side*side-1
        assert np.array_equal(boundary_old_chain(
            [*complete.old_filling_cells,*complete.new_filling_cells],grid),outer)
