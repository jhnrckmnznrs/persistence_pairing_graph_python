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
