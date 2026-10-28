"""Exactness of selective old-incidence closure against full old enumeration."""
from itertools import combinations
from math import prod
import numpy as np
import pytest

from persistence_pairing_graph import certify_relative_filling, certify_transition_relative_locality
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._transition_witness import _original_cell_birth


def _cells(image, dim, criterion):
    grid=tuple(2*x+1 for x in image.shape)
    return [c for c in range(prod(grid)) if
            sum(x&1 for x in np.unravel_index(c,grid))==dim
            and criterion(_original_cell_birth(image,c,grid))]


@pytest.mark.parametrize('shape',[(2,2),(2,2,2),(3,2,2)])
@pytest.mark.parametrize('dimension', [0,1])
def test_component_and_full_have_identical_feasibility_and_obstructions(shape,dimension):
    rng=np.random.default_rng(1209+dimension*17+sum(shape))
    for _ in range(18):
        image=rng.integers(0,4,shape,dtype=np.uint8)
        t=2
        old=_cells(image,dimension+1,lambda x:x<t)
        new=_cells(image,dimension+1,lambda x:x==t)
        chosen=[c for c in new if rng.random()<0.6]
        # A candidate relation from an arbitrary pre-event old boundary or a
        # valid old cycle, including nonbounding cycles from the entire event.
        support=[c for c in old+new if rng.random()<0.45]
        relation=boundary_old_chain(support,tuple(2*n+1 for n in shape))
        if any(_original_cell_birth(image,int(c),tuple(2*n+1 for n in shape))>=t
               for c in relation):
            continue
        ref=certify_relative_filling(
            image,relation,chosen,dimension=dimension,event_value=t,
            old_solver='full')
        selected=certify_relative_filling(
            image,relation,chosen,dimension=dimension,event_value=t,
            old_solver='component')
        assert selected.feasible==ref.feasible
        assert selected.old_candidate_count<=ref.old_candidate_count
        assert selected.old_solver=='component'
        if selected.feasible:
            combined=[*selected.old_filling_cells,*selected.new_filling_cells]
            assert np.array_equal(
                boundary_old_chain(combined,tuple(2*n+1 for n in shape)),
                relation)
        else:
            obstruction=set(selected.obstruction_cells)
            assert len(obstruction.intersection(relation))%2==1
            # Unlike heuristic local search, the dual is valid against ALL
            # old candidates, including unvisited disconnected components.
            assert all(len(obstruction.intersection(
                boundary_old_chain([c],tuple(2*n+1 for n in shape))))%2==0
                       for c in old+chosen)


def test_disconnected_old_components_are_not_enumerated():
    image=np.full((5,5),9,dtype=np.uint8)
    image[0,0]=image[4,4]=1
    grid=(11,11)
    old=_cells(image,2,lambda x:x<5)
    assert len(old)==2
    relation=boundary_old_chain([old[0]],grid)
    full=certify_relative_filling(image,relation,[],dimension=1,
                                  event_value=5,old_solver='full')
    selective=certify_relative_filling(image,relation,[],dimension=1,
                                       event_value=5,old_solver='component')
    assert full.feasible and selective.feasible
    assert full.old_candidate_count==2
    assert selective.old_candidate_count==1
    assert selective.old_cells_examined>=1
    assert selective.old_faces_visited>=len(relation)


def test_component_work_cap_and_unknown_mode():
    image=np.zeros((4,4),dtype=np.uint8)
    with pytest.raises(ValueError,match='old_solver'):
        certify_relative_filling(image,[],[],dimension=1,event_value=2,
                                 old_solver='unsafe')
    with pytest.raises(MemoryError,match='max_examined_cofaces'):
        certify_relative_filling(image,[],[],dimension=1,event_value=2,
                                 max_examined_cofaces=1)


def test_all_new_faces_seed_old_components():
    image=np.array([[1,1,1],[1,3,1],[1,1,1]],dtype=np.uint8)
    grid=(7,7)
    new=_cells(image,2,lambda x:x==3)
    old=_cells(image,2,lambda x:x<3)
    # Even the zero relation must examine old components adjacent to an
    # allowed new column, because its old boundary may need cancellation.
    selected=certify_relative_filling(image,[],new,dimension=1,event_value=3)
    ref=certify_relative_filling(image,[],new,dimension=1,event_value=3,
                                old_solver='full')
    assert selected.feasible==ref.feasible
    assert selected.old_candidate_count==len(old)
