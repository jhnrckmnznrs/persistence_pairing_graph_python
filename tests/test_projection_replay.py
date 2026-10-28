"""Independent projection oracle and exact chain-map consistency tests."""
from itertools import product
from math import prod

import numpy as np
import pytest

from persistence_pairing_graph import SelectiveProjectionReplay
from persistence_pairing_graph._cubical import build_cubical_complex, _expanded_values
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from test_reverse_replay import _direct_oracle


def _project_direct(ids, steps, cell_id, cutoff):
    chain=set(ids)
    for a,b,tail,_ in steps[:cutoff]:
        ai=cell_id(a)
        bi=cell_id(b)
        if ai in chain:
            chain.remove(ai)
            chain.symmetric_difference_update(cell_id(c) for c in tail)
        chain.discard(bi)
    return chain


@pytest.mark.parametrize('shape',[(2,3),(3,3),(2,2,2),(2,3,2)])
def test_selective_projection_matches_independent_original_cell_elimination(shape):
    rng=np.random.default_rng(819+sum(shape))
    for _ in range(4):
        image=rng.integers(0,4,shape,dtype=np.uint8)
        oracle, d, steps, live=_direct_oracle(image)
        compressed=compress_discrete_tensor_streaming(
            image,retain_inclusion_trace=True)
        try:
            replay=SelectiveProjectionReplay(compressed.lift_trace,owner_storage='dict')
            id_of=lambda j:int(oracle.grid_indices[j])
            original=[id_of(i) for i in range(len(oracle))]
            for j in range(0,len(original),max(1,len(original)//19)):
                src=(original[j],)
                expected=_project_direct(src,steps,id_of,len(steps))
                result=replay.project(src)
                assert set(map(int,result))==expected
                assert all(replay._owner(int(x)) is None for x in result)
            if len(steps):
                # Compare every event threshold with the first t-cancellations
                # of the direct oracle. Event ordering is nondecreasing.
                levels=np.asarray(compressed.lift_trace.pairs['value'])
                threshold=int(levels[len(levels)//2])
                cutoff=int(np.searchsorted(levels,threshold,side='left'))
                for src in original[::max(1,len(original)//11)]:
                    expected=_project_direct({src},steps,id_of,cutoff)
                    got=replay.project([src],stop_before_value=threshold)
                    assert set(map(int,got))==expected
                    assert replay.last_projection_stats.eligible_cancellations==cutoff
        finally:
            compressed.lift_trace.close()


def test_project_is_chain_map_to_final_compressed_boundary():
    rng=np.random.default_rng(177)
    for shape in ((2,3),(2,2,2)):
        image=rng.integers(0,5,shape,dtype=np.uint8)
        compressed=compress_discrete_tensor_streaming(image,retain_inclusion_trace=True)
        try:
            rep=SelectiveProjectionReplay(compressed.lift_trace,owner_storage='packed')
            grid=compressed.lift_trace.grid_shape
            idx={int(c):j for j,c in enumerate(compressed.original_grid_indices)}
            for original_cell in range(prod(grid)):
                projected=rep.project([original_cell])
                # d'_M p(c)
                reduced_boundary=set()
                for old in projected:
                    for k in compressed.boundary(idx[int(old)]):
                        if k in reduced_boundary:reduced_boundary.remove(k)
                        else:reduced_boundary.add(k)
                lhs={int(compressed.original_grid_indices[k]) for k in reduced_boundary}
                # p(d_C c)
                rhs=set(map(int,rep.project(
                    boundary_old_chain([original_cell],grid))))
                assert lhs==rhs
        finally:
            compressed.lift_trace.close()


def test_projection_is_left_inverse_of_exact_reverse_inclusion():
    from persistence_pairing_graph import SelectiveInclusionReplay
    image=np.random.default_rng(7).integers(0,4,(3,3,2),dtype=np.uint8)
    compressed=compress_discrete_tensor_streaming(image,retain_inclusion_trace=True)
    try:
        projector=SelectiveProjectionReplay(compressed.lift_trace)
        inclusion=SelectiveInclusionReplay(compressed.lift_trace)
        ids=compressed.original_grid_indices
        for cell in ids[::max(1,len(ids)//25)]:
            lifted=inclusion.lift([int(cell)])
            roundtrip=projector.project(lifted)
            assert list(roundtrip)==[int(cell)]
    finally:
        compressed.lift_trace.close()


def test_projection_resources_and_bounds():
    image=np.zeros((2,2),dtype=np.uint8)
    compressed=compress_discrete_tensor_streaming(image,retain_inclusion_trace=True)
    try:
        rep=SelectiveProjectionReplay(compressed.lift_trace)
        with pytest.raises(IndexError):
            rep.project([-1])
        with pytest.raises(ValueError):
            rep.project([],max_output_cells=0)
        with pytest.raises(ValueError):
            rep.project([],stop_before_value=float('inf'))
        assert len(rep.project([]))==0
    finally:
        compressed.lift_trace.close()
