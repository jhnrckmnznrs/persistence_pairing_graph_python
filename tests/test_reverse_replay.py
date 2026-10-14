"""Independent reference and streaming tests for filtered inclusion replay."""
from itertools import product

import numpy as np
import pytest

from persistence_pairing_graph import PPGAnalysis
from persistence_pairing_graph._cubical import build_cubical_complex, _expanded_values
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from persistence_pairing_graph._reverse_replay import SelectiveInclusionReplay
from test_transition_witness import PANTS, verify


def _direct_oracle(image):
    """Materialize every differential update, independently of the streaming trace."""
    complex_ = build_cubical_complex(image)
    live = set(range(len(complex_)))
    d = [set(complex_.boundary(j)) for j in range(len(complex_))]
    steps = []
    for j in range(len(complex_)):
        if j not in live:
            continue
        pivot_rows = [row for row in d[j]
                      if row in live and complex_.values[row] == complex_.values[j]]
        if not pivot_rows:
            continue
        a = max(pivot_rows)
        b = j
        tail = d[b] ^ {a}
        coupling = {c for c in live-{a,b} if a in d[c]}
        steps.append((a,b,tail,coupling))
        for c in live-{a,b}:
            if a in d[c]:
                d[c].symmetric_difference_update({a}|tail)
            d[c].discard(b)
        live.difference_update({a,b})
    return complex_, d, steps, live


def _direct_lift(cells, steps):
    result = set(cells)
    for a,b,_,coupling in reversed(steps):
        if len(result & coupling) % 2:
            result.add(b)
    return result


def _assert_exact_maps(image):
    filtered, d, steps, live = _direct_oracle(image)
    compressed = compress_discrete_tensor_streaming(image, retain_inclusion_trace=True)
    try:
        replay = SelectiveInclusionReplay(compressed.lift_trace, owner_storage='dict')
        compact_replay = SelectiveInclusionReplay(compressed.lift_trace, owner_storage='packed')
        direct_replay = SelectiveInclusionReplay(compressed.lift_trace, owner_storage='dense')
        assert compact_replay.owner_index_bytes == 16 * compressed.lift_trace.n_pairs
        assert direct_replay.owner_index_bytes == 4 * compressed.original_size
        old_index = lambda j: int(filtered.grid_indices[j])
        assert [int(x) for x in compressed.original_grid_indices] == [old_index(j) for j in sorted(live)]
        assert compressed.lift_trace.n_pairs == len(steps)
        for j,(a,b,r,_) in enumerate(steps):
            record = replay.records[j]
            assert int(record['birth']) == old_index(a)
            assert int(record['death']) == old_index(b)
            tail = replay._tail(j)
            assert set(map(int,tail)) == {old_index(x) for x in r}
        for j in sorted(live):
            expected = {old_index(c) for c in _direct_lift({j},steps)}
            actual = replay.lift((old_index(j),))
            assert set(map(int,actual)) == expected
            assert np.array_equal(actual,compact_replay.lift((old_index(j),)))
            assert np.array_equal(actual,direct_replay.lift((old_index(j),)))
            compressed_j = int(np.flatnonzero(compressed.original_grid_indices == old_index(j))[0])
            original_boundary = set(map(int, boundary_old_chain(actual, compressed.lift_trace.grid_shape)))
            target = replay.lift(compressed.original_grid_indices[compressed.boundary(compressed_j)])
            assert original_boundary == set(map(int,target))
            # Filtration on the original doubled grid cannot increase under inclusion.
            levels = _expanded_values(image).ravel()
            assert all(levels[x] <= filtered.values[j] for x in actual)
    finally:
        compressed.lift_trace.close()


@pytest.mark.parametrize('shape',[(2,3),(3,3),(2,2,2),(2,3,2)])
def test_random_reverse_replay_matches_independent_direct_inclusions(shape):
    for seed in range(5):
        image = np.random.default_rng(107+seed).integers(0,5,shape,dtype=np.uint8)
        _assert_exact_maps(image)


@pytest.mark.parametrize('shape',[(2,3),(2,2,2)])
def test_exhaustive_binary_exact_inclusions(shape):
    for bits in product((0,1), repeat=int(np.prod(shape))):
        _assert_exact_maps(np.asarray(bits, dtype=np.uint8).reshape(shape))


def test_replay_witness_matches_existing_pair_of_pants_and_3d_cavity(tmp_path):
    cavity = np.zeros((5,5,5),dtype=np.uint8)
    cavity[2,2,2] = 90
    for field in (PANTS,cavity):
        with PPGAnalysis(field, retain_inclusion_trace=True, temp_dir=tmp_path) as analysis:
            for i,node in enumerate(analysis.nodes):
                if node.essential or node.dimension >= field.ndim:
                    continue
                witness = analysis.witness(i, solver='replay')
                verify(field,witness)
                assert analysis.last_replay_stats is not None
                baseline = analysis.witness(i,solver='component')
                assert np.array_equal(witness.filling_grid_indices,baseline.filling_grid_indices)
                assert witness.target_nodes == baseline.target_nodes
                assert analysis.witness(i,solver='replay') is witness
        assert not list(tmp_path.iterdir())


def test_optional_trace_is_not_allocated_by_default():
    field = np.random.default_rng(13).integers(0,4,(4,4),dtype=np.uint16)
    compressed = compress_discrete_tensor_streaming(field,retain_chain_lifts=True)
    try:
        assert compressed.lift_trace.replay_records is None
        assert compressed.lift_trace.replay_tails is None
        with pytest.raises(ValueError,match='retain_inclusion_trace'):
            SelectiveInclusionReplay(compressed.lift_trace)
    finally:
        compressed.lift_trace.close()
    with PPGAnalysis(field) as analysis:
        finite=next(i for i,n in enumerate(analysis.nodes) if not n.essential)
        with pytest.raises(ValueError,match='retain_inclusion_trace'):
            analysis.witness(finite,solver='replay')


def test_replay_rejects_incompatible_resources_and_non_survivors():
    image=np.random.default_rng(13).integers(0,4,(2,3),dtype=np.uint8)
    compressed=compress_discrete_tensor_streaming(image,retain_inclusion_trace=True)
    try:
        replay=SelectiveInclusionReplay(compressed.lift_trace,max_cached_columns=50)
        if compressed.lift_trace.n_pairs:
            canceled=int(replay.records[0]['birth'])
            with pytest.raises(ValueError,match='canceled'):
                replay.lift([canceled])
        with pytest.raises(ValueError,match='max_output_cells'):
            replay.lift([],max_output_cells=0)
    finally:
        compressed.lift_trace.close()


def test_dense_owner_is_selected_and_rejects_corrupt_trace():
    image=np.zeros((6,6,6),dtype=np.uint8)
    compressed=compress_discrete_tensor_streaming(image,retain_inclusion_trace=True)
    try:
        replay=SelectiveInclusionReplay(compressed.lift_trace,owner_storage='auto')
        assert replay.owner_storage == 'dense'
        assert replay.owner_index_bytes == compressed.original_size*4
        assert replay._owner(int(compressed.original_grid_indices[0])) is None
    finally:
        compressed.lift_trace.close()
