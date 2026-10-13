"""Exact selective incidence lifting, reusable sessions and resource lifetimes."""
import numpy as np
import pytest

from persistence_pairing_graph import PPGAnalysis, compute_ppg, transition_filling_witness
from persistence_pairing_graph._lift import StreamingLiftData, boundary_old_chain
from persistence_pairing_graph._local_pairs import solve_component_correction
from test_transition_witness import PANTS, verify


def test_disconnected_cancellation_incidence_closure_is_strict_subset():
    # Two spatially isolated squares on the doubled grid. Only the first
    # canceled 2-cell is connected to the selected 1-boundary.
    grid = (5, 11)
    stride = grid[1]
    first = stride + 1
    second = stride + 9
    with StreamingLiftData(grid) as trace:
        trace.add_pair(first, 2, 3)
        trace.add_pair(second, 2, 3)
        trace.finish()
        boundary = boundary_old_chain((first,), grid)
        result, stats = solve_component_correction(
            trace, boundary, dimension=1, level=3,
            max_basis_columns=1, max_basis_entries=1000,
            max_chain_terms=100, max_component_faces=100)
        assert result.tolist() == [first]
        assert stats.eligible_deaths == 2
        assert stats.selected_deaths == 1
        assert stats.visited_faces == 4


def test_component_solver_refuses_insufficient_workspace():
    grid = (5, 11)
    death = grid[1] + 1
    with StreamingLiftData(grid) as trace:
        trace.add_pair(death, 2, 4)
        trace.finish()
        target = boundary_old_chain((death,), grid)
        with pytest.raises(MemoryError, match='max_component_faces'):
            solve_component_correction(trace, target, dimension=1, level=4,
                max_basis_columns=10, max_basis_entries=200,
                max_chain_terms=10, max_component_faces=3)
        with pytest.raises(RuntimeError, match='No canceled death'):
            solve_component_correction(trace, target, dimension=1, level=3,
                max_basis_columns=10, max_basis_entries=200,
                max_chain_terms=10)


def test_pair_of_pants_exact_witness_matches_prior_solver_and_caches_events(tmp_path, monkeypatch):
    with PPGAnalysis(PANTS, temp_dir=tmp_path) as analysis:
        regular = compute_ppg(PANTS, algorithm='stream-block')
        for k, n in enumerate(regular.nodes):
            if not n.essential:
                assert analysis.targets(k) == tuple(sorted(
                    e.to_node for e in regular.edges if e.from_node == k))
        assert analysis.targets(2) == (3, 4)
        first_event_count = analysis.event_solves
        assert analysis.targets(2) == (3, 4)
        assert analysis.event_solves == first_event_count
        first = analysis.witness(2)
        verify(PANTS, first)
        assert first.solver == 'component'
        baseline = transition_filling_witness(PANTS, 2)
        assert np.array_equal(first.filling_grid_indices, baseline.filling_grid_indices)
        assert analysis.cached_cycle_count == 3
        assert analysis.witness(2).filling_grid_indices.tolist() == first.filling_grid_indices.tolist()
        assert analysis.cached_cycle_count == 3
        second = analysis.witness(3)
        verify(PANTS, second)
        assert second.solver == 'component'
        assert np.array_equal(second.filling_grid_indices,
                              transition_filling_witness(PANTS, 3).filling_grid_indices)
        assert analysis.pairs_replayed_for_cycles > 0
        assert analysis.cached_cycle_count == 3
        full = analysis.witness(3, solver='pair-trace')
        assert np.array_equal(full.filling_grid_indices, second.filling_grid_indices)
        assert analysis.last_component_stats is None
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match='closed'):
        analysis.witness(2)


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2), (3, 3, 2)])
def test_random_exact_component_witness_matches_reference(shape, tmp_path):
    rng = np.random.default_rng(210 + len(shape))
    for trial in range(3):
        field = rng.integers(0, 5, shape, dtype=np.uint8)
        with PPGAnalysis(field, temp_dir=tmp_path) as analysis:
            finite = [i for i, n in enumerate(analysis.nodes) if not n.essential]
            for source in finite[:3]:
                selective = analysis.witness(source, solver='component')
                baseline = transition_filling_witness(field, source, solver='pair-trace')
                verify(field, selective)
                assert selective.target_nodes == baseline.target_nodes
                assert np.array_equal(selective.relation_support, baseline.relation_support)
                assert np.array_equal(selective.filling_grid_indices, baseline.filling_grid_indices)
    assert not list(tmp_path.iterdir())


def test_cavity_reconstruction_on_3d_streamed_analysis(tmp_path):
    field = np.zeros((5, 5, 5), dtype=np.uint8)
    field[2, 2, 2] = 90
    with PPGAnalysis(field, temp_dir=tmp_path) as analysis:
        node, = [i for i, n in enumerate(analysis.nodes) if n.dimension == 2 and not n.essential]
        witness = analysis.witness(node)
        assert witness.target_nodes == ()
        assert witness.event_value == 90
        verify(field, witness)
        assert analysis.last_component_stats.eligible_deaths >= analysis.last_component_stats.selected_deaths
        assert len(witness.filling_grid_indices) == 3
    assert not list(tmp_path.iterdir())


def test_early_failure_closes_disk_backed_trace(tmp_path):
    field = np.zeros((3,3), dtype=np.uint8)
    with pytest.raises(ValueError, match='positive'):
        PPGAnalysis(field, max_basis_columns=0, temp_dir=tmp_path)
    assert not list(tmp_path.iterdir())
