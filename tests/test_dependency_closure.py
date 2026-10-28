"""Query-local influence statistics must not modify exact reverse-replay outputs."""
import numpy as np

from persistence_pairing_graph import PPGAnalysis, DependencyClosureStatistics
from persistence_pairing_graph._reverse_replay import SelectiveInclusionReplay
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming


def test_empty_and_cached_closure_reports():
    image=np.random.default_rng(9).integers(0,3,(3,3),dtype=np.uint8)
    compressed=compress_discrete_tensor_streaming(image,retain_inclusion_trace=True)
    try:
        rep=SelectiveInclusionReplay(compressed.lift_trace)
        empty=rep.lift([],collect_dependency_stats=True)
        assert len(empty)==0
        stats=rep.last_closure_stats
        assert isinstance(stats,DependencyClosureStatistics)
        assert stats.seed_cells==0
        assert stats.reachable_records==0
        assert stats.maximum_observed_depth==0
        assert stats.dependency_arcs==0
        assert stats.replayed_records==0

        start=int(compressed.original_grid_indices[0])
        first=rep.lift([start],collect_dependency_stats=True)
        cold=rep.last_closure_stats
        hot_result=rep.lift([start],collect_dependency_stats=True)
        hot=rep.last_closure_stats
        assert np.array_equal(first,hot_result)
        assert cold==hot
        assert 0<=cold.replayed_records<=cold.reachable_records<=cold.total_cancellations
        assert 0<=cold.reachable_fraction<=1
        assert cold.influence_columns_queried>=1
        assert cold.maximum_observed_depth<=cold.total_cancellations
        without=rep.lift([start],collect_dependency_stats=False)
        assert np.array_equal(first,without)
        assert rep.last_closure_stats is None
    finally:
        compressed.lift_trace.close()


def test_replay_closure_event_api_and_cached_witness():
    image=np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,30,30,30,30,30,30,30,30,30,0],
        [0,30,10,10,10,30,20,20,20,30,0],
        [0,30,10,90,10,30,20,80,20,30,0],
        [0,30,10,10,10,30,20,20,20,30,0],
        [0,30,30,30,30,30,30,30,30,30,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ],dtype=np.uint8)
    with PPGAnalysis(image,retain_inclusion_trace=True) as analysis:
        event=analysis.witness(2,solver='replay')
        assert analysis.last_replay_stats is not None
        stats=analysis.replay_dependency_closure(2)
        assert stats.replayed_records <= stats.reachable_records
        assert stats.reachable_records <= stats.total_cancellations
        again=analysis.replay_dependency_closure(2)
        assert stats == again
        assert analysis.witness(2,solver='replay') is event
