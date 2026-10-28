"""PPG death-event (not just generator) 3D replay closure diagnostics.

This benchmark is opt-in, serial, and resource bounded. Synthetic fields are
not substitutes for measured porous-media or micro-CT volumes.
"""
import argparse
from time import perf_counter
from persistence_pairing_graph import PPGAnalysis, SelectiveProjectionReplay
from investigate_dependency_closure import volume


def run_case(size, pattern, max_events=2):
    image=volume(size,pattern)
    before=perf_counter()
    selected=[]
    with PPGAnalysis(image,retain_inclusion_trace=True,
                     max_original_cells=1_000_000,
                     max_replay_columns=50_000,
                     max_replay_visited=1_000_000) as analysis:
        prep=perf_counter()-before
        # Prioritize H1 and H2 where the event certificate is most useful.
        for q in (1,2,0):
            selected.extend(i for i,n in enumerate(analysis.nodes)
                            if not n.essential and n.dimension==q
                            and n.birth_value<n.death_value)
            if len(selected)>=max_events:
                break
        results=[]
        projector=SelectiveProjectionReplay(analysis.compressed.lift_trace)
        for i in selected[:max_events]:
            start=perf_counter()
            try:
                st=analysis.replay_dependency_closure(i)
                witness=analysis.witness(i,solver='replay')
                start_p=perf_counter()
                projected=projector.project(
                    witness.relation_support,stop_before_value=witness.event_value)
                projection_s=perf_counter()-start_p
                ps=projector.last_projection_stats
                results.append({"source":i,"dimension":analysis.nodes[i].dimension,
                    "reach":st.reachable_records,"replayed":st.replayed_records,
                    "fraction":round(st.reachable_fraction,8),
                    "depth":st.maximum_observed_depth,
                    "arcs":st.dependency_arcs,
                    "time_s":round(perf_counter()-start,6),
                    "cycle_cells":len(witness.relation_support),
                    "projection_touched":ps.processed_cancellations,
                    "projection_tail_terms":ps.tail_terms_read,
                    "projected_cells":len(projected),
                    "projection_s":round(projection_s,6)})
            except (MemoryError,RuntimeError) as ex:
                results.append({"source":i,"error":type(ex).__name__,
                                "message":str(ex)[:120]})
        return {"size":size,"pattern":pattern,"preparation_s":round(prep,6),
                "canceled_pairs":analysis.compressed.lift_trace.n_pairs,
                "event_samples":results}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sizes",nargs="+",type=int,default=[5,7,9])
    parser.add_argument("--patterns",nargs="+",
                        default=["noise","correlated","cavity","layers"])
    args=parser.parse_args()
    for n in args.sizes:
        for pattern in args.patterns:
            try:
                print("EVENT_CLOSURE",run_case(n,pattern),flush=True)
            except (MemoryError,RuntimeError) as ex:
                print("EVENT_CLOSURE",{"size":n,"pattern":pattern,
                                     "error":type(ex).__name__,
                                     "message":str(ex)[:120]},flush=True)


if __name__=="__main__":
    main()
