"""Research-only 3D selective replay closure measurements.

Run, e.g.:
    python benchmarks/investigate_dependency_closure.py --sizes 8 12 16
The JSON is deterministic with respect to the synthetic images; elapsed
seconds remain machine-dependent. Samples are surviving compressed GENERATORS
(not necessarily PPG death-event fillings), so this is a workload diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from persistence_pairing_graph._reverse_replay import SelectiveInclusionReplay
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming


def volume(n: int, name: str) -> np.ndarray:
    rng = np.random.default_rng(20261028)
    if name == "noise":
        return rng.integers(0, 16, (n,n,n), dtype=np.uint8)
    if name == "correlated":
        x = rng.integers(0, 16, (n,n,n), dtype=np.uint8).astype(np.float64)
        # Correlate on a 3x3x3 neighborhood (periodic synthetic test only).
        y = sum(np.roll(x,(i,j,k),axis=(0,1,2))
                for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)) / 27
        return np.clip(np.floor(y*2),0,255).astype(np.uint8)
    if name == "cavity":
        v = np.zeros((n,n,n), dtype=np.uint8)
        v[n//3:2*n//3,n//3:2*n//3,n//3:2*n//3]=12
        return v
    if name == "layers":
        z=np.arange(n,dtype=np.uint8)[:,None,None]
        field=np.broadcast_to(1+12*z//max(n-1,1), (n,n,n)).copy()
        field[:,n//3:2*n//3,n//3:2*n//3]//=2
        return field
    raise ValueError(name)


def run_case(n: int, name: str, sample_per_dim=8):
    data=volume(n,name)
    before=perf_counter()
    complex_=compress_discrete_tensor_streaming(
        data, retain_inclusion_trace=True, retain_chain_lifts=True)
    compression_s=perf_counter()-before
    try:
        before=perf_counter()
        replay=SelectiveInclusionReplay(complex_.lift_trace, owner_storage='packed')
        index_s=perf_counter()-before
        outputs=[]
        for dim in (1,2):
            ids=np.flatnonzero(complex_.dimensions == dim)
            # Uniformly span the actual compressed filtration order.
            selected=ids[np.unique(np.linspace(
                0,len(ids)-1,min(len(ids),sample_per_dim),dtype=int))] if len(ids) else []
            for idx in selected:
                before=perf_counter()
                answer=replay.lift([int(complex_.original_grid_indices[idx])],
                                   collect_dependency_stats=True,
                                   max_output_cells=1_000_000)
                elapsed=perf_counter()-before
                st=replay.last_closure_stats
                outputs.append({
                    "dimension":dim,"query_index":int(idx),
                    "reachable":st.reachable_records,
                    "replayed":st.replayed_records,
                    "arcs":st.dependency_arcs,"depth":st.maximum_observed_depth,
                    "fraction":st.reachable_fraction,"time_s":round(elapsed,6),
                    "output_cells":len(answer),
                })
        values=[x["fraction"] for x in outputs]
        return {
            "size":n,"pattern":name,"queries":len(outputs),
            "original_grid_cells":complex_.original_size,
            "compressed_generators":len(complex_),
            "canceled_pairs":complex_.lift_trace.n_pairs,
            "trace_record_bytes":complex_.lift_trace.n_pairs*20,
            "trace_tail_bytes":complex_.lift_trace.replay_tail_entries*4,
            "compression_s":round(compression_s,5),"owner_index_s":round(index_s,5),
            "median_fraction":round(float(median(values)),7) if values else 0.0,
            "maximum_fraction":round(max(values),7) if values else 0.0,
            "max_depth":max((v["depth"] for v in outputs),default=0),
            "median_replayed":median((v["replayed"] for v in outputs)) if outputs else 0,
            "samples":outputs,
        }
    finally:
        complex_.lift_trace.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sizes",nargs="+",type=int,default=[8,12,16])
    parser.add_argument("--patterns",nargs="+",
                        default=["noise","correlated","cavity","layers"])
    parser.add_argument("--samples",type=int,default=8)
    parser.add_argument("--json",type=Path)
    args=parser.parse_args()
    results=[]
    for n in args.sizes:
        for pattern in args.patterns:
            result=run_case(n,pattern,args.samples)
            results.append(result)
            print(f"CLOSURE n={n} pattern={pattern} pairs={result['canceled_pairs']} "
                  f"queries={result['queries']} median_fraction={result['median_fraction']:.6f} "
                  f"max_fraction={result['maximum_fraction']:.6f} "
                  f"max_depth={result['max_depth']} "
                  f"compression_s={result['compression_s']:.4f}",flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
