"""Synthetic 3D forward-projection dependency measurements.

A diagnostic of the stored cancellation projection, not a benchmark on
clinical or porous-media input and not an end-to-end PPG runtime comparison.
"""
from __future__ import annotations
import argparse
from statistics import median
from time import perf_counter
import numpy as np
from persistence_pairing_graph import SelectiveProjectionReplay
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from investigate_dependency_closure import volume


def measure(size,pattern,samples=8):
    image=volume(size,pattern)
    before=perf_counter()
    compressed=compress_discrete_tensor_streaming(
        image,retain_inclusion_trace=True)
    compression=perf_counter()-before
    try:
        projection=SelectiveProjectionReplay(compressed.lift_trace,owner_storage='packed')
        pair=compressed.lift_trace.pairs
        records=compressed.lift_trace.replay_records
        outputs=[]
        for q in (1,2):
            candidates=np.flatnonzero(pair['dimension']==q+1)
            for k in np.unique(np.linspace(
                    0,len(candidates)-1,min(len(candidates),samples),dtype=int)) if len(candidates) else []:
                j=int(candidates[k])
                src=int(records[j]['birth'])
                start=perf_counter()
                chain=projection.project([src])
                st=projection.last_projection_stats
                outputs.append({"dimension":q,"touched":st.processed_cancellations,
                    "fraction":round(st.processed_cancellations/max(st.total_cancellations,1),8),
                    "tails":st.tail_terms_read,"result_cells":len(chain),
                    "time_s":round(perf_counter()-start,6)})
        return {"n":size,"pattern":pattern,"pairs":projection.n,
                "compression_s":round(compression,6),"queries":len(outputs),
                "median_touched":median(x['touched'] for x in outputs) if outputs else 0,
                "median_fraction":round(float(median(x['fraction'] for x in outputs)),8) if outputs else 0,
                "max_fraction":max((x['fraction'] for x in outputs),default=0),
                "max_touched":max((x['touched'] for x in outputs),default=0),
                "details":outputs}
    finally:
        compressed.lift_trace.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',nargs='+',type=int,default=[8,16,24,32])
    p.add_argument('--patterns',nargs='+',default=['noise','correlated','cavity'])
    p.add_argument('--samples',type=int,default=8)
    a=p.parse_args()
    for n in a.sizes:
        for pattern in a.patterns:
            v=measure(n,pattern,a.samples)
            print(f"PROJECTION n={n} pattern={pattern} pairs={v['pairs']} "
                  f"queries={v['queries']} median_touched={v['median_touched']} "
                  f"median_fraction={v['median_fraction']:.7f} "
                  f"max_fraction={v['max_fraction']:.7f} "
                  f"max_touched={v['max_touched']}",flush=True)


if __name__=='__main__':
    main()
