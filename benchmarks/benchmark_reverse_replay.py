"""Separate-process memory/timing comparison for optional enriched replay traces.

Run: python benchmarks/benchmark_reverse_replay.py --sizes 12 16 24 --json results.json
Each observation is a *single measurement*, not a statistically stable result.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np

from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from persistence_pairing_graph._reverse_replay import SelectiveInclusionReplay
from persistence_pairing_graph._lift import boundary_old_chain


def run_one(size, pattern, trace_type, owner_storage):
    if pattern == 'random':
        image = np.random.default_rng(70).integers(0,32,(size,)*3,dtype=np.uint8)
    else:
        image = np.zeros((size,)*3,dtype=np.uint8)
        image[size//2,size//2,size//2] = 200
    began = perf_counter()
    compressed = compress_discrete_tensor_streaming(image,retain_chain_lifts=True,
                              retain_inclusion_trace=(trace_type=='enriched'))
    compression_s=perf_counter()-began
    out = {'size':size,'pattern':pattern,'trace':trace_type,'owner_storage':owner_storage,
           'original_cells':compressed.original_size,'generators':len(compressed),
           'canceled_pairs':compressed.lift_trace.n_pairs,'compression_s':compression_s,
           'pair_trace_bytes':compressed.lift_trace.n_pairs*8,
           'replay_record_bytes':(compressed.lift_trace.n_pairs*20 if trace_type=='enriched' else 0),
           'tail_bytes':(compressed.lift_trace.replay_tail_entries*4 if trace_type=='enriched' else 0)}
    if trace_type == 'enriched':
        began=perf_counter()
        replay=SelectiveInclusionReplay(compressed.lift_trace,owner_storage=owner_storage)
        out['index_s']=perf_counter()-began
        out['owner_index_bytes']=replay.owner_index_bytes
        selected=next((j for j in range(len(compressed)) if compressed.dimensions[j]>0),0)
        began=perf_counter()
        result=replay.lift([int(compressed.original_grid_indices[selected])])
        out['one_lift_s']=perf_counter()-began
        boundary=boundary_old_chain(result,tuple(2*size+1 for _ in range(3)))
        expected=replay.lift(compressed.original_grid_indices[compressed.boundary(selected)])
        assert np.array_equal(boundary,expected)
        out['output_cells']=len(result)
        out['visited_record_instances']=replay.last_stats.visited_record_instances
        out['cache_columns']=replay.last_stats.queried_original_columns
    out['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    compressed.lift_trace.close()
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sizes',type=int,nargs='+',default=[12,16,24])
    parser.add_argument('--patterns',nargs='+',default=['random','cavity'])
    parser.add_argument('--json',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--size',type=int)
    parser.add_argument('--pattern')
    parser.add_argument('--trace',default='enriched')
    parser.add_argument('--owner',default='packed')
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(run_one(args.size,args.pattern,args.trace,args.owner)))
        return
    data=[]
    for size in args.sizes:
        for pattern in args.patterns:
            for trace_type, owner in [('ordinary','none'),('enriched','packed'),('enriched','dict')]:
                command=[sys.executable,__file__,'--worker','--size',str(size),
                         '--pattern',pattern,'--trace',trace_type,'--owner',owner]
                env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
                p=subprocess.run(command,env=env,check=True,capture_output=True,text=True)
                result=json.loads(p.stdout)
                data.append(result)
                print(f'{size:3}³ {pattern:6s} {trace_type:8s} {owner:6s} '
                      f'{result["compression_s"]:.3f}s peak {result["peak_rss_mib"]:.1f} MiB')
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(data,indent=2))

if __name__=='__main__':
    main()
