"""Resource-bounded, deterministic H1/H2 witness benchmark for PPG v0.9.0a4.

Run from the directory with patterns.py, with PYTHONPATH pointed at v0.9.0a4/src.
Prints one JSON event per stage. Each scenario uses one reusable PPGAnalysis. When requested, component
correction is run after replay, so the two may benefit from different caches.
The default component limit is 60,000 basis columns.
"""
import argparse
import json
import hashlib
import os
import resource
import threading
from time import perf_counter, sleep

from structured_patterns import volume
from persistence_pairing_graph import PPGAnalysis
from persistence_pairing_graph._reverse_replay import SelectiveInclusionReplay

class MemorySampler:
    def __init__(self):
        self.running=False
        self.peak=0
        self.thread=None
        self.path=f'/proc/{os.getpid()}/status'
    def __enter__(self):
        self.running=True
        self.peak=0
        def poll():
            while self.running:
                with open(self.path) as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            self.peak=max(self.peak,int(line.split()[1])*1024)
                            break
                sleep(.025)
        self.thread=threading.Thread(target=poll,daemon=True)
        self.thread.start()
        return self
    def __exit__(self,*_):
        self.running=False
        self.thread.join(timeout=.2)

def emit(**data):
    print(json.dumps(data,separators=(',',':')),flush=True)

def benchmark(n,pattern,max_basis,skip_component=False,only_dimension=None,max_component_faces=500_000,owner_storage='packed'):
    img=volume(n,pattern)
    emit(stage='start',size=n,pattern=pattern,foreground_voxels=int((img==0).sum()),cgroup_memory_max=(open('/sys/fs/cgroup/memory.max').read().strip()
        if os.path.isfile('/sys/fs/cgroup/memory.max') else 'unknown'))
    with MemorySampler() as meter:
        start=perf_counter()
        with PPGAnalysis(img,retain_inclusion_trace=True,max_basis_columns=max_basis,max_basis_entries=8_000_000,
                         max_chain_terms=300_000,max_work=5_000_000,max_replay_columns=150_000,
                         max_replay_visited=5_000_000,max_cached_witnesses=0) as a:
            elapsed=perf_counter()-start
            nodes={}
            for q in (1,2):
                indices=[i for i,x in enumerate(a.nodes) if x.dimension==q and not x.essential and x.death_value>x.birth_value]
                if indices:
                    # At same death event, prioritize the largest persistence then stable ID.
                    indices.sort(key=lambda i:(-float(a.nodes[i].death_value-a.nodes[i].birth_value),i))
                    if pattern=='pants_cavity' and q==1:
                        positives=[i for i in indices if len(a.targets(i))>0]
                        if positives:
                            indices=positives
                    nodes[q]=indices[0]
            trace=a.compressed.lift_trace
            emit(stage='built',size=n,pattern=pattern,seconds=round(elapsed,5),peak_mib=round(meter.peak/2**20,3),
                 original_cells=a.compressed.original_size,compressed_generators=len(a.compressed),
                 cancelled_pairs=trace.n_pairs,enriched_tail_entries=trace.replay_tail_entries,
                 selected=nodes, counts={str(q):sum(1 for x in a.nodes if x.dimension==q and not x.essential and x.death_value>x.birth_value) for q in (1,2)})
            # Separate the packed-owner-index cost from per-witness replay.
            with MemorySampler() as index_meter:
                t0=perf_counter()
                a._replay=SelectiveInclusionReplay(trace,owner_storage=owner_storage,
                    max_cached_columns=a.max_replay_columns,
                    max_visited_records=a.max_replay_visited,
                    max_influence_entries=a.max_basis_entries)
                emit(stage='index',size=n,pattern=pattern,seconds=round(perf_counter()-t0,5),
                     peak_mib=round(index_meter.peak/2**20,3),
                     index_bytes=a._replay.owner_index_bytes,owner_storage=a._replay.owner_storage)
            for q,node_id in nodes.items():
                if only_dimension is not None and q != only_dimension:
                    continue
                for solver in (('replay',) if skip_component else ('replay','component')):
                    with MemorySampler() as local_mem:
                        start=perf_counter()
                        try:
                            w=a.witness(node_id,solver=solver,max_component_faces=max_component_faces)
                            sec=perf_counter()-start
                            ans={'stage':'witness','size':n,'pattern':pattern,'q':q,'node_id':node_id,'solver':solver,'status':'ok',
                                 'seconds':round(sec,5),'peak_mib':round(local_mem.peak/2**20,3),
                                 'filling_cells':w.filling_cell_count,'relation_cells':len(w.relation_support),
                                 'target_count':len(w.target_nodes),'death_level':int(w.event_value),
                                 'filling_sha256':hashlib.sha256(w.filling_grid_indices.tobytes()).hexdigest()[:16],
                                 'relation_sha256':hashlib.sha256(w.relation_support.tobytes()).hexdigest()[:16]}
                            if solver=='replay':
                                s=a.last_replay_stats
                                ans.update(total_cancellations=s.total_cancellations,replayed_nonzero_steps=s.replayed_nonzero_steps,
                                           queried_original_columns=s.queried_original_columns,
                                           visited_record_instances=s.visited_record_instances,
                                           tails_read=s.tails_read,influence_entries=s.influence_entries,
                                           trace_record_bytes=s.trace_record_bytes,trace_tail_bytes=s.trace_tail_bytes,
                                           owner_index_bytes=a._replay.owner_index_bytes,owner_storage=a._replay.owner_storage,
                                           unique_influence_steps=len(set().union(*a._replay.cache.values())) if a._replay.cache else 0)
                            else:
                                s=a.last_component_stats
                                ans.update(eligible_deaths=s.eligible_deaths,selected_deaths=s.selected_deaths,visited_faces=s.visited_faces)
                            emit(**ans)
                        except (MemoryError,RuntimeError,ValueError) as e:
                            emit(stage='witness',size=n,pattern=pattern,q=q,node_id=node_id,solver=solver,
                                 status='error',error_type=type(e).__name__,error=str(e),
                                 seconds=round(perf_counter()-start,5),peak_mib=round(local_mem.peak/2**20,3))
    emit(stage='done',size=n,pattern=pattern,process_peak_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,3))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--pattern',choices=('mixed','lattice','pants_cavity'),default='mixed')
    p.add_argument('--max-basis',type=int,default=60000)
    p.add_argument('--skip-component',action='store_true')
    p.add_argument('--only-dimension',type=int,choices=(1,2))
    p.add_argument('--max-faces',type=int,default=500000)
    p.add_argument('--owner',choices=('dense','packed','dict','auto'),default='packed')
    args=p.parse_args()
    benchmark(args.size,args.pattern,args.max_basis,skip_component=args.skip_component,only_dimension=args.only_dimension,max_component_faces=args.max_faces,owner_storage=args.owner)
