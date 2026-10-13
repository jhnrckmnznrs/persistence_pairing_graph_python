"""Verify edge-for-edge equivalence of full and pruned death-event solvers."""
import argparse, json, resource
from time import perf_counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.profile_structured import make_volume
from persistence_pairing_graph._streaming import compress_discrete_tensor_streaming
from persistence_pairing_graph._reduction import reduce_boundary
from persistence_pairing_graph._graph import build_nodes,iter_edges


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',type=int,default=48)
    p.add_argument('--case',choices=['random','checkerboard','multi-cavity'],default='checkerboard')
    args=p.parse_args()
    a=make_volume(args.case,args.size)
    c=compress_discrete_tensor_streaming(a)
    r=reduce_boundary(c)
    try:
        nodes=build_nodes(c.values,c.dimensions,r)
        times={}
        previous=None
        n_edges=None
        for enabled in (False,True):
            start=perf_counter()
            edges=[(e.dimension,e.from_node,e.to_node) for e in iter_edges(
                c.values,r,nodes,skip_empty_survivor_events=enabled)]
            times['pruned' if enabled else 'full']=round(perf_counter()-start,6)
            if previous is not None and previous!=edges:
                raise AssertionError('Event pruning changed the ordered directed edge list')
            previous=edges
            n_edges=len(edges)
        print(json.dumps({'case':args.case,'size':args.size,'edges':n_edges,'equivalent':True,
                          'full_solver_seconds':times['full'],
                          'pruned_solver_seconds':times['pruned'],
                          'peak_process_rss_mib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)}))
    finally:
        r.close()

if __name__=='__main__':main()
