"""Controlled comparisons of sparse old-level queries and full old-cell reference.

Usage:
  python benchmarks/benchmark_old_cohomology.py --sizes 5 7 9 --events 2

Synthetic 3D arrays, not real porous-media or bone scans. Timings are
one-shot diagnostics; setup/reduction and oracle indexing are reported
separately. The full reference reconstructs a witness even when the sparse
mode requests only a feasibility decision, so this is NOT a uniform-workload
end-to-end speed comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from persistence_pairing_graph import (
    PPGAnalysis, SparseOldLevelOracle, certify_relative_filling,
)
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._transition_witness import _original_cell_birth
from investigate_dependency_closure import volume


def run_case(n, pattern, max_events=2):
    image=volume(n,pattern)
    setup_begin=perf_counter()
    rows=[]
    with PPGAnalysis(image,retain_inclusion_trace=True,
                     max_original_cells=1_000_000) as a:
        setup_time=perf_counter()-setup_begin
        start=perf_counter()
        oracle=SparseOldLevelOracle(a)
        oracle_time=perf_counter()-start
        picked=[]
        for q in (1,2,0):
            for i,node in enumerate(a.nodes):
                if node.dimension==q and not node.essential and (
                        node.birth_value < node.death_value):
                    picked.append(i)
                    if len(picked)>=max_events:
                        break
            if len(picked)>=max_events:
                break
        for node_id in picked:
            w=a.witness(node_id,solver='replay')
            event=w.event_value
            new=[int(cell) for cell in w.filling_grid_indices
                 if _original_cell_birth(image,int(cell),a.grid_shape)==event]
            row={"node":node_id,"dimension":w.dimension,
                 "event":int(event),"relation_cells":len(w.relation_support),
                 "permitted_new":len(new)}
            for label, candidates in (("positive",new),("negative",[])):
                if not candidates and label=="positive":
                    continue
                begin=perf_counter()
                coordinate=oracle.restricted_relative(
                    w.relation_support,candidates,dimension=w.dimension,
                    event_value=event,reconstruct=False)
                coord_s=perf_counter()-begin
                begin=perf_counter()
                complete=oracle.restricted_relative(
                    w.relation_support,candidates,dimension=w.dimension,
                    event_value=event,reconstruct=True)
                sparse_complete_s=perf_counter()-begin
                assert coordinate.feasible==complete.feasible
                begin=perf_counter()
                reference=certify_relative_filling(
                    image,w.relation_support,candidates,
                    dimension=w.dimension,event_value=event,
                    old_solver='full',max_grid_cells=1_000_000,
                    max_old_cells=100_000,max_boundary_rows=150_000)
                full_s=perf_counter()-begin
                assert coordinate.feasible==reference.feasible
                if complete.feasible:
                    assert np.array_equal(
                        boundary_old_chain(
                            [*complete.old_filling_cells,*complete.new_filling_cells],
                            a.grid_shape),w.relation_support)
                row[label]={"feasible":coordinate.feasible,
                     "coordinate_s":round(coord_s,6),
                     "sparse_complete_s":round(sparse_complete_s,6),
                     "reference_full_s":round(full_s,6),
                     "full_old_columns":reference.old_candidate_count,
                     "relative_kernel":coordinate.relative_kernel_generators,
                     "coordinate_rank":coordinate.coordinate_rank,
                     "new_filling_cells":len(complete.new_filling_cells)}
            rows.append(row)
        return {"n":n,"pattern":pattern,
                "initial_ppg_s":round(setup_time,6),
                "oracle_index_s":round(oracle_time,6),
                "canceled_pairs":a.compressed.lift_trace.n_pairs,
                "compressed_generators":len(a.compressed),
                "events":rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sizes",nargs="+",type=int,default=[5,7,9])
    p.add_argument("--patterns",nargs="+",default=["noise","correlated","cavity"])
    p.add_argument("--events",type=int,default=2)
    p.add_argument("--json",type=Path)
    args=p.parse_args()
    out=[]
    for n in args.sizes:
        for pattern in args.patterns:
            try:
                report=run_case(n,pattern,args.events)
                out.append(report)
                for row in report["events"]:
                    pos=row.get("positive",{})
                    neg=row.get("negative",{})
                    print(
                        f"OLD_COHOMOLOGY n={n} pattern={pattern} q={row['dimension']} "
                        f"setup_s={report['initial_ppg_s']:.4f} "
                        f"index_s={report['oracle_index_s']:.4f} "
                        f"positive={pos.get('feasible','na')} "
                        f"coord_s={pos.get('coordinate_s','na')} "
                        f"complete_s={pos.get('sparse_complete_s','na')} "
                        f"full_s={pos.get('reference_full_s','na')} "
                        f"negative={neg.get('feasible','na')}",flush=True)
                if not report["events"]:
                    print(f"OLD_COHOMOLOGY n={n} pattern={pattern} no-events",flush=True)
            except (RuntimeError,ValueError,MemoryError) as error:
                # Results in failed/unsupported cases must remain visible.
                print(f"OLD_COHOMOLOGY n={n} pattern={pattern} "
                      f"ERROR={type(error).__name__}: {error}",flush=True)
                raise
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
