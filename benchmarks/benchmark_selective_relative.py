"""Bounded comparison of full old enumeration vs exact incidence selection.

Uses controlled 3D filtration-level examples; these are *locality probes*,
not general PPG witness or clinical image benchmarks.
"""
import argparse
from math import prod
from time import perf_counter

import numpy as np

from persistence_pairing_graph import certify_relative_filling
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._transition_witness import _original_cell_birth
from investigate_dependency_closure import volume


def run_case(n, pattern):
    image=volume(n,pattern)
    grid=tuple(2*n+1 for _ in range(3))
    event={"noise":8,"correlated":15,"cavity":12,"layers":7}[pattern]
    # Identify just one new square whose boundary is already old, if any.
    # Scanning to prepare this workload is EXCLUDED from solver runtimes.
    source=None
    for cell in range(prod(grid)):
        if sum(c&1 for c in np.unravel_index(cell,grid))!=2:
            continue
        if _original_cell_birth(image,cell,grid)!=event:
            continue
        faces=boundary_old_chain([cell],grid)
        if all(_original_cell_birth(image,int(f),grid)<event for f in faces):
            source=(cell,faces)
            break
    if source is None:
        return {"size":n,"pattern":pattern,"skipped":"no suitable one-square event"}
    chosen, relation=source
    out={"size":n,"pattern":pattern,"event":event,"new_cell":chosen}
    for old_solver in ("full","component"):
        before=perf_counter()
        cert=certify_relative_filling(
            image,relation,[chosen],dimension=1,event_value=event,
            max_grid_cells=1_000_000,max_old_cells=30_000,
            max_new_cells=100,max_boundary_rows=100_000,
            max_component_faces=100_000,max_examined_cofaces=200_000,
            old_solver=old_solver)
        elapsed=perf_counter()-before
        assert cert.feasible
        assert np.array_equal(boundary_old_chain(
            [*cert.old_filling_cells,*cert.new_filling_cells],grid),relation)
        out[old_solver+"_s"]=round(elapsed,6)
        out[old_solver+"_old_columns"]=cert.old_candidate_count
        out[old_solver+"_examined"]=cert.old_cells_examined
        out[old_solver+"_visited_faces"]=cert.old_faces_visited
    assert out["component_old_columns"]<=out["full_old_columns"]
    out["old_column_fraction"]=round(
        out["component_old_columns"]/max(out["full_old_columns"],1),6)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sizes",nargs="+",type=int,default=[5,7,9])
    parser.add_argument("--patterns",nargs="+",
                        default=["noise","correlated","cavity","layers"])
    args=parser.parse_args()
    for size in args.sizes:
        for pattern in args.patterns:
            out=run_case(size,pattern)
            print("RELATIVE",out,flush=True)


if __name__=="__main__":
    main()
