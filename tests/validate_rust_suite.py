"""Run actual Rust/Python differential validation on reproducible 2D/3D TIFFs.

From the Python package directory:

  python tests/validate_rust_suite.py --rust-bin /path/to/persistence_pairing_graph
  python tests/validate_rust_suite.py --rust-repo /path/to/rust/repository
  python tests/validate_rust_suite.py --rust-repo /path/to/rust/repository --algorithms stream-block block lazy --json results.json

The Rust repo is never modified. If --rust-repo is supplied, `cargo build
--release` is performed there and the resulting binary is executed. Requires
`tifffile` and NumPy. Both programs write to isolated temporary directories.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

# Allow running this file directly without installing the source package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_rust import compare_exports

# Benchmarks are development utilities, not package dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.profile_structured import make_volume

PANTS = np.array([
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,30,30,30,30,30,30,30,30,30,0],
    [0,30,10,10,10,30,20,20,20,30,0],
    [0,30,10,90,10,30,20,80,20,30,0],
    [0,30,10,10,10,30,20,20,20,30,0],
    [0,30,30,30,30,30,30,30,30,30,0],
    [0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.uint8)


def fixtures(size: int, seed: int, include_uint16: bool):
    if size < 8:
        raise ValueError('Size must be at least 8 to resolve 3D shell/torus fixtures')
    cases = [('pants', PANTS), ('constant', np.full((4,7,8),12,dtype=np.uint8)),
             ('monotone', np.arange(size**3,dtype=np.uint32).reshape(size,size,size)%256)]
    # Unwrapped monotone field with a compact value range.
    cases[-1] = ('monotone', np.broadcast_to(np.arange(size,dtype=np.uint8)[:,None,None],
                                             (size,size,size)).copy())
    for case in ('shell','torus','multi-cavity','lattice','checkerboard','smooth','random'):
        cases.append((case, make_volume(case,size,seed=seed)))
    for name, data in cases:
        yield name+'_u8', data
        if include_uint16:
            yield name+'_u16', data.astype(np.uint16)*np.uint16(257)+np.uint16(7)


def _run(command: list[str], *, timeout: float, cwd: Path | None = None) -> float:
    start=perf_counter()
    proc=subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'Exit {proc.returncode}: {command!r}\n'
                           f'--- STDOUT ---\n{proc.stdout[-3000:]}\n'
                           f'--- STDERR ---\n{proc.stderr[-5000:]}')
    return round(perf_counter()-start,5)


def validate(args) -> list[dict]:
    import tifffile
    if args.rust_bin:
        rust_bin=args.rust_bin.resolve()
    else:
        repo=args.rust_repo.resolve()
        if not (repo/'Cargo.toml').is_file():
            raise FileNotFoundError(f'No Cargo.toml in {repo}')
        cargo=shutil.which('cargo')
        if cargo is None:
            raise RuntimeError('cargo not available; provide --rust-bin or install Rust')
        _run([cargo,'build','--release','--manifest-path',str(repo/'Cargo.toml')],
             timeout=args.build_timeout)
        rust_bin=repo/'target'/'release'/('persistence_pairing_graph.exe' if os.name=='nt' else 'persistence_pairing_graph')
    if not rust_bin.is_file():
        raise FileNotFoundError(f'Rust binary not found: {rust_bin}')
    root_context=(tempfile.TemporaryDirectory(prefix='ppg_crosslang_')
                  if args.keep is None else None)
    root=Path(root_context.name if root_context else args.keep.resolve())
    root.mkdir(parents=True,exist_ok=True)
    report=[]
    try:
        for name, volume in fixtures(args.size,args.seed,args.uint16):
            for algorithm in args.algorithms:
                work=root/algorithm/name
                a,b=work/'rust',work/'python'
                a.mkdir(parents=True,exist_ok=True)
                b.mkdir(parents=True,exist_ok=True)
                # Independent identical byte streams for both executables.
                for path in (a/'sample.tif',b/'sample.tif'):
                    tifffile.imwrite(path,volume,photometric='minisblack',metadata=None)
                opts=['--algorithm',algorithm]
                if algorithm == 'stream-block':
                    opts+=['--export-cell-provenance']
                print(f'Checking {algorithm:12s} {name:20s} {volume.shape}',flush=True)
                outcome={'case':name,'algorithm':algorithm,'shape':list(volume.shape),'dtype':str(volume.dtype)}
                try:
                    outcome['rust_seconds']=_run([str(rust_bin),str(a/'sample.tif'),*opts],timeout=args.timeout)
                    outcome['python_seconds']=_run([sys.executable,'-m','persistence_pairing_graph',
                                                    str(b/'sample.tif'),*opts],timeout=args.timeout)
                    outcome.update(compare_exports(a,b,'sample'))
                    outcome['equivalent']=True
                    print(f"  PASS nodes={outcome['nodes']} edges={outcome['edges']}",flush=True)
                except Exception as exc:
                    outcome['equivalent']=False
                    outcome['error']=str(exc)
                    print(f"  FAIL {exc}",file=sys.stderr,flush=True)
                report.append(outcome)
                if args.json:
                    args.json.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report
    finally:
        if root_context:
            root_context.cleanup()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    select=p.add_mutually_exclusive_group(required=True)
    select.add_argument('--rust-bin',type=Path)
    select.add_argument('--rust-repo',type=Path)
    p.add_argument('--algorithms',choices=['stream-block','block','lazy'],nargs='+',default=['stream-block'])
    p.add_argument('--size',type=int,default=16)
    p.add_argument('--seed',type=int,default=271828)
    p.add_argument('--uint16',action='store_true',help='Also exercise high-range 16-bit grayscale TIFFs')
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--build-timeout',type=float,default=1200)
    p.add_argument('--json',type=Path,help='Save per-fixture results, including failure details')
    p.add_argument('--keep',type=Path,help='Keep all TIFF/CSV outputs under this path')
    args=p.parse_args()
    result=validate(args)
    failed=[r for r in result if not r['equivalent']]
    print(f'Cross-language validation: {len(result)-len(failed)}/{len(result)} passed')
    if failed: sys.exit(1)

if __name__=='__main__': main()
