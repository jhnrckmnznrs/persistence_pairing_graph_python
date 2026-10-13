"""Benchmark 3D scalar-field families in independent Python processes.

Example:
  python benchmarks/run_structured_suite.py --sizes 24 48 64 --output results.json

Peak RSS is a per-process Linux/macOS resource measurement. Inputs are generated
from fixed formulas and a fixed RNG seed, not bundled datasets. Each subprocess
prints one JSON object with construction, reduction, and graph timing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_CASES=('shell','torus','multi-cavity','lattice','checkerboard','smooth','random')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',type=int,nargs='+',default=[24,48])
    parser.add_argument('--cases',nargs='+',choices=DEFAULT_CASES,default=DEFAULT_CASES)
    parser.add_argument('--storage',choices=['packed','python','disk'],default='packed')
    parser.add_argument('--order-format',choices=['u32','delta'],default='u32')
    parser.add_argument('--timeout',type=int,default=180)
    parser.add_argument('--output',type=Path,default=Path('structured_profile.json'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    script=root/'benchmarks'/'profile_structured.py'
    env=os.environ.copy()
    env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')
    results=[]
    for size in args.sizes:
        for case in args.cases:
            cmd=[sys.executable,str(script),'--case',case,'--size',str(size),
                 '--storage',args.storage,'--order-format',args.order_format]
            print(f'Profiling {case} {size}³...',flush=True)
            try:
                p=subprocess.run(cmd,env=env,capture_output=True,text=True,
                                 check=True,timeout=args.timeout)
                record=json.loads(p.stdout.strip().splitlines()[-1])
            except (subprocess.SubprocessError,ValueError,IndexError) as exc:
                record={'case':case,'size':size,'error':str(exc)}
            results.append(record)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(f'Saved {len(results)} results to {args.output}')
    if any('error' in record for record in results):
        raise SystemExit(1)

if __name__=='__main__':main()
