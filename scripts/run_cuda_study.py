"""Reproducible CUDA study; logs stay local until explicitly reviewed."""
import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gpu_snapshot():
    return subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version,power.draw,temperature.gpu,clocks.current.graphics,clocks.current.memory',
         '--format=csv'], text=True, timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--baseline-runs', type=int, default=30)
    parser.add_argument('--baseline-max', type=int, default=100000000)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    if args.runs < 2 or args.baseline_runs < 2:
        parser.error('At least two replicates are required')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT/'results'/('cuda_study_'+stamp)
    out.mkdir(parents=True)
    sizes = [10000, 100000, 1000000, 10000000, 100000000, 1000000000]
    dimensions = [2, 8, 16, 32]
    if args.quick:
        sizes, dimensions = [10000, 1000000], [8]
    jobs = []
    for n in sizes:
        for rng in ('separated', 'repeated'):
            for block in (128, 256, 512):
                jobs.append(('cuda', 'pi', n, 2, rng, block, args.runs))
        if n <= args.baseline_max:
            jobs.append(('sequential', 'pi', n, 2, 'separated', 0, args.baseline_runs))
    for d in dimensions:
        for rng in ('separated', 'repeated'):
            for block in (128, 256, 512):
                jobs.append(('cuda', 'integral', 1000000, d, rng, block, args.runs))
        jobs.append(('sequential', 'integral', 1000000, d, 'separated', 0, args.baseline_runs))
    # Fixed shuffle reduces systematic device warming/order effects and is repeatable.
    random.Random(20261029).shuffle(jobs)
    metadata = dict(created_utc=stamp, order_seed=20261029, warmups=3, jobs=[],
                    gpu_start=gpu_snapshot(), source_commit=subprocess.check_output(
                        ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    source_dirty=bool(subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True)))
    metadata['launch_configurations'] = [json.loads(subprocess.check_output(
        [str(ROOT/'build/cuda.exe'),'--describe','--block',str(b)], cwd=ROOT, text=True))
        for b in (128,256,512)]
    metadata['source_sha256'] = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                               for path in ('src/cuda/main.cu','src/common.hpp','src/sequential/main.cpp',
                                            'scripts/run_cuda_study.py','scripts/analyze.py','build/cuda.exe','build/sequential.exe')}
    try:
        for index, (backend, problem, n, d, rng, block, runs) in enumerate(jobs):
            options = ['--samples',str(n),'--runs',str(runs),'--warmups','3','--problem',problem,
                       '--dimensions',str(d),'--rng',rng,'--seed','1729000']
            if backend == 'cuda':
                options += ['--block',str(block)]
            command = [str(ROOT/'build'/f'{backend}.exe'), *options]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=600, check=True)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            if len(rows) != runs or any(r['status'] != 'implemented' for r in rows):
                raise RuntimeError('Incomplete or starter results')
            path = out/f'{index:03d}_{backend}_{problem}_{n}_d{d}_{rng}_b{block}.csv'
            path.write_text(result.stdout, encoding='utf-8')
            metadata['jobs'].append(dict(file=path.name, command=command, gpu_after=gpu_snapshot()))
            (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(f'{index+1}/{len(jobs)} {backend} {problem} N={n} d={d} {rng} block={block}', flush=True)
    finally:
        metadata['gpu_end'] = gpu_snapshot()
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze.py'),str(out)], check=True, cwd=ROOT)
    print(out)


if __name__ == '__main__':
    main()
