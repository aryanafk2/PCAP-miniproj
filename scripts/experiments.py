"""Run measured implementations; starter outputs are never accepted."""
import argparse
import csv
import io
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', nargs='+', type=int, default=[10000, 100000, 1000000])
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--problem', choices=['pi', 'integral'], default='pi')
    parser.add_argument('--dimensions', type=int, default=8)
    parser.add_argument('--include-team-code', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'results' / stamp
    out.mkdir(parents=True)
    metadata = {'utc': stamp, 'platform': platform.platform(), 'runs': args.runs,
                'samples': args.samples, 'problem': args.problem, 'dimensions': args.dimensions,
                'scope': 'single laptop', 'commands': []}
    metadata['gpu'] = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv'], text=True)
    configs = [('sequential', [], [])]
    configs += [('cuda', ['--block', str(b)], []) for b in (128, 256, 512)]
    if args.include_team_code:
        configs += [('openmp', ['--workers', str(w)], []) for w in (1, 2, 4, 8)]
        configs += [('mpi', [], ['mpiexec', '-n', str(w)]) for w in (1, 2, 4, 8)]
    try:
        for n in args.samples:
            for rng in ('separated', 'repeated'):
                for backend, extra, prefix in configs:
                    command = [*prefix, str(ROOT/'build'/f'{backend}.exe'), '--samples', str(n),
                               '--runs', str(args.runs), '--rng', rng, '--problem', args.problem,
                               '--dimensions', str(args.dimensions), *extra]
                    result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
                    rows = list(csv.DictReader(io.StringIO(result.stdout)))
                    if len(rows) != args.runs or any(r['status'] != 'implemented' for r in rows):
                        raise RuntimeError('Refusing incomplete/starter results')
                    index = len(metadata['commands'])
                    (out/f'{index:03d}_{backend}_{n}_{rng}.csv').write_text(result.stdout, encoding='utf-8')
                    metadata['commands'].append(command)
                    print(backend, n, rng, extra or prefix, flush=True)
    finally:
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
