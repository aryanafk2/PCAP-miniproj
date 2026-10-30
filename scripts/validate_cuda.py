"""Check CUDA correctness with NVIDIA tools; keep diagnostic logs out of benchmarks.

Run after build_cuda.bat. Requires Compute Sanitizer from the CUDA Toolkit.
These checks detect memory/synchronization errors, not statistical independence.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitizer', type=Path)
    args = parser.parse_args()
    sanitizer = args.sanitizer or shutil.which('compute-sanitizer.exe')
    if not sanitizer:
        toolkit = Path(os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'))
        sanitizer = toolkit / 'compute-sanitizer/compute-sanitizer.exe'
    if not Path(sanitizer).is_file():
        parser.error('Compute Sanitizer not found; supply --sanitizer PATH')
    executable = ROOT / 'build/cuda.exe'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'results' / ('cuda_validation_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    # Tiny N leaves inactive sampling threads; odd N exercises loop remainders.
    # Both branches and stream modes must still reach every block barrier.
    cases = [('pi', 'separated', 2, 32, 2),
             ('pi', 'repeated', 40001, 1024, 2),
             ('integral', 'separated', 4097, 256, 64),
             ('integral', 'repeated', 40001, 128, 1)]
    for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
        for index, (problem, rng, samples, block, dimensions) in enumerate(cases):
            command = [str(sanitizer), '--tool', tool, '--error-exitcode', '99',
                       str(executable), '--problem', problem, '--rng', rng,
                       '--samples', str(samples), '--block', str(block),
                       '--dimensions', str(dimensions), '--runs', '1', '--warmups', '0']
            result = subprocess.run(command, capture_output=True, text=True,
                                    cwd=ROOT, timeout=180)
            log = output / f'{tool}_{index}.txt'
            log.write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            # Exit success alone is insufficient if instrumentation was skipped.
            combined = result.stdout + result.stderr
            clean = ('ERROR SUMMARY: 0 errors' in combined or
                     'RACECHECK SUMMARY: 0 hazards' in combined)
            passed = result.returncode == 0 and clean
            records.append(dict(tool=tool, command=command, passed=passed,
                                returncode=result.returncode, log=log.name))
            print(f'{tool} case {index}: {"PASS" if passed else "FAIL"}', flush=True)
    # Identical seeds must reproduce estimates regardless of excluded warm-ups.
    command = [str(executable), '--samples', '40001', '--runs', '2', '--seed', '98765']
    estimates = []
    for warmups in ('0', '3'):
        raw = subprocess.check_output(command + ['--warmups', warmups], text=True)
        estimates.append([(r['seed'], r['estimate']) for r in csv.DictReader(io.StringIO(raw))])
    reproducible = estimates[0] == estimates[1]
    manifest = dict(binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    checks=records, warmup_reproducibility=reproducible)
    (output / 'validation.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(output)
    if not reproducible or not all(r['passed'] for r in records):
        raise SystemExit('CUDA validation failed; inspect diagnostic logs.')
    print('PASS: 16 sanitizer cases and deterministic warm-up check')


if __name__ == '__main__':
    main()
