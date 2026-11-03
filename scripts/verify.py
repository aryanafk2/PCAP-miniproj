"""Build first, then run: python scripts/verify.py. No extra dependencies."""
import csv
import io
import math
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(name, *args, mpi=False, workers=2, success=True):
    command = [str(ROOT / 'build' / (name + '.exe')), *map(str, args)]
    if mpi:
        command = [shutil.which('mpiexec') or 'mpiexec', '-n', str(workers), *command]
    result = subprocess.run(command, text=True, capture_output=True, cwd=ROOT, timeout=90)
    if not success:
        assert result.returncode != 0, command
        return []
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert rows
    for row in rows:
        assert math.isfinite(float(row['estimate']))
        assert float(row['total_ms']) > 0
        if row['backend'] == 'cuda':
            assert 0 < float(row['kernel_ms']) <= float(row['total_ms'])
    return rows


def main():
    for problem in ('pi', 'integral'):
        for rng in ('separated', 'repeated'):
            for samples in (2, 4095, 4096, 4097, 100000):
                args = ('--samples', samples, '--runs', 2, '--warmups', 1,
                        '--problem', problem, '--rng', rng, '--dimensions', 8)
                baseline = run('sequential', *args)
                for workers in (1, 2, 4, 8):
                    for name, mpi in [('openmp', False), ('mpi', True)]:
                        rows = run(name, *args, '--workers', workers, mpi=mpi, workers=workers)
                        assert len(rows) == len(baseline)
                        for actual, expected in zip(rows, baseline):
                            assert actual['status'] == 'implemented'
                            assert int(actual['workers']) == workers
                            assert actual['seed'] == expected['seed']
                            # Pi sums are exact integers. Integral reductions
                            # may differ in their last bits due to addition order.
                            for field in ('estimate', 'ci95_low', 'ci95_high'):
                                assert math.isclose(float(actual[field]), float(expected[field]),
                                                    rel_tol=1e-12, abs_tol=1e-12), (name, field)
    for problem in ('pi', 'integral'):
        serial = run('sequential', '--samples', 100000, '--runs', 2, '--problem', problem)
        truth = math.pi if problem == 'pi' else 1/3
        assert all(abs(float(r['estimate'])-truth) < 0.05 for r in serial)
        for rng in ('separated', 'repeated'):
            for block in (128, 256, 512):
                rows = run('cuda', '--samples', 100000, '--runs', 2, '--problem', problem,
                           '--rng', rng, '--block', block)
                assert len(rows) == 2
                assert [r['seed'] for r in rows] == ['12345', '12346']
    run('sequential', '--samples', 0, success=False)
    run('cuda', '--block', 127, success=False)
    run('openmp', '--workers', 0, success=False)
    run('mpi', '--samples', 0, mpi=True, success=False)
    print('PASS: four backends, 1/2/4/8 CPU workers, both problems/RNGs, remainders, seeds and invalid inputs')


if __name__ == '__main__':
    main()
