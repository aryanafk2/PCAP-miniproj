"""Capture CUDA counters and derive the FP64 roofline using Nsight's formulas.

Diagnostic replay timings must never be merged into the benchmark CSVs.
Theoretical ceilings use the measured profiling clocks, not advertised boost.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def derive(raw):
    # Nsight raw CSV is wide: header, unit row, then one row per kernel.
    rows = list(csv.DictReader(io.StringIO(raw)))
    if len(rows) != 2:
        raise ValueError('Expected units and exactly one profiled kernel')
    row = rows[1]

    def metric(name):
        # Nsight follows the Windows locale's digit grouping (including India).
        return float(row[name].replace(',', ''))

    clock = metric('smsp__cycles_elapsed.avg.per_second')
    operations_per_cycle = sum(
        factor * metric(f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum.per_cycle_elapsed')
        for op, factor in [('dadd', 1), ('dmul', 1), ('dfma', 2)])
    achieved = operations_per_cycle * clock
    peak = (2 * metric('sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained')
            * metric('sm__cycles_elapsed.avg.per_second'))
    bandwidth = metric('dram__bytes.sum.per_second')
    peak_bandwidth = (metric('dram__bytes.sum.peak_sustained')
                      * metric('dram__cycles_elapsed.avg.per_second'))
    return dict(profile_kernel_ms=metric('gpu__time_duration.sum') / 1e6,
                fp64_gflops=achieved / 1e9, fp64_peak_gflops=peak / 1e9,
                fp64_peak_percent=100 * achieved / peak,
                dram_bytes_per_second=bandwidth,
                dram_peak_bytes_per_second=peak_bandwidth,
                arithmetic_intensity=achieved / bandwidth if bandwidth > 0 else None,
                achieved_occupancy_percent=metric('sm__warps_active.avg.pct_of_peak_sustained_active'),
                sm_clock_ghz=clock / 1e9,
                compute_throughput_percent=metric('sm__throughput.avg.pct_of_peak_sustained_elapsed'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ncu', type=Path, default=Path(
        r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for problem, samples, dimensions in [('pi', 100000000, 2),
                                          ('pi', 1000000000, 2),
                                          ('integral', 1000000, 8)]:
        name = f'{problem}_{samples}_d{dimensions}'
        report = args.output / name
        command = [str(args.ncu), '--set', 'full', '--launch-skip', '3',
                   '--launch-count', '1', '--clock-control', 'base',
                   '--export', str(report), '--force-overwrite',
                   str(ROOT / 'build/cuda.exe'), '--samples', str(samples),
                   '--problem', problem, '--dimensions', str(dimensions),
                   '--block', '256', '--runs', '1', '--warmups', '3']
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (args.output / (name + '_capture.txt')).write_text(
            result.stdout + '\n' + result.stderr, encoding='utf-8')
        if result.returncode or '==ERROR==' in result.stdout + result.stderr:
            raise RuntimeError('Profiling failed; inspect capture log')
        report = str(report) + '.ncu-rep'
        for page in ('raw', 'details'):
            export = [str(args.ncu), '--import', report, '--page', page]
            if page == 'raw':
                export += ['--csv', '--print-units', 'base']
            text = subprocess.check_output(export, text=True, timeout=60)
            (args.output / (name + ('_raw.csv' if page == 'raw' else '_details.txt'))).write_text(text, encoding='utf-8')
            if page == 'raw':
                counters = derive(text)
        records.append(dict(problem=problem, samples=samples, dimensions=dimensions,
                            block=256, **counters))
        print(name, json.dumps(counters), flush=True)
    with (args.output / 'roofline_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / 'profile_metadata.json').write_text(json.dumps(dict(
        binary_sha256=hashlib.sha256((ROOT / 'build/cuda.exe').read_bytes()).hexdigest(),
        clock_control='base', warmups=3, runs_per_configuration=1,
        note='Approximate derived values use rounded Nsight CSV counters; FMA counts as two FLOPs.'
    ), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
