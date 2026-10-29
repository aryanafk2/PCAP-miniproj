"""Report estimator uncertainty separately from runtime and kernel timing."""
import argparse
import csv
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


def wilson_interval(successes, count):
    """Wilson interval shows uncertainty in the measured coverage fraction."""
    if count < 1:
        raise ValueError('Coverage requires at least one replicate')
    z = 1.96
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return max(0, center-half), min(1, center+half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.folder.glob('*.csv'):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('status') != 'implemented':
                    continue
                key = tuple(row[k] for k in ('backend','problem','rng','generator','samples','dimensions','workers','block'))
                groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate experiment seeds within configuration: {key}')
        estimates = [float(r['estimate']) for r in rows]
        times = [float(r['total_ms']) for r in rows]
        kernels = [float(r['kernel_ms']) for r in rows if r['kernel_ms']]
        if any(not math.isfinite(x) for x in estimates+times+kernels) or min(times) <= 0:
            raise ValueError('Invalid numeric measurement')
        if kernels and any(float(r['kernel_ms']) > float(r['total_ms']) for r in rows):
            raise ValueError('Kernel time exceeds its containing total trial time')
        truth = math.pi if key[1] == 'pi' else 1/3
        variance = stats.variance(estimates) if len(rows)>1 else float('nan')
        theory = float(rows[0]['theoretical_variance'])
        successes = sum(float(r['ci95_low']) <= truth <= float(r['ci95_high']) for r in rows)
        lower, upper = wilson_interval(successes, len(rows))
        mean_time = stats.mean(times)
        time_std = stats.stdev(times) if len(times)>1 else 0
        time_half = 1.96*time_std/math.sqrt(len(times))
        output.append(dict(zip(('backend','problem','rng','generator','samples','dimensions','workers','block'), key),
                           runs=len(rows), mean_estimate=stats.mean(estimates),
                           mean_abs_error=stats.mean(abs(x-truth) for x in estimates),
                           bias=stats.mean(estimates)-truth,
                           rmse=math.sqrt(stats.mean((x-truth)**2 for x in estimates)),
                           estimator_variance=variance, theoretical_variance=theory,
                           variance_ratio=variance/theory,
                           coverage95=successes/len(rows), coverage_wilson_low=lower,
                           coverage_wilson_high=upper,
                           mean_total_ms=mean_time, median_total_ms=stats.median(times),
                           runtime_std_ms=time_std,
                           mean_time_ci95_low=max(0,mean_time-time_half),
                           mean_time_ci95_high=mean_time+time_half,
                           mean_kernel_ms=stats.mean(kernels) if kernels else '',
                           median_kernel_ms=stats.median(kernels) if kernels else '',
                           kernel_std_ms=stats.stdev(kernels) if len(kernels)>1 else '',
                           mean_host_overhead_ms=stats.mean(float(r['total_ms'])-float(r['kernel_ms']) for r in rows) if kernels else '',
                           samples_per_sec=int(key[4])/(mean_time/1000)))
    if not output:
        raise SystemExit('No implemented benchmark rows')
    baselines = {(r['problem'],r['rng'],r['samples'],r['dimensions']): r for r in output if r['backend']=='sequential'}
    for row in output:
        base = baselines.get((row['problem'],row['rng'],row['samples'],row['dimensions']))
        row['median_speedup_vs_serial'] = base['median_total_ms']/row['median_total_ms'] if base else ''
        row['cpu_efficiency'] = row['median_speedup_vs_serial']/int(row['workers']) if base and row['backend'] in ('openmp','mpi') else ''
        row['baseline_generator'] = base['generator'] if base else ''
        row['comparison_scope'] = ('implementation throughput including different RNG families'
                                   if base and base['generator'] != row['generator'] else
                                   'same RNG family' if base else 'no matched serial baseline')
    with (args.folder/'summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(args.folder/'summary.csv')


if __name__ == '__main__':
    main()
