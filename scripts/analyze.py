"""Summarize estimator reliability separately from runtime dispersion."""
import argparse
import csv
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


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
                key = tuple(row[k] for k in ('backend', 'problem', 'rng', 'generator', 'samples', 'dimensions', 'workers', 'block'))
                groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        estimates = [float(r['estimate']) for r in rows]
        times = [float(r['total_ms']) for r in rows]
        truth = math.pi if key[1] == 'pi' else 1/3
        variance = stats.variance(estimates) if len(rows) > 1 else float('nan')
        theory = float(rows[0]['theoretical_variance'])
        output.append(dict(zip(('backend','problem','rng','generator','samples','dimensions','workers','block'), key),
                           runs=len(rows), mean_estimate=stats.mean(estimates),
                           estimator_variance=variance, theoretical_variance=theory,
                           variance_ratio=variance/theory,
                           coverage95=sum(float(r['ci95_low']) <= truth <= float(r['ci95_high']) for r in rows)/len(rows),
                           mean_total_ms=stats.mean(times), median_total_ms=stats.median(times),
                           runtime_std_ms=stats.stdev(times) if len(times)>1 else 0,
                           samples_per_sec=int(key[4])/(stats.mean(times)/1000)))
    if not output:
        raise SystemExit('No implemented benchmark rows')
    with (args.folder/'summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(args.folder/'summary.csv')


if __name__ == '__main__':
    main()
