"""Integrate GPU power over a gracefully completed, counted CUDA window."""
import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def integrate(points):
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a, b in zip(points, points[1:]))


def window_points(points, start, stop):
    """Interpolate boundaries; never extrapolate missing power readings."""
    if not start < stop or len(points) < 2 or points[0][0] > start or points[-1][0] < stop:
        raise ValueError('Power samples must bracket the entire CUDA window')
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError('Non-monotonic telemetry timestamps')

    def at(t):
        for a, b in zip(points, points[1:]):
            if a[0] <= t <= b[0]:
                return a[1] + (b[1]-a[1])*(t-a[0])/(b[0]-a[0])
        raise ValueError('Missing boundary sample')
    return [(start, at(start)), *[p for p in points if start < p[0] < stop], (stop, at(stop))]


def read_power():
    before = time.time_ns()
    power = subprocess.check_output(
        ['nvidia-smi', '-i', '0', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    after = time.time_ns()
    watts = float(power.strip())
    if not math.isfinite(watts) or watts <= 0:
        raise ValueError('Invalid GPU power reading')
    return ((before+after)//2, watts, after-before)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--samples', type=int, default=1000000000)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 15 <= args.seconds <= 600 or args.repeats < 1:
        parser.error('Use a 15..600 second window and at least one repeat')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT/'results'/('cuda_energy_'+stamp)
    out.mkdir(parents=True)
    summaries = []
    for repeat in range(1, args.repeats+1):
        command = [str(ROOT/'build/cuda.exe'), '--samples', str(args.samples),
                   '--block', str(args.block), '--energy-ms', str(round(args.seconds*1000)),
                   '--seed', str(100000+repeat*100000)]
        points = [read_power()]
        trials_path = out/f'trials_{repeat}.csv'
        stderr_path = out/f'control_{repeat}.txt'
        # The child exits after a complete trial; stdout closes without truncation.
        with trials_path.open('w', encoding='utf-8') as trials, stderr_path.open('w', encoding='utf-8') as err:
            process = subprocess.Popen(command, cwd=ROOT, stdout=trials, stderr=err)
            deadline = time.monotonic()+args.seconds+120
            try:
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError('CUDA energy process exceeded its time budget')
                    points.append(read_power())
                    time.sleep(0.1)
                if process.returncode:
                    raise RuntimeError(f'CUDA failed; inspect {stderr_path}')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)
        points.append(read_power())
        controls = [line.split(',') for line in stderr_path.read_text().splitlines()
                    if line.startswith('ENERGY_WINDOW,')]
        if len(controls) != 1:
            raise RuntimeError('Missing or ambiguous CUDA energy window')
        _, start, stop, steady_ms, runs, samples = controls[0]
        start, stop, runs, samples = map(int, (start, stop, runs, samples))
        duration = (stop-start)/1e9
        if abs(duration-float(steady_ms)/1000) > 0.01:
            raise RuntimeError('Wall clock changed during measurement')
        with trials_path.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != runs or any(int(r['samples']) != samples for r in rows):
            raise RuntimeError('Completed work does not match CSV trial count')
        # Relative seconds reduce rounding loss in the trapezoidal calculation.
        relative = [((p[0]-start)/1e9, p[1]) for p in points]
        clipped = window_points(relative, 0, duration)
        joules = integrate(clipped)
        result = dict(repeat=repeat, samples=samples, block=args.block, runs=runs,
                      window_seconds=duration, completed_samples=runs*samples,
                      gpu_energy_j=joules, avg_gpu_power_w=joules/duration,
                      samples_per_joule=runs*samples/joules, power_readings=len(clipped),
                      max_query_ms=max(p[2] for p in points)/1e6,
                      scope='GPU-only, sustained workload including host gaps and CSV output',
                      limitation='Coarse power polling and boundary interpolation; no per-kernel energy or CPU/system comparison')
        summaries.append(result)
        with (out/f'power_{repeat}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['utc_ns','gpu_power_w','query_duration_ns'])
            writer.writerows(points)
        (out/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        with (out/'energy_summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"repeat {repeat}: {joules:.2f} J, {result['samples_per_joule']:.0f} samples/J", flush=True)
    print(out)


if __name__ == '__main__':
    main()
