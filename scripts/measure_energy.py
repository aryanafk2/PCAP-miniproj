"""GPU-only sustained-window estimate with actual timestamps and integration."""
import argparse
import csv
import io
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def integrate(points):
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points, points[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--samples', type=int, default=1000000000)
    parser.add_argument('--block', type=int, default=256)
    args = parser.parse_args()
    if args.seconds < 15:
        parser.error('Use at least 15 seconds for coarse GPU polling')
    # One sustained process reduces launch gaps. Its elapsed window still
    # includes warm-up, allocation and host work; this is not kernel energy.
    command = [str(ROOT/'build/cuda.exe'), '--samples', str(args.samples), '--runs', '1000000',
               '--block', str(args.block)]
    out = ROOT/'results' / ('energy_'+time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True)
    points = []
    with (out/'trials.csv').open('w', encoding='utf-8') as trials:
        start = time.perf_counter()
        process = subprocess.Popen(command, cwd=ROOT, stdout=trials, stderr=subprocess.PIPE, text=True)
        try:
            while time.perf_counter()-start < args.seconds:
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read())
                before = time.perf_counter()
                power = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.draw','--format=csv,noheader,nounits'], text=True)
                after = time.perf_counter()
                points.append(((before+after)/2-start, float(power.strip())))
                time.sleep(0.1)
        finally:
            process.terminate(); process.wait(timeout=15)
            process.stderr.close()
    with (out/'power.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(['elapsed_s','gpu_power_w']); writer.writerows(points)
    if len(points)<2:
        raise RuntimeError('Insufficient numeric GPU power telemetry')
    # Count only complete flushed CSV rows. Do not assign whole-run work to a
    # narrower sensor window or manufacture per-kernel energy from average power.
    complete = []
    with (out/'trials.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                if float(row['total_ms']) > 0 and row.get('status') == 'implemented': complete.append(row)
            except (KeyError, TypeError, ValueError): pass
    result = {'scope':'GPU-only power, trapezoidal integration over sensor timestamps',
              'command':command, 'sensor_window_s':points[-1][0]-points[0][0],
              'gpu_energy_j':integrate(points), 'power_samples':len(points),
              'completed_flushed_trials_in_process_window':len(complete),
              'samples_per_joule':None,
              'limitation':'Work and sensor boundaries differ; sustained-window energy only. Per-kernel efficiency withheld.'}
    (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2)); print(out)


if __name__ == '__main__':
    assert integrate([(0,10),(2,20)]) == 30
    main()
