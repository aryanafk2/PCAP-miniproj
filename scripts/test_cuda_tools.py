"""Small checks for energy boundaries, telemetry integration and CUDA modes."""
import csv
import io
import json
import subprocess
from pathlib import Path
from measure_energy import integrate, window_points
from analyze import wilson_interval

ROOT = Path(__file__).resolve().parents[1]


def main():
    points = [(0,10), (1,20), (2,30)]
    clipped = window_points(points, 0.5, 1.5)
    assert clipped == [(0.5,15), (1,20), (1.5,25)]
    assert integrate(clipped) == 20
    low, high = wilson_interval(30, 30)
    assert 0.88 < low < 0.89 and abs(high-1) < 1e-12
    for start, stop in [(-1,1), (1,3), (1,1)]:
        try:
            window_points(points, start, stop)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted an unbracketed energy window')
    for block in (128,256,512):
        description = json.loads(subprocess.check_output(
            [str(ROOT/'build/cuda.exe'),'--describe','--block',str(block)], text=True))
        assert description['block'] == block
        assert 0 < description['predicted_thread_occupancy'] <= 1
    command = [str(ROOT/'build/cuda.exe'),'--energy-ms','100','--samples','1000000']
    result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    control = [line.split(',') for line in result.stderr.splitlines() if line.startswith('ENERGY_WINDOW,')]
    assert len(control) == 1
    _, start, stop, ms, runs, samples = control[0]
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert int(stop) > int(start) and float(ms) >= 100
    assert int(runs) == len(rows) and int(samples) == 1000000
    for value in ('0','-1','600001','bad'):
        bad = subprocess.run([str(ROOT/'build/cuda.exe'),'--energy-ms',value], capture_output=True)
        assert bad.returncode != 0
    print('PASS: interpolated energy boundaries, graceful CUDA trial counts, launch metadata and invalid durations')


if __name__ == '__main__':
    main()
