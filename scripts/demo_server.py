"""Local presentation UI backed by the real compiled Monte Carlo programs.

Run: python scripts/demo_server.py, then open http://127.0.0.1:8765.
Only bounded, validated numerical options reach fixed executable paths.
"""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
RUN_LOCK = threading.Lock()


def mpi_launcher():
    found = shutil.which('mpiexec')
    fallback = Path(r'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe')
    return found or (str(fallback) if fallback.is_file() else None)


def availability():
    return {name: (ROOT/'build'/f'{name}.exe').is_file() and
            (name != 'mpi' or mpi_launcher() is not None)
            for name in ('sequential', 'openmp', 'mpi', 'cuda')}


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a JSON object')
    allowed = dict(samples=(10000, 100000, 1000000, 10000000), runs=(3, 5, 10),
                   workers=(1, 2, 4, 8), block=(128, 256, 512),
                   dimensions=(2, 8, 16, 32), problem=('pi', 'integral'),
                   rng=('separated', 'repeated'))
    defaults = dict(samples=1000000, runs=5, workers=4, block=256,
                    dimensions=8, problem='pi', rng='separated')
    if set(payload) - set(defaults):
        raise ValueError('Unknown experiment option')
    settings = defaults | payload
    for key, value in settings.items():
        valid_value = 10000 <= value <= 10000000 if key == 'samples' and type(value) is int else value in allowed[key]
        if type(value) is not type(defaults[key]) or not valid_value:
            raise ValueError(f'Unsupported {key}')
    return settings


def execute(command):
    # No shell: input cannot become executable command text. If a run hangs,
    # terminate this launched process tree, including its local MPI children.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, cwd=ROOT)
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
        else:
            process.kill()
        process.communicate()
        raise RuntimeError('Experiment exceeded 120 seconds; try fewer samples')
    if process.returncode:
        raise RuntimeError(stderr.strip()[:800] or 'Backend execution failed')
    return stdout


def experiment(settings):
    results = []
    ready = availability()
    if not ready['sequential']:
        raise RuntimeError('Build the executables with scripts/build.bat first')
    # Running backends one at a time avoids competing for laptop resources.
    for backend, available in ready.items():
        if not available:
            continue
        command = [str(ROOT/'build'/f'{backend}.exe')]
        if backend == 'mpi':
            command = [mpi_launcher(), '-n', str(settings['workers']), *command]
        for key in ('samples', 'runs', 'workers', 'block', 'dimensions', 'problem', 'rng'):
            command += ['--' + key, str(settings[key])]
        command += ['--warmups', '2', '--seed', '12345']
        start = time.perf_counter()
        rows = list(csv.DictReader(io.StringIO(execute(command))))
        command_seconds = time.perf_counter() - start
        if len(rows) != settings['runs'] or any(r['status'] != 'implemented' for r in rows):
            raise RuntimeError('Backend returned incomplete results')
        median = statistics.median(float(r['total_ms']) for r in rows)
        baseline = results[0]['median_ms'] if results else median
        last = rows[-1]
        times = [float(r['total_ms']) for r in rows]
        estimates = [float(r['estimate']) for r in rows]
        truth = math.pi if settings['problem'] == 'pi' else 1/3
        kernels = [float(r['kernel_ms']) for r in rows if r['kernel_ms']]
        results.append(dict(backend=backend, rows=rows, median_ms=median,
                            samples_per_sec=settings['samples']/(median/1000),
                            kernel_median_ms=statistics.median(kernels) if kernels else None,
                            runtime_std_ms=statistics.stdev(times),
                            estimator_std=statistics.stdev(estimates),
                            coverage_count=sum(float(r['ci95_low']) <= truth <= float(r['ci95_high']) for r in rows),
                            speedup=baseline/median,
                            efficiency=baseline/median/int(last['workers'])
                            if backend in ('openmp', 'mpi') else None,
                            estimate=float(last['estimate']), ci_low=float(last['ci95_low']),
                            ci_high=float(last['ci95_high']), command_seconds=command_seconds))
    return dict(results=results, settings=settings)


class DemoServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can let a second server bind the same port while
    # requests still reach an old process. Fail clearly instead of mixing APIs.
    allow_reuse_address = False

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def send(self, code, data, kind='application/json'):
        body = json.dumps(data).encode() if kind == 'application/json' else data
        self.send_response(code)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/status':
            self.send(200, dict(backends=availability(), cpu_threads=os.cpu_count()))
        elif self.path in ('/', '/index.html'):
            self.send(200, (ROOT/'demo/index.html').read_bytes(), 'text/html')
        else:
            self.send(404, dict(error='Not found'))

    def do_POST(self):
        # Cross-origin pages cannot trigger local CPU/GPU workloads.
        origin = self.headers.get('Origin')
        expected = f'http://{self.headers.get("Host")}'
        if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',
                                            f'localhost:{self.server.server_port}') or (origin and origin != expected):
            self.send(403, dict(error='Local same-origin requests only'))
            return
        if self.path != '/api/run':
            self.send(404, dict(error='Not found'))
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 2 or length > 4096:
                raise ValueError('Invalid request size')
            settings = validate(json.loads(self.rfile.read(length)))
        except (ValueError, TypeError) as error:
            self.send(400, dict(error=str(error)))
            return
        if not RUN_LOCK.acquire(blocking=False):
            self.send(409, dict(error='An experiment is already running'))
            return
        try:
            self.send(200, experiment(settings))
        except (RuntimeError, OSError) as error:
            self.send(500, dict(error=str(error)))
        finally:
            RUN_LOCK.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        server = DemoServer(('127.0.0.1', args.port), Handler)
    except OSError as error:
        raise SystemExit(f'Cannot start demo on port {args.port}: close the existing server or choose --port. {error}')
    print(f'Monte Carlo Lab: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
