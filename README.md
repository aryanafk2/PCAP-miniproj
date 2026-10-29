# Parallel Monte Carlo: Pi and High-Dimensional Integration

Public repository: https://github.com/aryanafk2/PCAP-miniproj

Sequential C++ baseline, OpenMP starter, MPI starter, and CUDA implementation.
Target machine: one Windows laptop with an NVIDIA RTX 5060 Laptop GPU.

## Contributors

| Member | Component |
|---|---|
| Rajrup Mal (@fastidiousflame) | MPI |
| Dev Visoulia | OpenMP |
| Aryan Sheikh (@aryanafk2) | CUDA |

## Build and Run

From the project folder in PowerShell or Command Prompt:

```bat
scripts\build.bat
build\sequential.exe --samples 1000000 --runs 30
build\cuda.exe --samples 1000000 --runs 30 --block 256
build\cuda.exe --problem integral --dimensions 8 --samples 1000000 --runs 30
build\openmp.exe --samples 10000 --workers 4 --allow-starter
mpiexec -n 4 build\mpi.exe --samples 10000 --allow-starter
```

OpenMP and MPI intentionally require `--allow-starter`: their current output
is a serial/reference result with `starter_*` status. It is excluded from
research speedup claims. See the TODOs in their source files.

`--rng separated` is the default. `--rng repeated` deliberately reuses
streams, for the negative-control experiment. CPU uses the standard-library
MT19937-64 generator; CUDA uses cuRAND Philox. These are different workloads
at the RNG level: cross-device speedup is an end-to-end implementation
comparison, not an isolated comparison of hardware arithmetic speed.

## Project Layout

- `src/common.hpp`: CLI, CPU reference sampling, CSV schema and statistics.
- `src/sequential/`: true serial baseline, without OpenMP calls.
- `src/openmp/`: Dev's compilable starter.
- `src/mpi/`: Rajrup's compilable starter.
- `src/cuda/`: Aryan's checked CUDA implementation and dual timing.
- `scripts/`: build, verification, experiments and analysis.
- `docs/`: architecture, methodology, validation and profiling notes.
- `results/`: benchmark CSVs and statistical summaries.

Verified commands after building:

```text
python scripts/verify.py
python scripts/experiments.py --samples 10000 1000000 --runs 30
python scripts/experiments.py --samples 100000 --runs 30 --problem integral --dimensions 8
python scripts/analyze.py results/EXPERIMENT_FOLDER
python scripts/measure_energy.py --seconds 15
```

Python scripts use only the standard library. Add `--include-team-code` to
experiments only after MPI/OpenMP are implemented. See
[initial validation](docs/initial_validation.md) for the measured results.

## CUDA Experiments

```text
scripts\build_cuda.bat
build\cuda.exe --describe --block 256
python scripts/test_cuda_tools.py
python scripts/run_cuda_study.py --runs 100 --baseline-runs 30
python scripts/measure_energy.py --seconds 20 --repeats 3
```

The CUDA study sweeps sample sizes, stream strategies, block sizes and
integration dimensions. Analysis separates kernel time, total time, estimator
variance and confidence-interval coverage. Energy uses complete sustained
workload windows with timestamp-aligned GPU power readings.
