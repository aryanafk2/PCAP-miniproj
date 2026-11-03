# Parallel Monte Carlo: Pi and High-Dimensional Integration

Public repository: https://github.com/aryanafk2/PCAP-miniproj

Sequential C++ baseline, OpenMP, MPI, and CUDA implementations with a local demonstration dashboard.
Target machine: one Windows laptop with an NVIDIA RTX 5060 Laptop GPU.

## Reports and Presentation

- [IEEE report](artifacts/reports/ieee.pdf) · [Springer report](artifacts/reports/springer.pdf)
- [PowerPoint slides](artifacts/presentation/parallel_monte_carlo.pptx) · [Slide PDF](artifacts/presentation/parallel_monte_carlo.pdf)
- [Speaker notes](artifacts/presentation/speaker_notes.md) · [Recorded measurements](artifacts/data/README.md)
- [Report sources and rebuild instructions](artifacts/README.md)

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
build\openmp.exe --samples 10000 --workers 4
mpiexec -n 4 build\mpi.exe --samples 10000
```

OpenMP distributes fixed chunks across threads and reduces both sample moments.
MPI distributes chunks across local processes and reduces the moments to rank 0.
The CPU sample streams stay fixed when thread/process counts change.

`--rng separated` is the default. `--rng repeated` deliberately reuses
streams, for the negative-control experiment. CPU uses the standard-library
MT19937-64 generator; CUDA uses cuRAND Philox. These are different workloads
at the RNG level: cross-device speedup is an end-to-end implementation
comparison, not an isolated comparison of hardware arithmetic speed.

## Project Layout

- `src/common.hpp`: CLI, CPU reference sampling, CSV schema and statistics.
- `src/sequential/`: true serial baseline, without OpenMP calls.
- `src/openmp/`: parallel CPU chunk loop and OpenMP reductions.
- `src/mpi/`: distributed chunk loop and MPI reductions.
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
python scripts/experiments.py --samples 1000000 10000000 --runs 10 --rounds 3 --include-team-code
python scripts/measure_energy.py --seconds 15
```

Public Python scripts use only the standard library. Add `--include-team-code`
to experiments to include OpenMP/MPI worker sweeps. See
[initial validation](docs/initial_validation.md) for the measured results.

Configuration order is shuffled reproducibly within each round. Round seed
ranges do not overlap, so `--runs 10 --rounds 3` gives 30 distinct run seeds
per configuration. Metadata records order seed, active power plan and binary
hashes. Repeated sessions are still needed to assess laptop timing variability.

## CUDA Experiments

```text
scripts\build_cuda.bat
build\cuda.exe --describe --block 256
python scripts/test_cuda_tools.py
python scripts/validate_cuda.py
python scripts/profile_cuda.py --output results/cuda_study_PROFILE/profiling/counter_study
python scripts/run_cuda_study.py --runs 100 --baseline-runs 30
python scripts/measure_energy.py --seconds 20 --repeats 3
```

The CUDA study sweeps sample sizes, stream strategies, block sizes and
integration dimensions. Analysis separates kernel time, total time, estimator
variance and confidence-interval coverage. Energy uses complete sustained
workload windows with timestamp-aligned GPU power readings.

## Live Demonstration

Build first, then double-click `scripts/start_demo.bat` or run:

```text
python scripts/demo_server.py
```

Open **http://127.0.0.1:8765**. Choose samples, workers and a problem, then run
the comparison. The dashboard invokes all available compiled backends serially
to avoid resource contention; charts and CSV downloads use their real results.
The point-cloud illustration is explanatory, not the benchmark's actual samples.
Use 1 million samples, five trials and four workers for a short presentation.
Hover over labels or focus their help buttons for definitions and explanations.
Click or tap to pin an explanation; press Escape to dismiss it.

Median trial times exclude process startup and warm-ups. MPI trial time includes
the moment reduction but excludes the pre-trial barrier and timing-metadata
reduction. The interface also reports command duration. Confidence intervals
belong to the last trial, not to the mean of the displayed estimates. Short
interactive runs are demonstrations; use 30+ trials for report benchmarks.
MPI runs on one laptop, not a cluster. CUDA speedup includes RNG-family differences.

Run `python scripts/test_demo.py` to check the HTTP API and real comparisons.
Stop the existing server before restarting after code changes. Duplicate servers
on the same port are rejected; use `--port 8766` if a different port is needed.
