# Initial Validation: 20 October 2026

All four targets compiled with MSVC/CUDA. The checks in `scripts/verify.py`
passed serial agreement for OpenMP/MPI starters, chunk remainders, CUDA
block sizes/RNG modes/problems, run-seed recording, and invalid input rejection.
These checks establish starter/reference behavior; they do not claim completed
OpenMP or MPI parallelization.

## New Pi Experiment

Folder: `results/20261020T173153Z`. 16 configurations, 30 measured runs each,
two warm-ups each; serial plus CUDA blocks 128/256/512; N=10^4 and 10^6;
separated and repeated streams. Output includes all seed/timing boundaries.

At N=10^6, CUDA 256 threads/block:

| Stream strategy | Empirical variance / theory | CI coverage | Mean total time ms |
|---|---:|---:|---:|
| Separated Philox subsequences | 0.8869 | 29/30 | 0.5306 |
| Reused streams across blocks | 115.3899 | 3/30 | 0.3709 |

This negative control shows variance inflation and undercoverage. An
apparently faster repeated-stream configuration is not an equivalent
statistical estimator. Small timing trials are noisy; these are validation
data rather than a final laptop throughput ranking. Thirty trials leave
substantial uncertainty in variance and coverage estimates.

## Sustained GPU Power

Folder: `results/energy_20261020T230226`. One sustained CUDA process with
N=10^9, 256 threads/block, default separated Philox streams. Recorded 101
power samples across a 14.9093-second sensor window; trapezoidal GPU-only
energy was 653.5477 J. The process yielded 192 complete flushed trial rows.
Sensor/work boundaries differ, so samples/J and per-kernel energy are withheld.
The process was terminated deliberately after the requested sustained window.

## Eight-Dimensional Integral

Folder: `results/20261020T173807Z`. Eight configurations with 30 runs each,
N=100000 and d=8. The exact answer is 1/3. Separated CPU streams gave mean
0.33335629, variance/theory ratio 1.0942 and coverage 28/30. Separated CUDA
streams gave mean 0.33341175, ratio 0.6815 and coverage 30/30 for the tested
block sizes. Matching CUDA streams across these block sizes are not
independent replicates; do not pool them as 90 independent runs.

Both experiment folders together contain 720 measured trial rows, excluding
warm-ups. Final four-backend and dimension-sweep measurements remain pending.

## Profiling

Nsight Compute 2026.2.1 launches the binary but returns ERR_NVGPUCTRPERM.
Counter-backed bottleneck analysis remains blocked by driver permissions.
