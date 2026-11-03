# Validation

## Parallel implementations: 3 November 2026

A follow-up comparison shuffled configurations within three rounds, using ten
distinct run seeds per round (30/configuration). For separated pi at N=10^7,
median serial time was 108.1431 ms; eight-thread OpenMP 27.1827 ms (3.98x),
eight-rank MPI 32.3538 ms (3.34x), CUDA block 256 1.5904 ms (68.00x, including
different RNG-family work). Earlier apparent superlinear eight-worker speedups
did not reproduce. Local MPI at two ranks was slower than serial in this run;
do not claim monotonic or universally efficient scaling. Shuffling reduces
order bias but does not replace repeated sessions with controlled power/thermals.

OpenMP and MPI now perform parallel chunk sampling and moment reductions.
The expanded verification suite passed for both problems, both RNG modes,
sample counts 2/4095/4096/4097/100000 and 1/2/4/8 threads or local ranks.
This includes ranks with no work, partial final chunks, repeatable seed mapping,
confidence-interval agreement and rejection of invalid input. Integral results
use a 1e-12 comparison tolerance to allow floating-point reduction ordering.
CUDA branch/block checks passed in the same suite.

A 72-configuration pi sweep and 24-configuration eight-dimensional integral
sweep completed with 30 trials each (2,880 recorded trials in total). These are
single-session measurements in fixed configuration order, not randomized
interleaved repetitions. Some apparent CPU efficiencies exceeded 100%; laptop
frequency changes, scheduling and baseline variability require controlled
repeats before strong scaling claims. Correctness does not depend on speedup.

The demo API passed real four-backend execution, malformed-input, cross-origin
and overlapping-request tests. Its short runs are intended for demonstrations.

## Historical initial validation: 20 October 2026

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
warm-ups. At that historical checkpoint, final four-backend and dimension-sweep measurements were still pending; subsequent studies supersede that status.

## Profiling

The initial profiling attempt returned ERR_NVGPUCTRPERM. Counter access was later enabled and profiling completed; see [profiling methodology and results](profiling.md) for the current evidence.
