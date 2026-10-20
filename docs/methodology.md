# Experimental Methodology

## Problem and Statistical Model

For pi, generate uniform x,y in the unit square. Let H count x^2+y^2 <= 1.
Then pi_hat=4H/N. This uses the quarter-circle, not a complete circle inside
the unit square. With p_hat=H/N, estimated SE=4*sqrt(p_hat(1-p_hat)/N).
Report the approximate CLT interval pi_hat +/- 1.96*SE. The implementation
uses the sample-variance correction N/(N-1). Exact theoretical estimator
variance is (4*pi-pi^2)/N for independent points.

The higher-dimensional target is I_d=integral_[0,1]^d (sum_j x_j^2/d) dx.
Its exact value is 1/3 and estimator variance is 4/(45*d*N). Test d=2,8,16,32.
This smooth separable target is a validation case, not evidence of overcoming
the curse of dimensionality. QMC is not yet implemented or claimed.

## RNG and Reliability

CPU uses standard-library MT19937-64 with a fixed 4096-sample chunk mapping;
chunk seed material comprises experiment seed and chunk index. Thread/rank
count cannot change pi's chunk assignment. Seed separation alone is not a
proof of independent MT streams: report this limitation and empirical tests.
CUDA uses cuRAND Philox with seed=experiment seed and subsequence=global
thread ID. The repeated-stream controls deliberately share chunk/worker
streams. Pseudorandom streams are deterministic, not truly independent draws.

Use a documented list of run seeds shared between implementations, and
different seeds between replicates. Report mean error, empirical variance,
variance/theory ratio and empirical 95% coverage. Repeated-stream intervals
assume independence that is deliberately violated; poor coverage is expected.
Passing these tests does not establish universal RNG correctness.

## Timing and Scaling

Use the same MSVC host optimization settings for serial/OpenMP/MPI, with
RNG and moment accumulation inside every timed trial. Two unreported warm-ups
precede measured trials. Run at least 30 trials per final configuration.
For CUDA export event-based kernel time and wall-clock total trial time.
Allocation, D2H transfer, final CPU accumulation and free are in total time;
context startup/event setup/process launch are excluded. Also record complete
command duration if studying startup cost. CPU-vs-CUDA RNG families differ;
label speedup as implementation-level throughput, not hardware-only speedup.

Test N=10^4 through 10^9 if feasible; CPU threads and local MPI processes
1,2,4,8; CUDA blocks 128,256,512. Compare medians and means with timing
dispersion. S=Tserial/Tparallel; E=S/P only for CPU threads or MPI processes.
CUDA thread count is not an appropriate denominator for efficiency.
Run plugged in and idle; capture hardware, compiler, driver and power state.

## GPU Profiling and Energy

Use Nsight Compute's SpeedOfLight/occupancy/instruction/memory sections.
Separate RNG arithmetic, double arithmetic, reduction and launch cost before
choosing an appropriate roofline. A floating-point roofline alone misses
integer RNG work. Never infer exact independence or bottlenecks from pi error.

Run one CUDA configuration continuously for at least 15 seconds. Record
timestamped power and integrate using the trapezoidal rule. Report GPU-only
energy, elapsed wall time, completed measured samples and samples/joule.
The window includes process setup and host work; label its scope and keep
idle power separate if subtracting it. This is not CPU/system energy.

## Sources

- Salmon et al., SC11, "Parallel Random Numbers: As Easy as 1, 2, 3",
  https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
- NVIDIA cuRAND device API,
  https://docs.nvidia.com/cuda/curand/device-api-overview.html
- Microsoft MPI local execution,
  https://learn.microsoft.com/en-us/powershell/high-performance-computing/mpiexec
