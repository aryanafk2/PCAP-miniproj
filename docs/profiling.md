# GPU Profiling

Installed CLI:
`C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe`

Reproduce the counter study after building and enabling counter access:
```text
python scripts/profile_cuda.py --output results/cuda_study_PROFILE/profiling/counter_study
```

Counter collection succeeded with Nsight Compute 2026.2.1. Each configuration
uses separated Philox streams, 256 threads/block and one profiled launch after
three warm-ups. Full profiling uses kernel replay and requests base clocks.

| Workload | FP64 GFLOP/s | Clock-adjusted FP64 peak GFLOP/s | Peak reached | Achieved occupancy | FP64 pipeline utilization |
|---|---:|---:|---:|---:|---:|
| Pi, N=10^8 | 53.99 | 126.36 | 42.7% | 72.0% | 85.4% |
| Pi, N=10^9 | 53.98 | 126.36 | 42.7% | 72.0% | 85.5% |
| Integral, N=10^6, d=8 | 78.06 | 126.28 | 61.8% | 74.2% | 86.8% |

These configurations are primarily FP64-arithmetic limited: Nsight identifies
FP64 as the highest-utilized pipeline, while DRAM throughput is at most 0.01%
of peak in its rounded summary. Pipeline utilization differs from FLOP peak
utilization because instruction throughput and the mix of fused/non-fused
operations differ. This is a measured diagnosis for these configurations,
not a claim about every Monte Carlo workload or every GPU.

Roofline derivation follows the installed `SpeedOfLight_RooflineChart.section`:
- Achieved FLOP/s = (DADD/cycle + DMUL/cycle + 2*DFMA/cycle) * SM clock.
- FP64 ceiling = 2 * peak DFMA thread instructions/cycle * SM clock.
- DRAM ceiling = peak DRAM bytes/cycle * DRAM clock.
- Arithmetic intensity = achieved FLOP/s / measured DRAM bytes/s.

The measured intensities are approximately 2,785, 15,520 and 3,836 FLOP/byte,
respectively. Points are far on the arithmetic side of the DRAM roofline.
Random values are generated inside the kernel rather than read from DRAM;
only block results are written out. The tiny DRAM denominator is sensitive to
cache state and background traffic, so these intensities are not portable
algorithm constants. Integer Philox work is not represented by FP64 FLOPs.
Derived numbers are approximate because the CSV export rounds counters.

The clock-adjusted theoretical ceiling is appropriate for the profile; it is
not the laptop's advertised boost-clock peak. Profiled durations and application
CUDA-event times under replay must not replace the original unprofiled timings.
One diagnostic launch per configuration is sufficient for this structural
comparison, but does not provide repeated-profile confidence intervals.
See [NVIDIA's profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline).

## CUDA Timeline

Nsight Systems 2026.1.3 captured a CUDA-only trace with CPU sampling and
context-switch collection disabled. At N=10^8 and 256 threads/block, 13
launches (3 warm-ups plus 10 diagnostic trials) averaged 7.1496 ms/kernel;
the result D2H copy averaged 1.7083 microseconds. This demonstrates that
the small result transfer is not the dominant measured operation. It does
not measure the kernel's internal DRAM traffic or establish a roofline limit.
Profiler timing is diagnostic and is separate from the unprofiled benchmark.
Host synchronization time largely overlaps GPU work and must not be summed
with kernel duration as if both were independent execution costs.

## Memory and Synchronization Validation

Run `python scripts/validate_cuda.py` after building. Compute Sanitizer runs
memcheck, racecheck, initcheck and synccheck against four configurations each:
tiny sample counts, non-divisible sample counts, both problem branches, both
stream modes, dimensions 1 and 64, and blocks from 32 to 1024 threads.
All 16 cases passed on the RTX 5060 Laptop GPU. A separate check confirmed
that changing warm-up count preserves the recorded seeds and estimates.
These are correctness diagnostics, not performance measurements or proof of
RNG independence. The script fails if a tool reports errors or does not emit
a clean diagnostic summary. Local logs include the tested binary's SHA-256.
