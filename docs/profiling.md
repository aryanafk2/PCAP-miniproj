# GPU Profiling

Installed CLI:
`C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe`

Tested command:
```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe' --set basic --launch-count 1 build\cuda.exe --samples 1000000 --runs 1 --warmups 0
```

Result: ERR_NVGPUCTRPERM. Driver settings deny GPU counter access.
No measured roofline/bottleneck classification is claimed. NVIDIA's official
instructions: https://developer.nvidia.com/ERR_NVGPUCTRPERM

After access is enabled, collect occupancy, instruction mix, FP64 throughput,
DRAM traffic, launch configuration and kernel duration. Philox integer work
is not represented by a floating-point-only roofline.

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
