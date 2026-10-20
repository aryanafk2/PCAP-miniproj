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
