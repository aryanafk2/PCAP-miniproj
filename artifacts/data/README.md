# Recorded measurements

| Folder | Scope |
|---|---|
| common_comparison | 48 configurations × three shuffled rounds × ten trials = 1440 observations; 30 distinct seeds per configuration; pi N=10^6 and 10^7. |
| cuda_study | Separate study: 60 CUDA configurations × 100 trials plus nine serial configurations × 30 trials = 6270 observations. |
| gpu_energy | Three sustained GPU-only workload windows with aligned power telemetry and completed sample counts. |
| profiling | Hardware counter evidence and FP64 roofline summaries; profiling clocks/replay differ from ordinary timing. |

The common comparison was recorded on 3 November 2026. The separate CUDA and
energy studies were recorded on 28–29 October 2026. Do not pool studies, block
sizes that share draws, or profiling timings as independent benchmark trials.

CPU: Intel Core Ultra 7 255HX (20 logical processors). GPU: RTX 5060 Laptop GPU.
Software: Windows, MSVC 2026, CUDA 13.3, NVIDIA driver 610.62, Microsoft MPI.
Serial/OpenMP/MPI use MT19937-64; CUDA uses cuRAND Philox. Cross-device speedups
include this RNG-family difference. MPI uses processes on one laptop.

For the common and CUDA study folders, rerun `python scripts/analyze.py FOLDER`
from the repository root. This reads measured CSVs and regenerates summary.csv.
The demo screenshots show a separate short run, not the study's median timings.
