# Related Work

The studies below provide context for parallel Monte Carlo implementation,
random-stream design and performance evaluation.

## Verified Recent Sources

| Study | Approach | Benefit | Limitation / relevance |
|---|---|---|---|
| Morgan et al., MC/DC, M&C 2025 | Event-based transport, Numba JIT and GPU portability | Tests both NVIDIA and AMD | Different workload and baseline: 15x/0.7x/12x vs 112 CPU cores for C5G7; not directly comparable to pi. |
| Tramm et al., OpenMC, 2024 | OpenMP target offloading across three GPU vendors | Production portability evidence | Uses supercomputers and transport physics, while ours uses one laptop. |
| Tramm et al., AI accelerator transport, 2023 preprint / subsequent journal paper | WSE-2 transport kernel, communication/load balancing algorithms | Reports 130x vs optimized A100 kernel | Different accelerator and kernel; no evidence that pi would see the same ratio. |
| Motta, Nemmen and Joshi, GPUmonty, ApJ 2026 | CUDA radiative transfer, analytic validation | About 12x vs original CPU code; validates MC convergence | Register pressure is identified as a limit; motivates actual profiling instead of assuming compute/memory saturation. |
| Ali, Aygul and Lee, GREAPMC, 2025 | GPU history tracking, reactor-specific optimizations | Large production transport acceleration | Reactor geometry and baseline differ; retain precise hardware context before quoting equivalent CPU-core counts. |
| Suchoski et al., Frontiers 2022 | GPU ensembles for epidemiological Monte Carlo analysis | Practical simulation/forecasting application | MCMC/ensemble work differs from independent point sampling. |

## Source Links

1. https://arxiv.org/abs/2501.05440 (published M&C 2025, pp. 1934-1943, DOI 10.13182/MC25-47142)
2. https://arxiv.org/abs/2403.12345
3. https://arxiv.org/abs/2311.01739
4. https://arxiv.org/abs/2602.13198 (ApJ 1001, 148, DOI 10.3847/1538-4357/ae5631)
5. https://doi.org/10.1051/epjn/2025046
6. https://doi.org/10.3389/fams.2022.818016

## Candidates Needing Full Bibliographic/Numerical Audit

- Electron tracks simulation in water (2022), DOI 10.1016/j.ejmp.2022.10.020.
- Exploring Numba and CuPy for GPU-Accelerated Monte Carlo Radiation Transport
  (2024), DOI 10.3390/computation12030061.
- A fast GPU Monte Carlo implementation for radiative heat transfer in
  graded-index media (July 2021), DOI 10.1016/j.jqsrt.2021.107680.

## Research Gap and Comparison Rule

Our scope is the effect of stream reuse on statistical reliability and
implementation throughput across CPU/GPU models on accessible hardware.
For each final comparison record generator, samples/work unit, CPU/GPU model,
baseline parallelism, timing scope, exact table/figure and limitations.
Numerically similar speedups are not evidence of equivalent performance.
Do not claim that adding basic pi timings is a novel algorithmic contribution.
