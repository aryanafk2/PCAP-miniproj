# Related work: audited comparison

Eight recent papers (2024–2026) were checked against the linked full-text
versions on 3 November 2026. Locations below identify the supporting evidence.
These are contextual comparisons, not a ranking across unlike benchmarks.

| Study and checked version | Reported evidence and location | Comparison with this project |
|---|---|---|
| [Morgan et al., MC/DC GPU portability (2025)](https://arxiv.org/html/2501.05440v2) | Table 1: C5G7 5080 s on 112 Sapphire Rapids cores versus 342.6 s on **four** V100s; ratio 14.83×. | Our 68.00× compares one laptop GPU against one serial CPU path, with different RNG families. Neither denominator nor workload matches. |
| [Tramm et al., portable OpenMC (2024)](https://arxiv.org/html/2403.12345v1) | Section 4.4 / Fig. 3: weak-scaling efficiency 97%, 96%, 99% on Aurora, Polaris, Frontier. | Our OpenMP/MPI efficiencies are 49.7%/41.8% at eight local workers under strong scaling. Weak and strong scaling are distinct. |
| [Cuneo et al., MC/DC scheduling (2025)](https://arxiv.org/html/2506.00263v1) | Section 5.1 / Fig. 5: both methods approach 6.9×10^5 histories/s on four V100s; asynchronous scheduling improves early scaling. | A transport history performs variable physics work; our uniform point samples are not equivalent work units. |
| [Cuneo and Variansyah, parallel RNG (2024)](https://arxiv.org/html/2403.06362v1) | Section 5 / Figs. 1–3: 1000 batches of 1000 source particles; stride/hash times 113.26/111.46 s. Normality-test failures: 369/600 and 339/600. | Similar normality behaviour does not prove independence. Our separate/repeated-stream experiment evaluates variance and interval coverage instead. |
| [Motta et al., GPUmonty (2026)](https://arxiv.org/html/2602.13198v2) | Section V.1: laptop RTX 3050 takes 332 s versus Ryzen 5 6600H 2692 s at 10^6 superphotons, about 8.11×. Table 1 / V.2 analyze kernel bottlenecks. | This is a physics solver with different work and a register-related limit; our simple kernel exhibits an FP64 arithmetic limit in tested profiles. |
| [Hasan et al., PUMI-Tally (2025)](https://arxiv.org/html/2504.19048v1) | Section 6 / Fig. 4: CPU/GPU PUMI-Tally difference at most roughly 2×; much improvement over current OpenMC is attributed to batching/cache-aware algorithms. Fig. 9 separately accounts for CPU and GPU energy. | Our GPU-only energy cannot establish whole-system savings. A better algorithm can matter as much as GPU execution. |
| [Godoy et al., GPU energy (2025)](https://arxiv.org/html/2505.05623v3) | Section 4.1 / Table 3: 6–25% mixed-precision QMCPACK energy savings; energy throughput normalizes walkers × blocks × steps by DMC energy. | Our ~303 million samples/GPU-joule uses a different work unit and has no mixed-precision comparator. |
| [Tolotti et al., cuVegas (2024)](https://arxiv.org/html/2408.09229v1) | Tables 4–5 / 7: A100 platform; 278.4× versus default Vegas, 21.6× versus VegasFlow, 6.3× versus TorchQuad. Section 4.3 explains error matching and early stopping. | Adaptive importance/stratified integration differs from our uniform sampling. Our 68.00× is not evidence of outperforming or underperforming cuVegas. |

## Bibliographic scope

The report cites these explicit arXiv versions. Morgan and Cuneo scheduling
papers also have M&C 2025 records (DOIs 10.13182/MC25-47142 and
10.13182/MC25-47174); GPUmonty has ApJ 1001, 148 (2026), DOI
10.3847/1538-4357/ae5631. Godoy has an ISC 2025 workshop chapter, DOI
10.1007/978-3-032-07612-0_14. These publication details are listed on the
primary arXiv records; no publication status is inferred for the other entries.

## Contribution and limits

Our contribution is a four-model implementation comparison with stream-reuse
negative controls, statistical coverage, and scoped hardware evidence. It is
not a new estimator, RNG, QMC method, or transport solver. The laptop experiment
has 30 seeds/configuration; the separate GPU reliability study has 100 trials.
We do not pool these studies or treat their timing ratios as universal.
