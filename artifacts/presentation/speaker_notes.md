# Presentation speaker notes

## 1. Parallel Monte Carlo

Open with the question: can a faster estimate still be statistically unreliable? This is an applied comparison, not a new Monte Carlo algorithm. Team: Dev explains OpenMP, Rajrup explains MPI, Aryan explains CUDA and integration.

## 2. Estimate an area by counting points

Uniform points lie in the unit square. The part inside x²+y²≤1 has area pi/4, so four times the inside fraction estimates pi. More independent samples reduce typical error at the rate 1/sqrt(N); ten times lower error needs roughly 100 times more samples.

## 3. What this project contributes

Do not describe pi estimation as novel. The contribution is a reproducible four-way implementation comparison plus a deliberately broken stream control, interval coverage checks, and scoped hardware measurements. We do not implement a production financial or physics solver.

## 4. One experiment, four execution paths

The input specifies problem, N, seed and backend configuration. Each backend produces sums and squared sums. Shared statistics compute the estimate and confidence interval, then CSV analysis produces comparisons. This diagram shows data flow, not simultaneous execution of every backend.

## 5. How work is divided

Dev: OpenMP uses a static parallel loop with private RNGs and reductions. Rajrup: rank r takes chunks r, r+P and so on; all ranks enter MPI collectives, even those without work. Aryan: CUDA threads process a grid-stride loop and reduce within blocks; the host combines block totals. CPU chunks have 4096 samples.

## 6. A fast estimate can still be unreliable

A seed initializes an RNG. A stream is the sequence of generated values. CPU engines are assigned to fixed chunks, which preserves sample mapping across worker counts. CUDA uses Philox subsequences indexed by global thread ID. The negative control deliberately repeats streams. Different seeds alone do not prove independence.

## 7. Measure under stated conditions

Hardware: Intel Core Ultra 7 255HX and RTX 5060 Laptop GPU. The randomized common comparison has 48 configurations, 3 shuffled rounds of 10 trials, totaling 1440 trials and 30 distinct seeds per configuration. The separate CUDA study has 60 GPU configurations times 100 trials and 9 serial configurations times 30, totaling 6270 observations. Do not combine their speedup figures.

## 8. Ten million samples: measured comparison

Values are medians in milliseconds for separated streams at N=10^7: sequential 108.1431; OpenMP with 8 threads 27.18275; MPI with 8 ranks 32.35380; CUDA block 256 1.5904. Ratios are serial median divided by backend median. Different RNG families prevent interpreting the CUDA ratio as hardware alone.

## 9. Scaling is useful—and imperfect

CPU efficiency is speedup divided by workers: 49.7% for OpenMP and 41.8% for MPI at eight workers. Two-rank MPI was slower than serial in the rerun; do not invent a confirmed cause. Earlier fixed-order superlinear values did not reproduce. Laptop scheduling, clock and thermal effects remain possible influences that need controlled experiments.

## 10. Reliability: test intervals across seeds

A 95% confidence procedure should contain the true value in roughly 95% of independent repetitions under its assumptions; it is not a 95% probability assigned to a fixed computed interval. At N=10^9 and block 256, 100 CUDA trials yielded coverage 98/100 for separated streams and 12/100 for repeated streams. Variance/theory was 0.781 versus 162.268. Finite trial uncertainty remains; these checks do not prove RNG independence.

## 11. Error decreases with sample count

RMSE means root mean squared error across run seeds; it combines squared departures from true pi. The dashed theoretical line assumes independent points. Repeated streams can still improve with N while remaining much worse than the independent reference. The known higher-dimensional integral averages squared coordinates and equals 1/3; its variance decreases with dimension by construction, so it does not demonstrate a cure for the curse of dimensionality.

## 12. What limits the tested CUDA kernels?

Roofline compares floating-point throughput with compute and memory ceilings. Pi achieved about 54 FP64 GFLOP/s against 126.36, or 42.7%. FP64 pipeline utilization near 85–87% is a different counter and must not be called percent of theoretical FLOP peak. Low DRAM throughput and high pipeline activity support an FP64 arithmetic bottleneck for tested configurations. Philox integer work is excluded from the FLOP count. Profiling uses replay and requested base clocks. Energy averages about 303 million samples per GPU joule over three sustained windows, not whole-system energy.

## 13. How this fits existing research

References: Tramm et al., arXiv:2403.12345, portable production transport; Cuneo and Variansyah, arXiv:2403.06362, transport RNG management; Tolotti et al., arXiv:2408.09229, cuVegas adaptive integration; Salmon et al., SC11 DOI 10.1145/2063384.2063405, counter-based RNGs. Production studies have different workloads, hardware and baselines, so do not compare raw speedup ratios as rankings. Full bibliography is in the report.

## 14. Live demonstration

Start scripts/start_demo.bat beforehand and confirm all four backend badges. Use one million samples, five trials, four CPU workers, block 256 and separated streams. Run once; explain total time, kernel time and confidence intervals by hovering labels. Download CSV. A five-trial demo is not evidence of 95% coverage. If execution fails, show explicitly labelled previously measured results.

## 15. Conclusions and limits

Main finding: the GPU is fastest in the quoted tested configuration, while repeated streams severely weaken confidence-interval coverage. The result is scoped to this laptop and these implementations. Remaining work: matched RNG families, controlled CPU anomaly investigation, repeated profiles and broader hardware. No A-star acceptance, universal independence or whole-system energy claim is justified. Invite questions and use backup slides for formulas and timing boundaries.

## 16. Backup: formulas for the viva

Let p=pi/4 and H be the inside count among N independent points. H is binomial, so Var(4H/N)=(4pi-pi²)/N. For per-point values Z=4 times the inside indicator, sample variance is divided by N−1. The interval uses estimate ±1.96 sqrt(s²/N). For the integral, averaging d independent x² values yields variance 4/(45dN). Explain assumptions before quoting formulas.

## 17. Backup: timing and correctness

OpenMP includes parallel-region creation and reduction. MPI excludes startup and the pre-trial barrier; it includes sampling and moment reduction and reports the maximum rank time. CUDA total includes allocation, kernel, transfer, host accumulation and free, but excludes context startup and process launch. Kernel includes RNG initialization. CPU checks cover 1,2,4,8 workers; sample counts 2,4095,4096,4097,100000; both problems and stream modes; integral tolerance 1e-12. Sixteen CUDA sanitizer cases were recorded as passing.