# System Architecture

```mermaid
flowchart TD
    A[CLI: N, seed, problem, stream strategy] --> B[Sequential reference]
    A --> C[OpenMP chunk loop - Dev]
    A --> D[MPI chunk partition - Rajrup]
    A --> E[CUDA grid-stride sampling - Aryan]
    B --> F[Sample moments: sum and squared sum]
    C --> F
    D --> F
    E --> F
    F --> G[Estimate, confidence interval, total/kernel time CSV]
    G --> H[Estimator variance, coverage, runtime dispersion]
    H --> I[Shared IEEE/Springer report body]
    J[Timestamped GPU power] --> K[GPU-only energy integration]
    K --> I
```

Pi reference: quarter-circle indicator times four. Integration reference:
mean of coordinate squares. OpenMP uses a static parallel chunk loop and
reductions of both moments. MPI ranks own chunks rank, rank+size, ... and
combine both moments using MPI_SUM. Rank 0 reports maximum rank elapsed time.

The demonstration dashboard is served on localhost by scripts/demo_server.py.
It validates bounded inputs, invokes fixed executable paths without a shell,
and converts their CSV output to chart data. A single-run lock prevents
simultaneous dashboard experiments. No external frontend dependencies are needed.
