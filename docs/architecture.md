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
mean of coordinate squares. OpenMP/MPI are currently reference skeletons;
their TODOs implement worker ownership and moment reduction.
