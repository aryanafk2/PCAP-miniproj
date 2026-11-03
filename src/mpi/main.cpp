// Component: Rajrup Mal. MPI ranks share chunk IDs, not random engine state.
#include "../common.hpp"
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        auto o = parse_options(argc, argv);
        o.workers = size;
        if (rank == 0) csv_header();
        for (int run = -o.warmups; run < o.runs; ++run) {
            auto seed = o.seed + static_cast<std::uint64_t>(std::max(0, run));
            MPI_Barrier(MPI_COMM_WORLD);
            double start = MPI_Wtime();
            const auto chunks = (o.samples + chunk_size - 1)/chunk_size;
            double local[2] = {0, 0}, combined[2] = {0, 0};
            // Round-robin ownership covers every chunk exactly once. Ranks
            // with no chunks still participate in every collective operation.
            for (std::uint64_t chunk = rank; chunk < chunks; chunk += size) {
                auto part = cpu_chunk(chunk, o, seed);
                local[0] += part.sum;
                local[1] += part.squares;
            }
            MPI_Reduce(local, combined, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            Moments m{combined[0], combined[1]};
            // Include sampling and the moment reduction. The pre-trial barrier,
            // MPI startup and the timing-metadata reduction are excluded.
            double local_ms = 1000*(MPI_Wtime()-start), max_ms = 0;
            MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0 && run >= 0) print_result(o, "mpi", "implemented", "mt19937_64", run+1, seed, m, max_ms);
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }
}
