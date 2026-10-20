// Owner: Rajrup Mal. MPI lifecycle is ready; distributed counting is TODO.
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
        if (!o.allow_starter) throw std::runtime_error("MPI starter: complete TODOs, or use --allow-starter for rank-0 reference output.");
        if (rank == 0) csv_header();
        for (int run = -o.warmups; run < o.runs; ++run) {
            auto seed = o.seed + static_cast<std::uint64_t>(std::max(0, run));
            MPI_Barrier(MPI_COMM_WORLD);
            double start = MPI_Wtime();
            Moments m;
            // TODO(Rajrup): assign chunks rank, rank+size, ... to each rank;
            // use MPI_Reduce(MPI_SUM) for sum and squares. Handle remainders
            // through cpu_chunk() and keep rank out of the chunk seed.
            if (rank == 0) m = serial_count(o, seed);
            double local_ms = 1000*(MPI_Wtime()-start), max_ms = 0;
            MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0 && run >= 0) print_result(o, "mpi", "starter_rank0", "mt19937_64", run+1, seed, m, max_ms);
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }
}
