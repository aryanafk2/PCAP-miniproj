// Component: Dev Visoulia. Each fixed sample chunk owns its random stream.
#include "../common.hpp"
#include <omp.h>

int main(int argc, char** argv) {
    try {
        auto o = parse_options(argc, argv);
        omp_set_dynamic(0); // Request a stable team; record the actual team below.
        omp_set_num_threads(o.workers);
        csv_header();
        for (int run = -o.warmups; run < o.runs; ++run) {
            auto seed = o.seed + static_cast<std::uint64_t>(std::max(0, run));
            double start = omp_get_wtime();
            double sum = 0, squares = 0;
            int actual_workers = 1;
            const auto chunks = static_cast<std::int64_t>((o.samples + chunk_size - 1)/chunk_size);
            // RNG state lives inside cpu_chunk, so threads never share engines.
            // Reductions combine private accumulators after the work is done.
            #pragma omp parallel reduction(+:sum,squares)
            {
                #pragma omp single
                actual_workers = omp_get_num_threads();
                #pragma omp for schedule(static)
                for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
                    auto part = cpu_chunk(static_cast<std::uint64_t>(chunk), o, seed);
                    sum += part.sum;
                    squares += part.squares;
                }
            }
            Moments m{sum, squares};
            double ms = 1000*(omp_get_wtime()-start);
            auto measured = o;
            measured.workers = actual_workers;
            if (run >= 0) print_result(measured, "openmp", "implemented", "mt19937_64", run+1, seed, m, ms);
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
