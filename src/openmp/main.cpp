// Owner: Dev Visoulia. Compilable reference skeleton; parallel work is TODO.
#include "../common.hpp"
#include <omp.h>

int main(int argc, char** argv) {
    try {
        auto o = parse_options(argc, argv);
        if (!o.allow_starter) throw std::runtime_error("OpenMP starter: complete TODOs, or use --allow-starter for reference output.");
        omp_set_num_threads(o.workers);
        csv_header();
        for (int run = -o.warmups; run < o.runs; ++run) {
            auto seed = o.seed + static_cast<std::uint64_t>(std::max(0, run));
            double start = omp_get_wtime();
            // TODO(Dev): parallel-for over fixed chunk IDs. Reduce BOTH sum
            // and squares; keep the sample-to-stream mapping in cpu_chunk().
            auto m = serial_count(o, seed);
            double ms = 1000*(omp_get_wtime()-start);
            if (run >= 0) print_result(o, "openmp", "starter_serial", "mt19937_64", run+1, seed, m, ms);
        }
        // TODO(Dev): compare 1/2/4/8 threads, verify counts match baseline,
        // record actual team size, then change status to implemented.
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
