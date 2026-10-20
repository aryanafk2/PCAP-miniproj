#include "../common.hpp"

int main(int argc, char** argv) {
    try {
        auto o = parse_options(argc, argv);
        o.workers = 1;
        csv_header();
        for (int run = -o.warmups; run < o.runs; ++run) {
            auto seed = o.seed + static_cast<std::uint64_t>(std::max(0, run));
            auto start = std::chrono::steady_clock::now();
            auto m = serial_count(o, seed); // No OpenMP runtime in the baseline.
            auto ms = elapsed_ms(start);
            if (run >= 0) print_result(o, "sequential", "implemented", "mt19937_64", run+1, seed, m, ms);
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
