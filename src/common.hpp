#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

struct Options {
    std::uint64_t samples = 1000000;
    std::uint64_t seed = 12345;
    int runs = 5;
    int warmups = 2;
    int workers = 4;
    int block = 256;
    int dimensions = 8;
    std::string problem = "pi";
    std::string rng = "separated";
    bool allow_starter = false;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--allow-starter") { o.allow_starter = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value: " + key);
        std::string value = argv[++i];
        std::size_t consumed = 0;
        if (key == "--problem") o.problem = value;
        else if (key == "--rng") o.rng = value;
        else {
            if (value.empty() || value[0] == '-') throw std::invalid_argument("Expected nonnegative integer: " + key);
            auto n = std::stoull(value, &consumed);
            if (consumed != value.size()) throw std::invalid_argument("Invalid integer: " + key);
            if (key == "--samples") o.samples = n;
            else if (key == "--seed") o.seed = n;
            else {
                if (n > 1000000) throw std::invalid_argument("Integer option too large: " + key);
                if (key == "--runs") o.runs = static_cast<int>(n);
                else if (key == "--warmups") o.warmups = static_cast<int>(n);
                else if (key == "--workers") o.workers = static_cast<int>(n);
                else if (key == "--block") o.block = static_cast<int>(n);
                else if (key == "--dimensions") o.dimensions = static_cast<int>(n);
                else throw std::invalid_argument("Unknown option: " + key);
            }
        }
    }
    if (o.samples < 2 || o.samples > 1000000000000ULL || o.runs < 1 || o.workers < 1 ||
        o.dimensions < 1 || o.dimensions > 64 || o.block < 32 || o.block > 1024 ||
        (o.block & (o.block - 1)) || (o.problem != "pi" && o.problem != "integral") ||
        (o.rng != "separated" && o.rng != "repeated"))
        throw std::invalid_argument("Invalid samples/runs/workers/dimensions/block/problem/rng");
    return o;
}

struct Moments {
    double sum = 0;
    double squares = 0;
};

// A fixed chunk owns a repeatable CPU stream regardless of scheduling. Seed
// separation is a practical mapping, NOT a mathematical independence proof.
constexpr std::uint64_t chunk_size = 4096;
inline Moments cpu_chunk(std::uint64_t chunk, const Options& o, std::uint64_t seed) {
    auto stream = o.rng == "repeated" ? 0ULL : chunk;
    std::seed_seq keys{static_cast<unsigned>(seed), static_cast<unsigned>(seed >> 32),
                       static_cast<unsigned>(stream), static_cast<unsigned>(stream >> 32)};
    std::mt19937_64 engine(keys);
    auto uniform = [&]() { return (engine() >> 11) * 0x1.0p-53; };
    auto begin = chunk * chunk_size;
    auto end = std::min(o.samples, begin + chunk_size);
    Moments m;
    for (auto i = begin; i < end; ++i) {
        double value = 0;
        if (o.problem == "pi") {
            double x = uniform(), y = uniform();
            value = x*x + y*y <= 1.0 ? 4.0 : 0.0;
        } else {
            // Integral of mean(x_j^2) over [0,1]^d is exactly 1/3.
            for (int j = 0; j < o.dimensions; ++j) { double x = uniform(); value += x*x; }
            value /= o.dimensions;
        }
        m.sum += value;
        m.squares += value * value;
    }
    return m;
}

inline Moments serial_count(const Options& o, std::uint64_t seed) {
    Moments total;
    for (std::uint64_t chunk = 0; chunk < (o.samples + chunk_size - 1)/chunk_size; ++chunk) {
        auto m = cpu_chunk(chunk, o, seed);
        total.sum += m.sum; total.squares += m.squares;
    }
    return total;
}

inline void csv_header() {
    std::cout << "backend,status,problem,rng,generator,samples,dimensions,workers,block,run,seed,estimate,abs_error,ci95_low,ci95_high,theoretical_variance,total_ms,kernel_ms,samples_per_sec\n";
}

inline void print_result(const Options& o, const std::string& backend, const std::string& status,
                         const std::string& generator, int run, std::uint64_t seed,
                         Moments m, double total_ms, double kernel_ms = -1) {
    const double pi = std::acos(-1.0);
    const double truth = o.problem == "pi" ? pi : 1.0/3.0;
    const double estimate = m.sum / o.samples;
    const double variance = std::max(0.0, (m.squares - m.sum*m.sum/o.samples)/(o.samples - 1));
    const double half = 1.96 * std::sqrt(variance/o.samples);
    const double theory = (o.problem == "pi" ? 4*pi - pi*pi : 4.0/(45*o.dimensions))/o.samples;
    std::cout << std::setprecision(17) << backend << ',' << status << ',' << o.problem << ',' << o.rng
              << ',' << generator << ',' << o.samples << ',' << (o.problem == "pi" ? 2 : o.dimensions)
              << ',' << o.workers << ',' << (backend == "cuda" ? o.block : 0) << ',' << run << ',' << seed
              << ',' << estimate << ',' << std::abs(estimate-truth) << ',' << estimate-half << ',' << estimate+half
              << ',' << theory << ',' << total_ms << ',';
    if (kernel_ms >= 0) std::cout << kernel_ms;
    std::cout << ',' << o.samples/(total_ms/1000) << '\n';
}

inline double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
}
