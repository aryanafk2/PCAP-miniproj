// Owner: Aryan Sheikh. RNG generation and reduction are included in timing.
#include "../common.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

inline void check(cudaError_t error, const char* call) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(call)+": "+cudaGetErrorString(error));
}
#define CUDA(call) check((call), #call)

// UTC timestamps align with the external power sampler. A steady clock is
// also recorded so the sampler can reject a wall-clock adjustment.
static long long utc_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

__global__ void sample_kernel(std::uint64_t n, std::uint64_t seed, bool repeated,
                             bool integral, int dimensions, Moments* output) {
    extern __shared__ double shared[];
    double* sums = shared;
    double* squares = shared + blockDim.x;
    auto tid = threadIdx.x;
    std::uint64_t id = static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+tid;
    std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x)*blockDim.x;
    curandStatePhilox4_32_10_t rng;
    // Separated mode assigns a cuRAND Philox subsequence to each global
    // thread. Repeated mode deliberately reuses streams across CUDA blocks.
    curand_init(seed, repeated ? tid : id, 0, &rng);
    double sum = 0, sum_squares = 0;
    for (auto i = id; i < n; i += stride) {
        double value = 0;
        if (!integral) {
            double x = curand_uniform_double(&rng), y = curand_uniform_double(&rng);
            value = x*x+y*y <= 1.0 ? 4.0 : 0.0;
        } else {
            for (int j = 0; j < dimensions; ++j) { double x = curand_uniform_double(&rng); value += x*x; }
            value /= dimensions;
        }
        sum += value; sum_squares += value*value;
    }
    sums[tid] = sum; squares[tid] = sum_squares;
    __syncthreads();
    for (unsigned step = blockDim.x/2; step; step >>= 1) {
        if (tid < step) { sums[tid] += sums[tid+step]; squares[tid] += squares[tid+step]; }
        __syncthreads();
    }
    if (tid == 0) { output[blockIdx.x].sum = sums[0]; output[blockIdx.x].squares = squares[0]; }
}

int main(int argc, char** argv) {
    try {
        // CUDA-only controls leave the CPU/MPI command interfaces unchanged.
        std::vector<char*> common_args{argv[0]};
        unsigned long long energy_ms = 0;
        bool describe = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--describe") describe = true;
            else if (arg == "--energy-ms") {
                if (++i >= argc) throw std::invalid_argument("Missing --energy-ms value");
                std::string value = argv[i];
                std::size_t used = 0;
                energy_ms = std::stoull(value, &used);
                if (used != value.size() || energy_ms < 1 || energy_ms > 600000)
                    throw std::invalid_argument("--energy-ms must be 1..600000");
            } else common_args.push_back(argv[i]);
        }
        auto o = parse_options(static_cast<int>(common_args.size()), common_args.data());
        CUDA(cudaSetDevice(0));
        CUDA(cudaFree(nullptr)); // Context startup is excluded and noted in methodology.
        int sms = 0, active = 0;
        CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, sample_kernel, o.block, 2*o.block*sizeof(double)));
        if (!active) throw std::runtime_error("Selected block size cannot launch this kernel");
        int blocks = sms*active;
        o.workers = blocks*o.block;
        if (describe) {
            cudaDeviceProp device{};
            cudaFuncAttributes kernel{};
            int runtime = 0, driver = 0;
            CUDA(cudaGetDeviceProperties(&device, 0));
            CUDA(cudaFuncGetAttributes(&kernel, sample_kernel));
            CUDA(cudaRuntimeGetVersion(&runtime)); CUDA(cudaDriverGetVersion(&driver));
            std::cout << "{\"gpu\":\"" << device.name << "\",\"compute_major\":" << device.major
                      << ",\"compute_minor\":" << device.minor << ",\"sms\":" << sms
                      << ",\"block\":" << o.block << ",\"blocks\":" << blocks
                      << ",\"launched_threads\":" << o.workers
                      << ",\"registers_per_thread\":" << kernel.numRegs
                      << ",\"local_bytes_per_thread\":" << kernel.localSizeBytes
                      << ",\"dynamic_shared_bytes_per_block\":" << 2*o.block*sizeof(double)
                      << ",\"predicted_active_blocks_per_sm\":" << active
                      << ",\"predicted_thread_occupancy\":"
                      << static_cast<double>(active*o.block)/device.maxThreadsPerMultiProcessor
                      << ",\"runtime_version\":" << runtime << ",\"driver_api_version\":" << driver << "}\n";
            return 0;
        }
        std::vector<Moments> host(blocks);
        cudaEvent_t begin, end;
        CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
        csv_header();
        long long window_start_ns = 0;
        std::chrono::steady_clock::time_point window_start;
        for (int run = -o.warmups; energy_ms || run < o.runs; ++run) {
            if (energy_ms && run == 0) {
                window_start_ns = utc_ns();
                window_start = std::chrono::steady_clock::now();
            }
            auto seed = o.seed + static_cast<std::uint64_t>(std::max(0, run));
            auto start = std::chrono::steady_clock::now();
            Moments* device = nullptr;
            CUDA(cudaMalloc(&device, blocks*sizeof(Moments)));
            CUDA(cudaEventRecord(begin));
            sample_kernel<<<blocks, o.block, 2*o.block*sizeof(double)>>>(o.samples, seed, o.rng == "repeated", o.problem == "integral", o.dimensions, device);
            CUDA(cudaGetLastError());
            CUDA(cudaEventRecord(end));
            CUDA(cudaEventSynchronize(end));
            float kernel_ms = 0;
            CUDA(cudaEventElapsedTime(&kernel_ms, begin, end));
            CUDA(cudaMemcpy(host.data(), device, blocks*sizeof(Moments), cudaMemcpyDeviceToHost));
            Moments total;
            for (auto m : host) { total.sum += m.sum; total.squares += m.squares; }
            CUDA(cudaFree(device));
            double total_ms = elapsed_ms(start);
            if (run >= 0) print_result(o, "cuda", "implemented", "curand_philox", run+1, seed, total, total_ms, kernel_ms);
            if (energy_ms && run >= 0 && elapsed_ms(window_start) >= energy_ms) {
                // Exit only after a complete trial. The whole window includes
                // allocation, transfers, host reduction and CSV output.
                auto stop_ns = utc_ns();
                double steady_ms = elapsed_ms(window_start);
                std::cerr << std::setprecision(17) << "ENERGY_WINDOW," << window_start_ns << ','
                          << stop_ns << ',' << steady_ms << ',' << run+1 << ',' << o.samples << '\n';
                break;
            }
        }
        CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
