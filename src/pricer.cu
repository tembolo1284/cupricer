// src/pricer.cu
#include "qmc/detail/mc_launch.hpp"
#include "qmc/cuda_check.hpp"

#include <cmath>     // std::exp, std::sqrt (host reduction)
#include <cstddef>
#include <vector>

#include <curand_kernel.h>   // device-side Philox RNG

namespace qmc {

// ── Kernel ───────────────────────────────────────────────────────────────────
// One thread walks a grid-stride slice of the paths, accumulating the payoff sum
// and the sum of squared payoffs in registers. A shared-memory tree reduction
// collapses each block to a single (sum, sum_sq) pair, written to global memory.
// The final cross-block reduction happens on the host (see launcher) so the
// result is deterministic for a fixed launch geometry + seed.
//
// PRECONDITION: blockDim.x is a power of two (the tree reduction relies on it).
__global__ void mc_european_kernel(OptionSpec opt,
                                    std::size_t n_paths,
                                    unsigned long long seed,
                                    double* __restrict__ block_sums,
                                    double* __restrict__ block_sum_sqs) {
    // 2 * blockDim.x doubles of dynamic shared memory: [sums | sum_sqs].
    extern __shared__ double sdata[];
    double* s_sum    = sdata;
    double* s_sum_sq = sdata + blockDim.x;

    const unsigned    tid    = threadIdx.x;
    const std::size_t gid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x)  * blockDim.x;

    // Counter-based Philox: cheap to init, tiny state, distinct stream per thread
    // (subsequence = global thread id). Sequential draws within a thread advance
    // the same stream, so the whole simulation is reproducible for a fixed seed.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, gid, 0, &state);

    // GBM terminal value: S_T = S0 * exp(drift + diffusion * Z),  Z ~ N(0,1).
    const double drift     = (opt.rate - 0.5 * opt.vol * opt.vol) * opt.maturity;
    const double diffusion = opt.vol * sqrt(opt.maturity);
    const int    phi       = (opt.type == OptionType::Call) ? 1 : -1;

    double local_sum    = 0.0;
    double local_sum_sq = 0.0;

    for (std::size_t i = gid; i < n_paths; i += stride) {
        const double z   = curand_normal_double(&state);
        const double sT  = opt.spot * exp(drift + diffusion * z);
        const double pay = fmax(static_cast<double>(phi) * (sT - opt.strike), 0.0);
        local_sum    += pay;
        local_sum_sq += pay * pay;
    }

    s_sum[tid]    = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Tree reduction (requires power-of-two blockDim.x).
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]    += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x]    = s_sum[0];
        block_sum_sqs[blockIdx.x] = s_sum_sq[0];
    }
}

// ── Host launch wrapper ──────────────────────────────────────────────────────
namespace detail {

MCResult launch_european_mc(const OptionSpec& opt, const MCConfig& cfg) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    const unsigned    threads     = cfg.threads_per_block;
    const std::size_t shmem_bytes = std::size_t{2} * threads * sizeof(double);

    // Grid sizing. blocks == 0 means "fill the device": ask the occupancy API how
    // many blocks of this kernel fit per SM at this block size + shared-mem usage,
    // times the SM count. The grid-stride loop decouples grid size from n_paths.
    unsigned blocks = cfg.blocks;
    if (blocks == 0) {
        int sm_count = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &sm_count, cudaDevAttrMultiProcessorCount, device));

        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, mc_european_kernel,
            static_cast<int>(threads), shmem_bytes));

        blocks = static_cast<unsigned>(sm_count) *
                 static_cast<unsigned>(blocks_per_sm);
        if (blocks == 0) {
            blocks = static_cast<unsigned>(sm_count > 0 ? sm_count : 1);
        }
    }

    // For small runs, don't launch more threads than there are paths — the extra
    // threads would just do zero loop iterations and write zeros.
    if (cfg.n_paths > 0) {
        const std::size_t total = static_cast<std::size_t>(blocks) * threads;
        if (total > cfg.n_paths) {
            const std::size_t needed = (cfg.n_paths + threads - 1) / threads;
            blocks = static_cast<unsigned>(needed == 0 ? 1 : needed);
        }
    }

    // Per-block partials live on the device; one (sum, sum_sq) pair per block.
    double* d_block_sums    = nullptr;
    double* d_block_sum_sqs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums,    blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_sum_sqs, blocks * sizeof(double)));

    mc_european_kernel<<<blocks, threads, shmem_bytes>>>(
        opt, cfg.n_paths, static_cast<unsigned long long>(cfg.seed),
        d_block_sums, d_block_sum_sqs);
    CUDA_CHECK_KERNEL();

    std::vector<double> h_sums(blocks);
    std::vector<double> h_sum_sqs(blocks);
    CUDA_CHECK(cudaMemcpy(h_sums.data(), d_block_sums,
                          blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_sqs.data(), d_block_sum_sqs,
                          blocks * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_sum_sqs));

    // Final cross-block reduction, in fixed array order ⇒ deterministic.
    double sum    = 0.0;
    double sum_sq = 0.0;
    for (unsigned i = 0; i < blocks; ++i) {
        sum    += h_sums[i];
        sum_sq += h_sum_sqs[i];
    }

    const double n           = static_cast<double>(cfg.n_paths);
    const double mean_payoff = sum / n;
    const double mean_sq     = sum_sq / n;

    // Population variance of the payoff, then Bessel-corrected to sample variance.
    // The clamp guards the rare case where round-off drives E[X^2]-E[X]^2 slightly
    // negative (deep ITM, tiny variance).
    double var_payoff = mean_sq - mean_payoff * mean_payoff;
    if (var_payoff < 0.0) var_payoff = 0.0;
    const double sample_var = (n > 1.0) ? var_payoff * (n / (n - 1.0)) : 0.0;

    const double disc      = std::exp(-opt.rate * opt.maturity);
    const double price     = disc * mean_payoff;
    const double std_error = disc * std::sqrt(sample_var / n);

    return MCResult{
        .price     = price,
        .std_error = std_error,
        .n_paths   = cfg.n_paths,
    };
}

}  // namespace detail
}  // namespace qmc
