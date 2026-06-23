// include/qmc/mc_config.hpp
#ifndef QMC_MC_CONFIG_HPP
#define QMC_MC_CONFIG_HPP

#include <cstddef>
#include <cstdint>

namespace qmc {

// Knobs for a single Monte Carlo pricing run.
struct MCConfig {
    // Total number of simulated terminal-price paths. More paths ⇒ tighter
    // confidence interval, with error shrinking like 1/sqrt(n_paths).
    std::size_t n_paths = 1'000'000;

    // RNG seed. Default is the 64-bit golden-ratio constant — arbitrary but
    // well-mixed. Fix this for reproducible runs; vary it for independent ones.
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;

    // CUDA block size (threads per block). Keep it a multiple of 32 (warp size);
    // a power of two also keeps the shared-memory reduction simple. 256 is a
    // solid default for a compute-light kernel like this.
    unsigned threads_per_block = 256;

    // Number of blocks in the grid. 0 ⇒ the launcher sizes the grid to saturate
    // the GPU. Because the kernel uses a grid-stride loop, the grid is decoupled
    // from n_paths: each thread strides over many paths, so you pick a grid that
    // fills the device's SMs rather than one thread per path.
    unsigned blocks = 0;
};

}  // namespace qmc
#endif // QMC_MC_CONFIG_HPP
