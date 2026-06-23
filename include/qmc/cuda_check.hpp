// include/qmc/cuda_check.hpp
#ifndef QMC_CUDA_CHECK_HPP
#define QMC_CUDA_CHECK_HPP

#include <cstdio>    // std::fprintf
#include <cstdlib>   // std::abort

#include <cuda_runtime.h>   // cudaError_t, cudaSuccess, cudaGetErrorString

namespace qmc {

// Host-side handler for a CUDA runtime status. On failure it reports
// file:line, the offending expression, and the driver's message, then aborts.
// Aborting (rather than throwing) keeps this usable inside .cu host code, which
// nvcc compiles at the *device* standard (C++20) — so nothing C++23-only here,
// and no exceptions to unwind across the launch boundary.
inline void cuda_check(cudaError_t err, const char* expr,
                       const char* file, int line) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA error at %s:%d\n  call: %s\n  what: %s (%d)\n",
                     file, line, expr, cudaGetErrorString(err),
                     static_cast<int>(err));
        std::abort();
    }
}

}  // namespace qmc

// Wrap any call returning cudaError_t. Evaluates `expr` exactly once.
//   CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
#define CUDA_CHECK(expr) ::qmc::cuda_check((expr), #expr, __FILE__, __LINE__)

// Use immediately after a kernel launch. The launch returns void, so errors
// surface in two separate places: cudaGetLastError() catches a bad launch
// configuration (reported synchronously), and cudaDeviceSynchronize() catches
// faults that occur during execution (asynchronous). Checking only one of the
// two misses half the failure modes.
#define CUDA_CHECK_KERNEL()                  \
    do {                                     \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

#endif  // QMC_CUDA_CHECK_HPP
