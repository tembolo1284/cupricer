// include/qmc/detail/mc_launch.hpp
#ifndef QMC_DETAIL_MC_LAUNCH_HPP
#define QMC_DETAIL_MC_LAUNCH_HPP

#include <cstddef>

#include "qmc/option.hpp"
#include "qmc/mc_config.hpp"

namespace qmc {

// Outcome of a Monte Carlo pricing run.
struct MCResult {
    double price;         // discounted MC estimate of the option value
    double std_error;     // standard error of that estimate (1 sigma)
    std::size_t n_paths;  // number of paths actually simulated
};

namespace detail {

// C++20-clean entry point, implemented in pricer.cu. No std::expected here:
// this header is included by the .cu, which nvcc compiles at the device
// standard (C++20). The C++23 public API lives one layer up in pricer.hpp.
// Preconditions — positive spot/strike/vol/maturity, threads_per_block a
// power of two and a multiple of 32 — are validated by that wrapper BEFORE
// this is called. This function assumes them and does not re-check.
MCResult launch_european_mc(const OptionSpec& opt, const MCConfig& cfg);

}  // namespace detail
}  // namespace qmc

#endif  // QMC_DETAIL_MC_LAUNCH_HPP
