// include/qmc/pricer.hpp
#ifndef QMC_PRICER_HPP
#define QMC_PRICER_HPP

#include <bit>            // std::has_single_bit
#include <expected>       // std::expected, std::unexpected  (C++23)
#include <string_view>

#include "qmc/detail/mc_launch.hpp"   // MCResult, detail::launch_european_mc

namespace qmc {

// Why a pricing request was rejected. These are all caught BEFORE any GPU work:
// bad inputs are recoverable and reported here, whereas genuine device failures
// (OOM, launch faults) abort loudly via CUDA_CHECK. Two different failure
// channels for two different kinds of problem.
enum class Error {
    InvalidSpot,        // spot <= 0 or NaN
    InvalidStrike,      // strike <= 0 or NaN
    InvalidVolatility,  // vol <= 0 or NaN
    InvalidMaturity,    // maturity <= 0 or NaN
    NoPaths,            // n_paths == 0
    InvalidBlockSize,   // threads_per_block not a power of two in [32, 1024]
};

[[nodiscard]] constexpr std::string_view error_message(Error e) noexcept {
    switch (e) {
        case Error::InvalidSpot:       return "spot must be positive";
        case Error::InvalidStrike:     return "strike must be positive";
        case Error::InvalidVolatility: return "volatility must be positive";
        case Error::InvalidMaturity:   return "maturity must be positive";
        case Error::NoPaths:           return "n_paths must be non-zero";
        case Error::InvalidBlockSize:  return "threads_per_block must be a power of two in [32, 1024]";
    }
    return "unknown error";
}

// Public entry point: validate, then price. Returns the MC estimate on success
// or an Error on bad input. Not noexcept — launch_european_mc allocates a small
// host buffer, so std::bad_alloc can in principle escape; that's a catastrophic
// resource failure (consistent with the abort-on-device-failure stance), not an
// input error, so it is deliberately not folded into the expected channel.
[[nodiscard]] inline std::expected<MCResult, Error>
price_european(const OptionSpec& opt, const MCConfig& cfg) {
    // Each check is written as !(x > 0) rather than (x <= 0) so that a NaN input
    // is also rejected: every comparison against NaN is false, so !(NaN > 0) is
    // true and the contract is refused instead of launching a kernel that would
    // quietly produce NaN. Note the risk-free rate is intentionally unconstrained
    // — negative rates are economically valid.
    if (!(opt.spot > 0.0))     return std::unexpected(Error::InvalidSpot);
    if (!(opt.strike > 0.0))   return std::unexpected(Error::InvalidStrike);
    if (!(opt.vol > 0.0))      return std::unexpected(Error::InvalidVolatility);
    if (!(opt.maturity > 0.0)) return std::unexpected(Error::InvalidMaturity);

    if (cfg.n_paths == 0) return std::unexpected(Error::NoPaths);

    // The kernel's shared-memory tree reduction requires a power-of-two block
    // size; CUDA caps a block at 1024 threads; below 32 wastes a warp. A power
    // of two in [32, 1024] satisfies "multiple of warp size" for free.
    if (!std::has_single_bit(cfg.threads_per_block) ||
        cfg.threads_per_block < 32u || cfg.threads_per_block > 1024u) {
        return std::unexpected(Error::InvalidBlockSize);
    }

    return detail::launch_european_mc(opt, cfg);
}

}  // namespace qmc

#endif  // QMC_PRICER_HPP
