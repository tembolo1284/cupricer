// include/qmc/pricer.hpp
#ifndef QMC_PRICER_HPP
#define QMC_PRICER_HPP

#include <bit>            // std::has_single_bit
#include <string_view>

#include "qmc/detail/mc_launch.hpp"   // MCResult, detail::launch_european_mc

namespace qmc {

// Why a pricing request was rejected.
enum class Error {
    InvalidSpot,
    InvalidStrike,
    InvalidVolatility,
    InvalidMaturity,
    NoPaths,
    InvalidBlockSize,
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

// Small C++20 replacement for std::expected<MCResult, Error>.
struct PriceResult {
    bool ok{};
    MCResult value{};
    Error error{};
};

[[nodiscard]] inline PriceResult
price_european(const OptionSpec& opt, const MCConfig& cfg) {
    if (!(opt.spot > 0.0)) {
        return PriceResult{false, {}, Error::InvalidSpot};
    }

    if (!(opt.strike > 0.0)) {
        return PriceResult{false, {}, Error::InvalidStrike};
    }

    if (!(opt.vol > 0.0)) {
        return PriceResult{false, {}, Error::InvalidVolatility};
    }

    if (!(opt.maturity > 0.0)) {
        return PriceResult{false, {}, Error::InvalidMaturity};
    }

    if (cfg.n_paths == 0) {
        return PriceResult{false, {}, Error::NoPaths};
    }

    if (!std::has_single_bit(cfg.threads_per_block) ||
        cfg.threads_per_block < 32u ||
        cfg.threads_per_block > 1024u) {
        return PriceResult{false, {}, Error::InvalidBlockSize};
    }

    return PriceResult{
        true,
        detail::launch_european_mc(opt, cfg),
        {}
    };
}

}  // namespace qmc

#endif  // QMC_PRICER_HPP
