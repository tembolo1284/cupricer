// include/qmc/option.hpp
#ifndef QMC_OPTION_HPP
#define QMC_OPTION_HPP

#include <cstdint>

namespace qmc {

// Which side of the contract we're pricing.
enum class OptionType : std::uint8_t {
    Call,
    Put,
};

// A fully-specified European option contract + market state.
// Plain aggregate on purpose: trivially copyable, so it can be passed
// by value straight into a kernel, and its fields are readable from both
// host and device with no annotations.
struct OptionSpec {
    double spot;       // S0 — current underlying price
    double strike;     // K  — strike price
    double rate;       // r  — continuously-compounded risk-free rate (annualized)
    double vol;        // σ  — volatility (annualized)
    double maturity;   // T  — time to expiry, in years
    OptionType type{OptionType::Call};
};

}  // namespace qmc

#endif // QMC_OPTION_HPP
