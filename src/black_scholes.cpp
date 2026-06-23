// src/black_scholes.cpp
#include "qmc/black_scholes.hpp"

#include <algorithm>   // std::max
#include <cmath>       // std::exp, std::log, std::sqrt, std::erfc
#include <numbers>     // std::numbers::sqrt2

namespace qmc {

double norm_cdf(double x) noexcept {
    // N(x) = 0.5 * erfc(-x / sqrt(2)). The erfc form keeps full precision in the
    // left tail, unlike 0.5 * (1 + erf(x / sqrt(2))) which cancels there.
    return 0.5 * std::erfc(-x / std::numbers::sqrt2);
}

double black_scholes_price(const OptionSpec& opt) noexcept {
    const double S     = opt.spot;
    const double K     = opt.strike;
    const double r     = opt.rate;
    const double sigma = opt.vol;
    const double T     = opt.maturity;

    const double disc       = std::exp(-r * T);    // e^{-rT}, discount factor
    const double vol_sqrt_t = sigma * std::sqrt(T); // σ√T, total log-return stddev

    // +1 for a call, -1 for a put. Lets one expression cover both via symmetry.
    const int phi = (opt.type == OptionType::Call) ? 1 : -1;

    // Degenerate limit σ√T == 0 (zero vol or zero maturity): no diffusion, so the
    // terminal price is the deterministic forward F = S e^{rT}. Return the
    // discounted intrinsic on that forward — finite and correct, never NaN.
    if (vol_sqrt_t == 0.0) {
        const double fwd    = S * std::exp(r * T);
        const double payoff = std::max(phi * (fwd - K), 0.0);
        return disc * payoff;
    }

    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t;
    const double d2 = d1 - vol_sqrt_t;

    // call = S·N(d1) − K·e^{-rT}·N(d2)
    // put  = K·e^{-rT}·N(−d2) − S·N(−d1)
    // Both collapse into one line by folding the sign through phi:
    return phi * S * norm_cdf(phi * d1) - phi * K * disc * norm_cdf(phi * d2);
}

}  // namespace qmc
