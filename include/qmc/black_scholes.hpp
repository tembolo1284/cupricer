// include/qmc/black_scholes.hpp
#ifndef QMC_BLACK_SCHOLES_HPP
#define QMC_BLACK_SCHOLES_HPP

#include "qmc/option.hpp"

namespace qmc {

// Standard normal CDF, N(x) = P(Z <= x), via the complementary error function:
//   N(x) = 0.5 * erfc(-x / sqrt(2))
// erfc is used rather than (1 + erf)/2 because it stays accurate in the left
// tail where 1 + erf(...) loses precision to cancellation. Public because the
// tests and a later analytic-Greeks phase both reuse it.
[[nodiscard]] double norm_cdf(double x) noexcept;

// Closed-form Black-Scholes-Merton price of a European option — the oracle the
// Monte Carlo result is validated against.
//
//   d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
//   call = S N(d1) - K e^{-rT} N(d2)
//   put  = K e^{-rT} N(-d2) - S N(-d1)
//
// Non-dividend-paying underlying. Inputs are assumed already validated by the
// caller (positive spot/strike, non-negative vol/maturity) — this is the
// oracle, not the user-facing entry point, so it doesn't re-check.
//
// The degenerate limit sigma*sqrt(T) == 0 would divide by zero in d1/d2, so it
// is handled explicitly as the discounted deterministic payoff. A validation
// oracle that returns NaN at a boundary is worse than useless, so this case
// returns a finite, correct value.
[[nodiscard]] double black_scholes_price(const OptionSpec& opt) noexcept;

}  // namespace qmc

#endif  // QMC_BLACK_SCHOLES_HPP
