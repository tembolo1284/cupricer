// src/main.cpp
#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <print>      // std::println  (C++23; needs libstdc++ 14+ / libc++ 17+)

#include "qmc/black_scholes.hpp"
#include "qmc/option.hpp"
#include "qmc/mc_config.hpp"
#include "qmc/pricer.hpp"

int main() {
    // Canonical ATM benchmark: analytic call ≈ 10.4506, put ≈ 5.5735.
    const qmc::OptionSpec opt{
        .spot = 100.0, .strike = 100.0, .rate = 0.05,
        .vol = 0.20, .maturity = 1.0, .type = qmc::OptionType::Call,
    };

    const qmc::MCConfig cfg{
        .n_paths = 10'000'000,
    };

    const auto result = qmc::price_european(opt, cfg);
    if (!result) {
        std::println(stderr, "pricing rejected: {}",
                     qmc::error_message(result.error()));
        return EXIT_FAILURE;
    }

    const double analytic = qmc::black_scholes_price(opt);
    const double mc        = result->price;
    const double se        = result->std_error;
    const double diff      = mc - analytic;

    // The validation metric: how many standard errors separate the MC estimate
    // from the analytic truth. |z| under ~3 is consistent with a correct kernel;
    // a large z means a bug, not bad luck. (z itself is undefined if se == 0,
    // e.g. a degenerate zero-variance payoff — guard against div-by-zero.)
    const double z = (se > 0.0) ? diff / se : 0.0;

    std::println("European {} | S={:.2f} K={:.2f} r={:.4f} sigma={:.4f} T={:.2f}",
                 opt.type == qmc::OptionType::Call ? "call" : "put",
                 opt.spot, opt.strike, opt.rate, opt.vol, opt.maturity);
    std::println("paths      : {}", result->n_paths);
    std::println("MC price   : {:.6f}", mc);
    std::println("std error  : {:.6f}", se);
    std::println("analytic   : {:.6f}", analytic);
    std::println("difference : {:+.6f}  ({:+.2f} std errors)", diff, z);
    std::println("95% CI     : [{:.6f}, {:.6f}]  analytic {} interval",
                 mc - 1.96 * se, mc + 1.96 * se,
                 (analytic >= mc - 1.96 * se && analytic <= mc + 1.96 * se)
                     ? "INSIDE" : "OUTSIDE");

    return EXIT_SUCCESS;
}
