// src/main.cpp
#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <iomanip>    // std::fixed, std::setprecision
#include <iostream>

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
    if (!result.ok) {
        std::cerr << "pricing rejected: "
                  << qmc::error_message(result.error)
                  << '\n';
        return EXIT_FAILURE;
    }

    const double analytic = qmc::black_scholes_price(opt);
    const double mc       = result.value.price;
    const double se       = result.value.std_error;
    const double diff     = mc - analytic;

    // The validation metric: how many standard errors separate the MC estimate
    // from the analytic truth. |z| under ~3 is consistent with a correct kernel;
    // a large z means a bug, not bad luck. (z itself is undefined if se == 0,
    // e.g. a degenerate zero-variance payoff — guard against div-by-zero.)
    const double z = (se > 0.0) ? diff / se : 0.0;

    std::cout << "European "
              << (opt.type == qmc::OptionType::Call ? "call" : "put")
              << " | S=" << std::fixed << std::setprecision(2) << opt.spot
              << " K=" << std::fixed << std::setprecision(2) << opt.strike
              << " r=" << std::fixed << std::setprecision(4) << opt.rate
              << " sigma=" << std::fixed << std::setprecision(4) << opt.vol
              << " T=" << std::fixed << std::setprecision(2) << opt.maturity
              << '\n';

    std::cout << "paths      : " << result.value.n_paths << '\n';
    std::cout << "MC price   : " << mc << '\n';
    std::cout << "std error  : " << se << '\n';
    std::cout << "analytic   : " << analytic << '\n';

    std::cout << std::showpos << std::fixed << std::setprecision(6)
              << "difference : " << diff
              << std::noshowpos << "  ("
              << std::showpos << std::fixed << std::setprecision(2)
              << z
              << std::noshowpos << " std errors)\n";

    std::cout << std::fixed << std::setprecision(6)
              << "95% CI     : [" << (mc - 1.96 * se)
              << ", " << (mc + 1.96 * se)
              << "]  analytic "
              << ((analytic >= mc - 1.96 * se && analytic <= mc + 1.96 * se)
                    ? "INSIDE" : "OUTSIDE")
              << " interval\n";

    return EXIT_SUCCESS;
}
