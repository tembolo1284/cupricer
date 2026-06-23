// tests/test_european.cpp
#include <cmath>     // std::abs, std::exp, std::log, std::nan
#include <cstddef>

#include <gtest/gtest.h>

#include "qmc/black_scholes.hpp"
#include "qmc/option.hpp"
#include "qmc/mc_config.hpp"
#include "qmc/pricer.hpp"

namespace {

constexpr double kSigmaGate = 5.0;

// Canonical ATM benchmark: analytic call ≈ 10.4506, put ≈ 5.5735.
qmc::OptionSpec atm_call() {
    return qmc::OptionSpec{
        .spot = 100.0, .strike = 100.0, .rate = 0.05,
        .vol = 0.20, .maturity = 1.0, .type = qmc::OptionType::Call};
}

// Assert a contract is rejected before any launch, with the expected reason.
void expect_rejected(const qmc::OptionSpec& opt,
                     const qmc::MCConfig& cfg,
                     qmc::Error expected) {
    const auto r = qmc::price_european(opt, cfg);
    ASSERT_FALSE(r.ok) << "expected rejection, but pricing succeeded";
    EXPECT_EQ(r.error, expected);
}

}  // namespace

TEST(BlackScholesOracle, CanonicalAtmValues) {
    const auto call = atm_call();
    EXPECT_NEAR(qmc::black_scholes_price(call), 10.450584, 1e-4);

    auto put = call;
    put.type = qmc::OptionType::Put;
    EXPECT_NEAR(qmc::black_scholes_price(put), 5.573526, 1e-4);
}

TEST(EuropeanMC, CallWithinFiveSigma) {
    const auto opt = atm_call();
    const qmc::MCConfig cfg{.n_paths = 10'000'000};

    const auto result = qmc::price_european(opt, cfg);
    ASSERT_TRUE(result.ok)
        << "rejected: " << qmc::error_message(result.error);
    ASSERT_GT(result.value.std_error, 0.0);

    const double analytic = qmc::black_scholes_price(opt);
    const double z = (result.value.price - analytic) / result.value.std_error;
    EXPECT_LT(std::abs(z), kSigmaGate)
        << "MC=" << result.value.price << " analytic=" << analytic
        << " se=" << result.value.std_error << " z=" << z;
}

TEST(EuropeanMC, PutWithinFiveSigma) {
    auto opt = atm_call();
    opt.type = qmc::OptionType::Put;
    const qmc::MCConfig cfg{.n_paths = 10'000'000};

    const auto result = qmc::price_european(opt, cfg);
    ASSERT_TRUE(result.ok)
        << "rejected: " << qmc::error_message(result.error);
    ASSERT_GT(result.value.std_error, 0.0);

    const double analytic = qmc::black_scholes_price(opt);
    const double z = (result.value.price - analytic) / result.value.std_error;
    EXPECT_LT(std::abs(z), kSigmaGate)
        << "MC=" << result.value.price << " analytic=" << analytic
        << " se=" << result.value.std_error << " z=" << z;
}

TEST(EuropeanMC, ErrorConvergesAtRootN) {
    const auto opt = atm_call();

    auto se_at = [&](std::size_t n) {
        const qmc::MCConfig cfg{.n_paths = n};
        const auto r = qmc::price_european(opt, cfg);
        EXPECT_TRUE(r.ok)
            << "rejected: " << qmc::error_message(r.error);
        return r.value.std_error;
    };

    const double se1  = se_at(1'000'000);
    const double se4  = se_at(4'000'000);
    const double se16 = se_at(16'000'000);

    const double p1 = std::log(se1 / se4)  / std::log(4.0);
    const double p2 = std::log(se4 / se16) / std::log(4.0);

    EXPECT_NEAR(p1, 0.5, 0.1) << "se(1M)=" << se1 << " se(4M)="  << se4;
    EXPECT_NEAR(p2, 0.5, 0.1) << "se(4M)=" << se4 << " se(16M)=" << se16;
}

TEST(EuropeanMC, LowVolatilityApproachesAnalytic) {
    auto opt = atm_call();
    opt.vol = 1e-6;

    const qmc::MCConfig cfg{.n_paths = 1'000'000};
    const auto result = qmc::price_european(opt, cfg);
    ASSERT_TRUE(result.ok)
        << "rejected: " << qmc::error_message(result.error);

    const double analytic = qmc::black_scholes_price(opt);
    EXPECT_NEAR(result.value.price, analytic, 1e-3);
    EXPECT_LT(result.value.std_error, 1e-2);
}

TEST(EuropeanMC, PutCallParity) {
    const auto call = atm_call();
    auto put = call;
    put.type = qmc::OptionType::Put;

    const qmc::MCConfig cfg{.n_paths = 10'000'000};
    const auto rc = qmc::price_european(call, cfg);
    const auto rp = qmc::price_european(put,  cfg);

    ASSERT_TRUE(rc.ok)
        << "call rejected: " << qmc::error_message(rc.error);
    ASSERT_TRUE(rp.ok)
        << "put rejected: " << qmc::error_message(rp.error);

    const double parity =
        call.spot - call.strike * std::exp(-call.rate * call.maturity);

    EXPECT_NEAR(rc.value.price - rp.value.price, parity, 5e-2);
}

TEST(Validation, RejectsBadContracts) {
    const qmc::MCConfig cfg{};

    auto spot0 = atm_call();  spot0.spot = 0.0;
    expect_rejected(spot0, cfg, qmc::Error::InvalidSpot);

    auto spotN = atm_call();  spotN.spot = std::nan("");
    expect_rejected(spotN, cfg, qmc::Error::InvalidSpot);

    auto strk = atm_call();  strk.strike = -1.0;
    expect_rejected(strk, cfg, qmc::Error::InvalidStrike);

    auto vol0 = atm_call();  vol0.vol = 0.0;
    expect_rejected(vol0, cfg, qmc::Error::InvalidVolatility);

    auto matN = atm_call();  matN.maturity = -0.5;
    expect_rejected(matN, cfg, qmc::Error::InvalidMaturity);
}

TEST(Validation, RejectsBadConfig) {
    const auto opt = atm_call();

    expect_rejected(opt, qmc::MCConfig{.n_paths = 0},
                    qmc::Error::NoPaths);

    expect_rejected(opt, qmc::MCConfig{.threads_per_block = 100},
                    qmc::Error::InvalidBlockSize);

    expect_rejected(opt, qmc::MCConfig{.threads_per_block = 16},
                    qmc::Error::InvalidBlockSize);

    expect_rejected(opt, qmc::MCConfig{.threads_per_block = 2048},
                    qmc::Error::InvalidBlockSize);
}
