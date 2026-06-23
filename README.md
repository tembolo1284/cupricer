# cuPricer

> From-scratch CUDA Monte Carlo option pricer in modern C++23, validated against closed-form Black-Scholes.

cuPricer prices European options by Monte Carlo simulation on the GPU, with every
result checked against the analytic Black-Scholes price. The kernel, reduction, and
host orchestration are written from scratch — the only library leaned on for the
core algorithm is cuRAND, for the normal random draws.

This is **Phase 1**: European vanilla options, validated. See the [Roadmap](#roadmap)
for what comes next.

---

## Features

- GPU Monte Carlo pricing of European **calls and puts** under geometric Brownian motion.

- Hand-written CUDA kernel with a shared-memory block reduction; the final cross-block
  reduction runs on the host so results are **bitwise-reproducible** for a fixed seed and
  launch geometry.

- Reports both the price estimate **and its standard error**, so correctness is judged
  statistically — the analytic price should fall inside the MC confidence interval —
  rather than against a hand-tuned tolerance.

- Modern C++23 public API: input validation via `std::expected`, formatted output via
  `std::print`.

- Occupancy-based grid sizing on top of a grid-stride kernel, which decouples grid size
  from path count.

- GoogleTest suite covering the oracle, a five-sigma accuracy gate, the `1/sqrt(N)`
  convergence rate, put-call parity, and every input-validation path.

---

## Requirements

| Component     | Minimum            | Notes                                                                 |

| ------------- | ------------------ | --------------------------------------------------------------------- |

| NVIDIA GPU    | any CUDA-capable   | the GPU-dependent tests must run on the GPU machine                   |

| CUDA Toolkit  | 12.x               | 13.3+ enables C++23 *device* code; otherwise device code stays C++20  |

| CMake         | 3.24               | for `CUDA_ARCHITECTURES native`; 3.25+ for the C++23 device bump      |

| Host compiler | GCC 14+ / Clang 17+| needs a C++23 standard library (`std::print`, `std::expected`)        |

| Ninja         | optional           | recommended — noticeably faster `.cu` rebuilds                        |

The first configure needs network access to fetch GoogleTest. Check your toolchain with
`nvcc --version` and `cmake --version`.

---

## Build

```bash
cmake -B build -G Ninja
cmake --build build -j
```

Drop `-G Ninja` to use Make instead. The configure step prints the detected CUDA version
and the resolved host/device C++ standards.

## Run

```bash
./build/mc_pricer
```

Sample output:

```
European call | S=100.00 K=100.00 r=0.0500 sigma=0.2000 T=1.00
paths      : 10000000
MC price   : 10.45....
std error  : 0.00....
analytic   : 10.450584
difference : +0.00....  (+0.xx std errors)
95% CI     : [10.45...., 10.45....]  analytic INSIDE
```

The headline number is the difference expressed in **standard errors**. For a correct
kernel it sits comfortably under 2 almost every run; a large value means a bug, not bad luck.

## Test

```bash
ctest --test-dir build --output-on-failure
```

The `test_european` suite covers:

- **Oracle** — analytic call/put match the canonical ATM benchmarks.
- **Accuracy** — MC lands within 5 standard errors of analytic (fixed seed).
- **Convergence** — the standard error shrinks at the `1/sqrt(N)` rate. This catches a
  class of bug a point estimate cannot see: correlated paths or threads sharing an RNG
  stream can land near the right price yet converge at the wrong rate.
- **Parity** — put-call parity holds on the MC estimates directly (needs no oracle).
- **Validation** — every malformed input is rejected before a kernel launches.

---

## Project structure

```
cupricer/

├── CMakeLists.txt

├── README.md

├── QUICK_START.md

├── include/

│   └── qmc/

│       ├── option.hpp            # OptionSpec: contract + market state

│       ├── mc_config.hpp         # MCConfig: paths, seed, launch geometry


│       ├── black_scholes.hpp     # closed-form oracle (declarations)

│       ├── cuda_check.hpp        # CUDA_CHECK error-handling macros

│       ├── pricer.hpp            # C++23 public API -> std::expected<MCResult, Error>

│       └── detail/

│           └── mc_launch.hpp     # C++20-clean host/device boundary

├── src/

│   ├── black_scholes.cpp         # C++23 host: analytic price via std::erf


│   ├── pricer.cu                 # C++20 device: kernel + reduction + launcher

│   └── main.cpp                  # C++23 host: run, validate vs BS, print

└── tests/

    └── test_european.cpp         # GoogleTest suite
```

---

## Design notes

- **Standard split.** Host-only translation units compile as C++23; the single `.cu`
  and its boundary header stay C++20-clean, because nvcc compiles a `.cu` file's host half
  at the *device* standard. The boundary lives in `include/qmc/detail/mc_launch.hpp`, which
  is why `std::expected` appears only one layer above it, in `pricer.hpp`.

- **Two error channels.** Bad *inputs* are recoverable and reported through `std::expected`;
  genuine *device* failures (out-of-memory, launch faults) abort loudly via `CUDA_CHECK`.
  Different kinds of problem, different mechanisms.

- **Determinism by construction.** Per-thread RNG streams plus a host-ordered final
  reduction make a run reproducible to the bit for a fixed seed and launch geometry — so
  "did my number actually move?" after a kernel change is always answerable.

---

## Roadmap

- **Phase 1 — European vanilla, validated.** ✅ *(current)*

- **Phase 2 — Reductions, properly.** Warp-shuffle block reduction (`__shfl_down_sync`),
  benchmarked against `cub::DeviceReduce` and `thrust::reduce`; single- vs double-precision
  throughput; coalescing and occupancy tuning.

- **Phase 3 — Path-dependent options.** Multi-step path simulation for arithmetic-average
  Asian and barrier options — where Monte Carlo earns its keep, with no closed form to lean on.

- **Phase 4 — Greeks + benchmarking.** Pathwise and likelihood-ratio Greeks, throughput
  comparison against a CPU baseline, and light roofline analysis.

- **Later.** A from-scratch counter-based Philox RNG to drop the cuRAND dependency;
  variance reduction (antithetic and control variates, Sobol quasi-Monte-Carlo); and a
  finite-difference fork (Crank-Nicolson via cuSPARSE `gtsv` or a hand-rolled cyclic reduction).

---
