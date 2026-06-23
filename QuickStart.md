# Quick Start

A copy-paste path from clone to a validated price. For the full picture see [README.md](README.md).

## 1. Prerequisites

- NVIDIA GPU + CUDA Toolkit **12.x or newer** (13.3+ for C++23 device code)
- **CMake 3.24+**, a **C++23 host compiler** (GCC 14+ / Clang 17+), and ideally **Ninja**
- Network access on the first build (CMake fetches GoogleTest)

Verify the toolchain:

```bash
nvcc --version      # CUDA toolkit version
cmake --version     # need >= 3.24
g++ --version       # need a C++23 stdlib (GCC 14+)
```

## 2. Build

```bash
cmake -B build -G Ninja
cmake --build build -j
```

Configure prints the detected CUDA version and the host/device C++ standards it settled on.

## 3. Run

```bash
./build/mc_pricer
```

You want the difference, shown in standard errors, to be small (well under 2) and the
analytic price reported as `INSIDE` the 95% confidence interval.

## 4. Test

```bash
ctest --test-dir build --output-on-failure
```

All cases should pass. The statistical and convergence tests launch real kernels, so run
this on the GPU machine.

---

## Troubleshooting

- **`fatal error: print: No such file` or `std::println is not a member of std`**
  Your host standard library is too old. Point CMake at GCC 14+:
  ```bash
  cmake -B build -G Ninja \
        -DCMAKE_CXX_COMPILER=g++-14 \
        -DCMAKE_CUDA_HOST_COMPILER=g++-14
  ```

- **CUDA architecture detection fails at configure time**
  `CUDA_ARCHITECTURES native` probes the installed GPU, so `nvcc` must be on your `PATH`
  when you run the configure step.

- **Tests fail to launch / "no CUDA-capable device"**
  The accuracy, convergence, and parity tests need a real NVIDIA device. Run the suite on
  the GPU machine, not a CPU-only box.

- **FetchContent can't reach GitHub**
  The first configure clones GoogleTest from `github.com`. Make sure outbound network
  access is available, or pre-populate CMake's FetchContent cache.
