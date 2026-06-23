# ─────────────────────────────────────────────────────────────────────────────
# cuda-mc-pricer — Phase 1: European vanilla, validated against Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.24)   # 3.24+ needed for CUDA_ARCHITECTURES=native

project(cuda_mc_pricer
  VERSION 0.1.0
  DESCRIPTION "From-scratch CUDA Monte Carlo option pricer"
  LANGUAGES CXX CUDA
)

# Default to an optimized build if the user didn't pick one (single-config gens).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA::cudart, CUDA::curand, and CUDAToolkit_VERSION come from here.
find_package(CUDAToolkit REQUIRED)

# ── Language standards ───────────────────────────────────────────────────────
# These variables are PER-LANGUAGE: CMAKE_CXX_STANDARD drives .cpp compilation,
# CMAKE_CUDA_STANDARD drives .cu compilation. A target with both kinds of source
# gets each language compiled at its own standard. That's the whole trick.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Device code: C++20 is the safe floor. nvcc only got OFFICIAL C++23 in CUDA 13.3,
# and CMake's CUDA_STANDARD=23 needs 3.25+. Bump only when both are satisfied;
# otherwise device stays at 20 while host code is C++23 regardless.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Build for the GPU actually installed in this machine (3.24+).
set(CMAKE_CUDA_ARCHITECTURES native)

message(STATUS "CUDA Toolkit : ${CUDAToolkit_VERSION}")
message(STATUS "Host C++     : C++${CMAKE_CXX_STANDARD}")
message(STATUS "Device C++   : C++${CMAKE_CUDA_STANDARD}")

# ── Shared warning / flag set (consumed privately by our targets) ─────────────
add_library(qmc_flags INTERFACE)
target_compile_options(qmc_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  # --expt-relaxed-constexpr lets device code call constexpr host helpers
  # (e.g. shared math/constants). -Xcompiler forwards warnings to the host pass.
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)

# ── Core library: host analytics + the device kernel ─────────────────────────
add_library(qmc STATIC
  src/black_scholes.cpp   # C++20 host: the Black-Scholes oracle
  src/pricer.cu           # C++20 device: kernel + reduction + launch wrapper
)
target_include_directories(qmc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qmc
  PUBLIC  CUDA::cudart CUDA::curand
  PRIVATE qmc_flags
)

# ── CLI executable ───────────────────────────────────────────────────────────
add_executable(mc_pricer src/main.cpp)
target_link_libraries(mc_pricer PRIVATE qmc qmc_flags)

# ── Tests (CTest + GoogleTest) ───────────────────────────────────────────────
include(CTest)   # defines BUILD_TESTING (default ON) and calls enable_testing()
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0          # pinned; bump up as needed
    GIT_SHALLOW    TRUE
  )
  # Honored on Windows (MSVC runtime), harmless on Linux. Standard hygiene.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(test_european tests/test_european.cpp)
  # gtest_main supplies main(); qmc_flags stays PRIVATE so our -Wpedantic
  # never leaks onto GoogleTest's own headers.
  target_link_libraries(test_european
    PRIVATE qmc qmc_flags GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(test_european)   # registers each TEST() with CTest
endif()
