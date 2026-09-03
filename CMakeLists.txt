cmake_minimum_required(VERSION 4.3)
project(cuda_qmc_exploration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(FMT_INSTALL OFF)
set(CLI11_PRECOMPILED OFF)
set(JSON_Install OFF)
set(JSON_BuildTests OFF)

FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.7.2
        GIT_SHALLOW TRUE
)
FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.2.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json cli11 fmt)

option(QMC_LINEINFO "Emit -lineinfo for Nsight Compute" OFF)

add_library(qmc_harness
        harness/env.cpp
        harness/results.cpp
        harness/dump.cpp
        harness/verify_gate.cpp
        harness/stats/ks.cpp
        harness/stats/chi2.cpp
        harness/stats/moments.cpp
)
target_include_directories(qmc_harness PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_definitions(qmc_harness PUBLIC
        QMC_DEFAULT_VERIFY_GATE="${CMAKE_BINARY_DIR}/.verify_full_ok")
target_link_libraries(qmc_harness PUBLIC CUDA::cudart nlohmann_json::nlohmann_json fmt::fmt)
target_compile_options(qmc_harness PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
set_target_properties(qmc_harness PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
if (QMC_LINEINFO)
    target_compile_options(qmc_harness PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif ()

add_executable(cuda_qmc_exploration
        main.cpp
        harness/microbench.cu
        harness/stats_smoke.cpp
)
target_include_directories(cuda_qmc_exploration PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_qmc_exploration PRIVATE qmc_harness CUDA::curand CLI11::CLI11 fmt::fmt)
set_target_properties(cuda_qmc_exploration PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
if (QMC_LINEINFO)
    target_compile_options(cuda_qmc_exploration PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif ()

enable_testing()
add_test(NAME verify-fast COMMAND cuda_qmc_exploration stats-smoke)
