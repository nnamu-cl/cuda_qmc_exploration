#ifndef HARNESS_TIMING_H_
#define HARNESS_TIMING_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ranges>
#include <vector>

namespace qmc::harness {

inline constexpr int kDefaultWarmup = 3;
inline constexpr int kDefaultReps = 20;

struct TimingConfig {
  int warmup = kDefaultWarmup;
  int reps = kDefaultReps;
};

struct TimingResult {
  std::vector<double> reps_ms;
  double median_ms = 0.0;
};

[[nodiscard]] inline double MedianMs(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::ranges::sort(values);
  const auto n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

inline void CheckCudaCall(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", what,
                 cudaGetErrorString(err));
    std::abort();
  }
}

inline void CheckCuda(const char* what) {
  CheckCudaCall(cudaGetLastError(), what);
}

template <class Launch>
[[nodiscard]] TimingResult TimeCudaLaunch(const TimingConfig& config,
                                          Launch&& launch) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CheckCudaCall(cudaEventCreate(&start), "cudaEventCreate(start)");
  CheckCudaCall(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  for (int i = 0; i < config.warmup; ++i) {
    launch();
  }
  CheckCudaCall(cudaDeviceSynchronize(), "warmup sync");
  TimingResult result;
  result.reps_ms.reserve(static_cast<size_t>(config.reps));
  for (int i = 0; i < config.reps; ++i) {
    CheckCudaCall(cudaEventRecord(start), "cudaEventRecord(start)");
    launch();
    CheckCudaCall(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCudaCall(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0f;
    CheckCudaCall(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    result.reps_ms.push_back(static_cast<double>(ms));
  }
  CheckCudaCall(cudaEventDestroy(start), "cudaEventDestroy(start)");
  CheckCudaCall(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  result.median_ms = MedianMs(result.reps_ms);
  return result;
}

}  // namespace qmc::harness

#endif
