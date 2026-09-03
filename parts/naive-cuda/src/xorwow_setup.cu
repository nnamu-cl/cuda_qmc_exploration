#include "parts/naive-cuda/src/sampler.h"

#include <curand_kernel.h>

#include "harness/timing.h"

namespace qmc::naive {
namespace {

constexpr int kThreads = 256;

__global__ void SetupXorwowKernel(curandStateXORWOW_t* states,
                                  unsigned long long seed, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    curand_init(seed, static_cast<unsigned long long>(i), 0, &states[i]);
  }
}

}  // namespace

std::size_t XorwowStateBytes() { return sizeof(curandStateXORWOW_t); }

void SetupXorwowStates(void* states, std::uint64_t seed, int n) {
  const int blocks = (n + kThreads - 1) / kThreads;
  SetupXorwowKernel<<<blocks, kThreads>>>(
      static_cast<curandStateXORWOW_t*>(states), seed, n);
  qmc::harness::CheckCuda("SetupXorwowKernel");
}

}  // namespace qmc::naive
