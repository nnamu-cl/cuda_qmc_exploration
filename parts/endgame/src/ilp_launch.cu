#include "parts/endgame/src/philox_draw.cuh"
#include "parts/endgame/src/sampler.h"

#include <algorithm>

#include "harness/timing.h"

namespace qmc::endgame {
namespace {

template <int kSamples>
__global__ void SamplePhiloxIlpKernel(
    const qmc::alias::AliasBin* __restrict__ radial, int n_radial,
    const qmc::alias::AliasBin* __restrict__ theta, int n_theta,
    qmc::alias::PackedXyzw* __restrict__ packed, float* __restrict__ x,
    float* __restrict__ y, float* __restrict__ z, float* __restrict__ density,
    float* __restrict__ r, float* __restrict__ theta_out, float* __restrict__ phi,
    int n, int principal, int l, int m_abs, float radial_norm, float y_norm,
    std::uint64_t seed) {
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int stride = static_cast<int>(blockDim.x * gridDim.x);
  for (int base = tid; base < n; base += stride * kSamples) {
#pragma unroll
    for (int s = 0; s < kSamples; ++s) {
      const int i = base + s * stride;
      if (i < n) {
        const qmc::alias::DrawnSample sample = DrawPhiloxSample(
            static_cast<std::uint32_t>(i), seed, radial, n_radial, theta,
            n_theta, principal, l, m_abs, radial_norm, y_norm);
        StoreSample(sample, i, packed, x, y, z, density, r, theta_out, phi);
      }
    }
  }
}

int MultiprocessorCount() {
  int sm = 0;
  qmc::harness::CheckCudaCall(
      cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0),
      "sm count");
  return sm;
}

template <int kSamples>
int OccupancyFor(int threads) {
  int blocks = 0;
  qmc::harness::CheckCudaCall(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, SamplePhiloxIlpKernel<kSamples>, threads, 0),
      "occupancy");
  return blocks;
}

template <int kSamples>
void LaunchS(const qmc::alias::AliasBin* radial, int n_radial,
             const qmc::alias::AliasBin* theta, int n_theta,
             qmc::alias::PackedXyzw* packed, float* x, float* y, float* z,
             float* density, float* r, float* theta_out, float* phi, int n,
             int principal, int l, int m_abs, float radial_norm, float y_norm,
             std::uint64_t seed, const LaunchConfig& cfg) {
  int blocks = cfg.grid_blocks;
  if (blocks <= 0) {
    const int occ = OccupancyFor<kSamples>(cfg.threads);
    blocks = MultiprocessorCount() * std::max(occ, 1);
  }
  SamplePhiloxIlpKernel<kSamples><<<blocks, cfg.threads>>>(
      radial, n_radial, theta, n_theta, packed, x, y, z, density, r, theta_out,
      phi, n, principal, l, m_abs, radial_norm, y_norm, seed);
  qmc::harness::CheckCuda("SamplePhiloxIlpKernel");
}

}  // namespace

int IlpOccupancyBlocks(int samples_per_thread, int threads) {
  switch (samples_per_thread) {
    case 2:
      return OccupancyFor<2>(threads);
    case 4:
      return OccupancyFor<4>(threads);
    case 8:
      return OccupancyFor<8>(threads);
    default:
      return OccupancyFor<1>(threads);
  }
}

int IlpGridBlocks(int samples_per_thread, int threads) {
  return MultiprocessorCount() * std::max(IlpOccupancyBlocks(samples_per_thread, threads), 1);
}

void LaunchPhiloxIlp(const qmc::alias::AliasBin* radial, int n_radial,
                     const qmc::alias::AliasBin* theta, int n_theta,
                     qmc::alias::PackedXyzw* packed, float* x, float* y,
                     float* z, float* density, float* r, float* theta_out,
                     float* phi, int n, int principal, int l, int m_abs,
                     float radial_norm, float y_norm, std::uint64_t seed,
                     const LaunchConfig& cfg) {
  switch (cfg.samples_per_thread) {
    case 2:
      LaunchS<2>(radial, n_radial, theta, n_theta, packed, x, y, z, density, r,
                 theta_out, phi, n, principal, l, m_abs, radial_norm, y_norm,
                 seed, cfg);
      return;
    case 4:
      LaunchS<4>(radial, n_radial, theta, n_theta, packed, x, y, z, density, r,
                 theta_out, phi, n, principal, l, m_abs, radial_norm, y_norm,
                 seed, cfg);
      return;
    case 8:
      LaunchS<8>(radial, n_radial, theta, n_theta, packed, x, y, z, density, r,
                 theta_out, phi, n, principal, l, m_abs, radial_norm, y_norm,
                 seed, cfg);
      return;
    default:
      LaunchS<1>(radial, n_radial, theta, n_theta, packed, x, y, z, density, r,
                 theta_out, phi, n, principal, l, m_abs, radial_norm, y_norm,
                 seed, cfg);
      return;
  }
}

}  // namespace qmc::endgame
