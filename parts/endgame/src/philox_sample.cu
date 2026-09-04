#include "parts/endgame/src/philox_draw.cuh"
#include "parts/endgame/src/sampler.h"

#include "harness/timing.h"

namespace qmc::endgame {
namespace {

__global__ void SamplePhiloxKernel(const qmc::alias::AliasBin* __restrict__ radial,
                                   int n_radial,
                                   const qmc::alias::AliasBin* __restrict__ theta,
                                   int n_theta,
                                   qmc::alias::PackedXyzw* __restrict__ packed,
                                   float* __restrict__ x, float* __restrict__ y,
                                   float* __restrict__ z,
                                   float* __restrict__ density,
                                   float* __restrict__ r,
                                   float* __restrict__ theta_out,
                                   float* __restrict__ phi, int n, int principal,
                                   int l, int m_abs, float radial_norm,
                                   float y_norm, std::uint64_t seed) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  const qmc::alias::DrawnSample s =
      DrawPhiloxSample(static_cast<std::uint32_t>(i), seed, radial, n_radial,
                       theta, n_theta, principal, l, m_abs, radial_norm, y_norm);
  StoreSample(s, i, packed, x, y, z, density, r, theta_out, phi);
}

__global__ void DumpPhiloxKernel(std::uint32_t* out, int n, std::uint64_t seed,
                                 std::uint32_t slot) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  const qmc::rng::Philox4x32Ctr ctr = qmc::rng::Philox4x32TenRounds(
      qmc::rng::MakeCounter(static_cast<std::uint32_t>(i), slot),
      qmc::rng::SeedToKey(seed));
  out[static_cast<size_t>(i) * 4 + 0] = ctr.v[0];
  out[static_cast<size_t>(i) * 4 + 1] = ctr.v[1];
  out[static_cast<size_t>(i) * 4 + 2] = ctr.v[2];
  out[static_cast<size_t>(i) * 4 + 3] = ctr.v[3];
}

}  // namespace

void LaunchPhiloxSample(const qmc::alias::AliasBin* radial, int n_radial,
                        const qmc::alias::AliasBin* theta, int n_theta,
                        qmc::alias::PackedXyzw* packed, float* x, float* y,
                        float* z, float* density, float* r, float* theta_out,
                        float* phi, int n, int principal, int l, int m_abs,
                        float radial_norm, float y_norm, std::uint64_t seed,
                        int threads) {
  const int blocks = (n + threads - 1) / threads;
  SamplePhiloxKernel<<<blocks, threads>>>(
      radial, n_radial, theta, n_theta, packed, x, y, z, density, r, theta_out,
      phi, n, principal, l, m_abs, radial_norm, y_norm, seed);
  qmc::harness::CheckCuda("SamplePhiloxKernel");
}

void LaunchDumpPhilox(std::uint32_t* out, int n, std::uint64_t seed,
                      std::uint32_t slot, int threads) {
  const int blocks = (n + threads - 1) / threads;
  DumpPhiloxKernel<<<blocks, threads>>>(out, n, seed, slot);
  qmc::harness::CheckCuda("DumpPhiloxKernel");
}

}  // namespace qmc::endgame
