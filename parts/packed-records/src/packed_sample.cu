#include "parts/packed-records/src/packed_draw.cuh"
#include "parts/packed-records/src/sampler.h"

#include <algorithm>

#include "harness/timing.h"

namespace qmc::packed {
namespace {

// K9 core: one thread, one sample, four aligned gathers.
template <class BinT>
__global__ void SamplePackedKernel(
    const AliasDraw* __restrict__ radial_draw,
    const BinT* __restrict__ radial_bin, int n_radial,
    const AliasDraw* __restrict__ theta_draw, const BinT* __restrict__ theta_bin,
    int n_theta, qmc::alias::PackedXyzw* __restrict__ packed,
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
    float* __restrict__ density, float* __restrict__ r,
    float* __restrict__ theta_out, float* __restrict__ phi, int n,
    int principal, int l, int m_abs, float radial_norm, float y_norm,
    std::uint64_t seed) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  const qmc::alias::DrawnSample s = DrawPackedSample<BinT>(
      static_cast<std::uint32_t>(i), seed, radial_draw, radial_bin, n_radial,
      theta_draw, theta_bin, n_theta, principal, l, m_abs, radial_norm, y_norm);
  qmc::endgame::StoreSample(s, i, packed, x, y, z, density, r, theta_out, phi);
}

// Draw arrays staged in shared memory. Grid-stride so the staging cost is paid
// once per block and not once per sample; the counter is still the sample
// index, never the thread id.
__global__ void SamplePackedSharedKernel(
    const AliasDraw* __restrict__ radial_draw,
    const BinUniform* __restrict__ radial_bin, int n_radial,
    const AliasDraw* __restrict__ theta_draw,
    const BinUniform* __restrict__ theta_bin, int n_theta,
    qmc::alias::PackedXyzw* __restrict__ packed, float* __restrict__ x,
    float* __restrict__ y, float* __restrict__ z, float* __restrict__ density,
    float* __restrict__ r, float* __restrict__ theta_out,
    float* __restrict__ phi, int n, int principal, int l, int m_abs,
    float radial_norm, float y_norm, std::uint64_t seed) {
  extern __shared__ __align__(8) unsigned char smem_raw[];
  AliasDraw* s_radial = reinterpret_cast<AliasDraw*>(smem_raw);
  AliasDraw* s_theta = s_radial + n_radial;
  const int total = n_radial + n_theta;
  for (int t = static_cast<int>(threadIdx.x); t < total;
       t += static_cast<int>(blockDim.x)) {
    s_radial[t] = (t < n_radial) ? radial_draw[t] : theta_draw[t - n_radial];
  }
  __syncthreads();
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int stride = static_cast<int>(blockDim.x * gridDim.x);
  for (int i = tid; i < n; i += stride) {
    const qmc::alias::DrawnSample s = DrawPackedSample<BinUniform>(
        static_cast<std::uint32_t>(i), seed, s_radial, radial_bin, n_radial,
        s_theta, theta_bin, n_theta, principal, l, m_abs, radial_norm, y_norm);
    qmc::endgame::StoreSample(s, i, packed, x, y, z, density, r, theta_out,
                              phi);
  }
}

// Grid-stride global control. kSamples == 1 is the honest control for the
// shared kernel (same grid shape, same loop); kSamples > 1 is the ILP retest.
template <int kSamples>
__global__ void SamplePackedIlpKernel(
    const AliasDraw* __restrict__ radial_draw,
    const BinUniform* __restrict__ radial_bin, int n_radial,
    const AliasDraw* __restrict__ theta_draw,
    const BinUniform* __restrict__ theta_bin, int n_theta,
    qmc::alias::PackedXyzw* __restrict__ packed, float* __restrict__ x,
    float* __restrict__ y, float* __restrict__ z, float* __restrict__ density,
    float* __restrict__ r, float* __restrict__ theta_out,
    float* __restrict__ phi, int n, int principal, int l, int m_abs,
    float radial_norm, float y_norm, std::uint64_t seed) {
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int stride = static_cast<int>(blockDim.x * gridDim.x);
  for (int base = tid; base < n; base += stride * kSamples) {
#pragma unroll
    for (int s = 0; s < kSamples; ++s) {
      const int i = base + s * stride;
      if (i < n) {
        const qmc::alias::DrawnSample sample = DrawPackedSample<BinUniform>(
            static_cast<std::uint32_t>(i), seed, radial_draw, radial_bin,
            n_radial, theta_draw, theta_bin, n_theta, principal, l, m_abs,
            radial_norm, y_norm);
        qmc::endgame::StoreSample(sample, i, packed, x, y, z, density, r,
                                  theta_out, phi);
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
int IlpOccupancyFor(int threads) {
  int blocks = 0;
  qmc::harness::CheckCudaCall(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, SamplePackedIlpKernel<kSamples>, threads, 0),
      "ilp occupancy");
  return blocks;
}

template <int kSamples>
void LaunchIlpS(const AliasDraw* radial_draw, const BinUniform* radial_bin,
                int n_radial, const AliasDraw* theta_draw,
                const BinUniform* theta_bin, int n_theta, const Outputs& out,
                const Params& params, const LaunchConfig& cfg) {
  int blocks = cfg.grid_blocks;
  if (blocks <= 0) {
    blocks = MultiprocessorCount() * std::max(IlpOccupancyFor<kSamples>(cfg.threads), 1);
  }
  SamplePackedIlpKernel<kSamples><<<blocks, cfg.threads>>>(
      radial_draw, radial_bin, n_radial, theta_draw, theta_bin, n_theta,
      out.packed, out.x, out.y, out.z, out.density, out.r, out.theta, out.phi,
      params.n, params.principal, params.l, params.m_abs, params.radial_norm,
      params.y_norm, params.seed);
  qmc::harness::CheckCuda("SamplePackedIlpKernel");
}

}  // namespace

int SharedOccupancyBlocks(int threads, std::size_t shared_bytes) {
  int blocks = 0;
  qmc::harness::CheckCudaCall(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, SamplePackedSharedKernel, threads,
          static_cast<int>(shared_bytes)),
      "shared occupancy");
  return blocks;
}

int IlpOccupancyBlocks(int samples_per_thread, int threads) {
  switch (samples_per_thread) {
    case 2:
      return IlpOccupancyFor<2>(threads);
    case 4:
      return IlpOccupancyFor<4>(threads);
    case 8:
      return IlpOccupancyFor<8>(threads);
    default:
      return IlpOccupancyFor<1>(threads);
  }
}

int GridBlocksFor(KernelKind kind, const LaunchConfig& cfg,
                  std::size_t shared_bytes) {
  if (cfg.grid_blocks > 0) {
    return cfg.grid_blocks;
  }
  if (kind == KernelKind::PackedShared) {
    return MultiprocessorCount() *
           std::max(SharedOccupancyBlocks(cfg.threads, shared_bytes), 1);
  }
  return MultiprocessorCount() *
         std::max(IlpOccupancyBlocks(cfg.samples_per_thread, cfg.threads), 1);
}

void LaunchPacked(const AliasDraw* radial_draw, const BinUniform* radial_bin,
                  int n_radial, const AliasDraw* theta_draw,
                  const BinUniform* theta_bin, int n_theta, const Outputs& out,
                  const Params& params, const LaunchConfig& cfg) {
  const int blocks = (params.n + cfg.threads - 1) / cfg.threads;
  SamplePackedKernel<BinUniform><<<blocks, cfg.threads>>>(
      radial_draw, radial_bin, n_radial, theta_draw, theta_bin, n_theta,
      out.packed, out.x, out.y, out.z, out.density, out.r, out.theta, out.phi,
      params.n, params.principal, params.l, params.m_abs, params.radial_norm,
      params.y_norm, params.seed);
  qmc::harness::CheckCuda("SamplePackedKernel<BinUniform>");
}

void LaunchPackedLinear(const AliasDraw* radial_draw,
                        const BinLinear* radial_bin, int n_radial,
                        const AliasDraw* theta_draw, const BinLinear* theta_bin,
                        int n_theta, const Outputs& out, const Params& params,
                        const LaunchConfig& cfg) {
  const int blocks = (params.n + cfg.threads - 1) / cfg.threads;
  SamplePackedKernel<BinLinear><<<blocks, cfg.threads>>>(
      radial_draw, radial_bin, n_radial, theta_draw, theta_bin, n_theta,
      out.packed, out.x, out.y, out.z, out.density, out.r, out.theta, out.phi,
      params.n, params.principal, params.l, params.m_abs, params.radial_norm,
      params.y_norm, params.seed);
  qmc::harness::CheckCuda("SamplePackedKernel<BinLinear>");
}

void LaunchPackedShared(const AliasDraw* radial_draw,
                        const BinUniform* radial_bin, int n_radial,
                        const AliasDraw* theta_draw, const BinUniform* theta_bin,
                        int n_theta, const Outputs& out, const Params& params,
                        const LaunchConfig& cfg) {
  const std::size_t shared_bytes = SharedBytesFor(n_radial, n_theta);
  const int blocks = GridBlocksFor(KernelKind::PackedShared, cfg, shared_bytes);
  SamplePackedSharedKernel<<<blocks, cfg.threads, shared_bytes>>>(
      radial_draw, radial_bin, n_radial, theta_draw, theta_bin, n_theta,
      out.packed, out.x, out.y, out.z, out.density, out.r, out.theta, out.phi,
      params.n, params.principal, params.l, params.m_abs, params.radial_norm,
      params.y_norm, params.seed);
  qmc::harness::CheckCuda("SamplePackedSharedKernel");
}

void LaunchPackedIlp(const AliasDraw* radial_draw, const BinUniform* radial_bin,
                     int n_radial, const AliasDraw* theta_draw,
                     const BinUniform* theta_bin, int n_theta,
                     const Outputs& out, const Params& params,
                     const LaunchConfig& cfg) {
  switch (cfg.samples_per_thread) {
    case 2:
      LaunchIlpS<2>(radial_draw, radial_bin, n_radial, theta_draw, theta_bin,
                    n_theta, out, params, cfg);
      return;
    case 4:
      LaunchIlpS<4>(radial_draw, radial_bin, n_radial, theta_draw, theta_bin,
                    n_theta, out, params, cfg);
      return;
    case 8:
      LaunchIlpS<8>(radial_draw, radial_bin, n_radial, theta_draw, theta_bin,
                    n_theta, out, params, cfg);
      return;
    default:
      LaunchIlpS<1>(radial_draw, radial_bin, n_radial, theta_draw, theta_bin,
                    n_theta, out, params, cfg);
      return;
  }
}

}  // namespace qmc::packed
