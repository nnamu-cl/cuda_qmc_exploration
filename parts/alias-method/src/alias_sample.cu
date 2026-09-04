#include "parts/alias-method/src/alias_draw.cuh"
#include "parts/alias-method/src/sampler.h"

#include "harness/timing.h"

namespace qmc::alias {
namespace {

template <bool Linear, bool Packed>
__global__ void SampleAliasKernel(const AliasBin* radial, int n_radial,
                                  const AliasBin* theta, int n_theta,
                                  curandStateXORWOW_t* states, float* x,
                                  float* y, float* z, float* density, float* r,
                                  float* theta_out, float* phi,
                                  PackedXyzw* packed,
                                  int n, int principal, int l, int m_abs,
                                  float radial_norm, float y_norm) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  curandStateXORWOW_t rng = states[i];
  const DrawnSample s =
      DrawAliasSample<Linear>(&rng, radial, n_radial, theta, n_theta, principal,
                              l, m_abs, radial_norm, y_norm);
  states[i] = rng;
  if constexpr (Packed) {
    packed[i] = PackedXyzw{s.x, s.y, s.z, s.density};
  } else {
    x[i] = s.x;
    y[i] = s.y;
    z[i] = s.z;
    density[i] = s.density;
  }
  if (r != nullptr) {
    r[i] = s.r;
    theta_out[i] = s.theta;
    phi[i] = s.phi;
  }
}

}  // namespace

void LaunchAliasSample(const AliasBin* radial, int n_radial,
                       const AliasBin* theta, int n_theta, void* states,
                       float* x, float* y, float* z, float* density, float* r,
                       float* theta_out, float* phi, void* packed, int n,
                       int principal, int l, int m_abs, float radial_norm,
                       float y_norm, bool linear, int threads) {
  const int blocks = (n + threads - 1) / threads;
  auto* xorwow = static_cast<curandStateXORWOW_t*>(states);
  auto* packed4 = static_cast<PackedXyzw*>(packed);
  if (packed4 != nullptr) {
    if (linear) {
      SampleAliasKernel<true, true><<<blocks, threads>>>(
          radial, n_radial, theta, n_theta, xorwow, x, y, z, density, r,
          theta_out, phi, packed4, n, principal, l, m_abs, radial_norm, y_norm);
    } else {
      SampleAliasKernel<false, true><<<blocks, threads>>>(
          radial, n_radial, theta, n_theta, xorwow, x, y, z, density, r,
          theta_out, phi, packed4, n, principal, l, m_abs, radial_norm, y_norm);
    }
  } else if (linear) {
    SampleAliasKernel<true, false><<<blocks, threads>>>(
        radial, n_radial, theta, n_theta, xorwow, x, y, z, density, r,
        theta_out, phi, packed4, n, principal, l, m_abs, radial_norm, y_norm);
  } else {
    SampleAliasKernel<false, false><<<blocks, threads>>>(
        radial, n_radial, theta, n_theta, xorwow, x, y, z, density, r,
        theta_out, phi, packed4, n, principal, l, m_abs, radial_norm, y_norm);
  }
  qmc::harness::CheckCuda("SampleAliasKernel");
}

}  // namespace qmc::alias
