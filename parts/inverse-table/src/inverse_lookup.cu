#include "parts/inverse-table/src/lookup.cuh"
#include "parts/inverse-table/src/sampler.h"

#include "harness/timing.h"

namespace qmc::inverse {
namespace {

template <bool Snap>
__global__ void SampleInverseKernel(
    const float* r_table, int k_radial, const float* theta_table, int k_theta,
    const std::uint8_t* r_flags, const std::uint8_t* theta_flags,
    curandStateXORWOW_t* states, float* x, float* y, float* z, float* density,
    float* r, float* theta, float* phi, int n, int principal, int l, int m_abs,
    float radial_norm, float y_norm) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  curandStateXORWOW_t rng = states[i];
  float xi = 0.0f;
  float yi = 0.0f;
  float zi = 0.0f;
  float wi = 0.0f;
  float ri = 0.0f;
  float thi = 0.0f;
  float phi_i = 0.0f;
  DrawQuantile<Snap>(&rng, r_table, k_radial, theta_table, k_theta, r_flags,
                     theta_flags, principal, l, m_abs, radial_norm, y_norm,
                     &xi, &yi, &zi, &wi, &ri, &thi, &phi_i);
  states[i] = rng;
  x[i] = xi;
  y[i] = yi;
  z[i] = zi;
  density[i] = wi;
  if (r != nullptr) {
    r[i] = ri;
    theta[i] = thi;
    phi[i] = phi_i;
  }
}

}  // namespace

void LaunchInverseLookup(const float* r_table, int k_radial,
                         const float* theta_table, int k_theta,
                         const std::uint8_t* r_flags,
                         const std::uint8_t* theta_flags, void* states,
                         float* x, float* y, float* z, float* density, float* r,
                         float* theta, float* phi, int n, int principal, int l,
                         int m_abs, float radial_norm, float y_norm,
                         bool snap_jumps, int threads) {
  const int blocks = (n + threads - 1) / threads;
  auto* xorwow = static_cast<curandStateXORWOW_t*>(states);
  if (snap_jumps) {
    SampleInverseKernel<true><<<blocks, threads>>>(
        r_table, k_radial, theta_table, k_theta, r_flags, theta_flags, xorwow, x,
        y, z, density, r, theta, phi, n, principal, l, m_abs, radial_norm,
        y_norm);
  } else {
    SampleInverseKernel<false><<<blocks, threads>>>(
        r_table, k_radial, theta_table, k_theta, r_flags, theta_flags, xorwow, x,
        y, z, density, r, theta, phi, n, principal, l, m_abs, radial_norm,
        y_norm);
  }
  qmc::harness::CheckCuda("SampleInverseKernel");
}

}  // namespace qmc::inverse
