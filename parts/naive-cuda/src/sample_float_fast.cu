#include "parts/naive-cuda/src/device_math.cuh"
#include "parts/naive-cuda/src/sampler.h"

#include "harness/timing.h"

namespace qmc::naive {
namespace {

constexpr int kThreads = 256;

__global__ void SampleFp32FastKernel(
    const float* r_nodes, const float* r_cdf, int n_radial,
    const float* theta_nodes, const float* theta_cdf, int n_theta,
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
  DrawOne<float, MathMode::Fast>(
      &rng, r_nodes, r_cdf, n_radial, theta_nodes, theta_cdf, n_theta, principal,
      l, m_abs, radial_norm, y_norm, &xi, &yi, &zi, &wi, &ri, &thi, &phi_i);
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

int Blocks(int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

void LaunchSampleFp32Fast(const float* r_nodes, const float* r_cdf,
                          int n_radial, const float* theta_nodes,
                          const float* theta_cdf, int n_theta, void* states,
                          float* x, float* y, float* z, float* density,
                          float* r, float* theta, float* phi, int n,
                          int principal, int l, int m_abs, float radial_norm,
                          float y_norm) {
  SampleFp32FastKernel<<<Blocks(n), kThreads>>>(
      r_nodes, r_cdf, n_radial, theta_nodes, theta_cdf, n_theta,
      static_cast<curandStateXORWOW_t*>(states), x, y, z, density, r, theta,
      phi, n, principal, l, m_abs, radial_norm, y_norm);
  qmc::harness::CheckCuda("SampleFp32FastKernel");
}

}  // namespace qmc::naive
