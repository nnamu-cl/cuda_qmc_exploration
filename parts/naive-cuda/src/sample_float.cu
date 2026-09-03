#include "parts/naive-cuda/src/device_math.cuh"
#include "parts/naive-cuda/src/sampler.h"

#include "harness/timing.h"

namespace qmc::naive {
namespace {

constexpr int kThreads = 256;

__global__ void SampleFp32Kernel(
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
  DrawOne<float, MathMode::Ieee>(
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

__global__ void EvalRadialKernelF32(const int* n, const int* l, const float* r,
                                    const float* norm, float* out, int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    out[i] = RadialRnl(n[i], l[i], r[i], norm[i]);
  }
}

__global__ void EvalLegendreKernelF32(const int* l, const int* m,
                                      const float* x, float* out, int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    out[i] = AssociatedLegendre(l[i], m[i], x[i]);
  }
}

int Blocks(int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

void LaunchSampleFp32(const float* r_nodes, const float* r_cdf, int n_radial,
                      const float* theta_nodes, const float* theta_cdf,
                      int n_theta, void* states, float* x, float* y, float* z,
                      float* density, float* r, float* theta, float* phi, int n,
                      int principal, int l, int m_abs, float radial_norm,
                      float y_norm) {
  SampleFp32Kernel<<<Blocks(n), kThreads>>>(
      r_nodes, r_cdf, n_radial, theta_nodes, theta_cdf, n_theta,
      static_cast<curandStateXORWOW_t*>(states), x, y, z, density, r, theta,
      phi, n, principal, l, m_abs, radial_norm, y_norm);
  qmc::harness::CheckCuda("SampleFp32Kernel");
}

void EvaluateRadialGpuF32(const int* n, const int* l, const float* r,
                          const float* norm, float* out, int count) {
  EvalRadialKernelF32<<<Blocks(count), kThreads>>>(n, l, r, norm, out, count);
  qmc::harness::CheckCuda("EvalRadialKernelF32");
}

void EvaluateLegendreGpuF32(const int* l, const int* m, const float* x,
                            float* out, int count) {
  EvalLegendreKernelF32<<<Blocks(count), kThreads>>>(l, m, x, out, count);
  qmc::harness::CheckCuda("EvalLegendreKernelF32");
}

}  // namespace qmc::naive
