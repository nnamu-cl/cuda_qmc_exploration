#include "parts/naive-cuda/src/device_math.cuh"
#include "parts/naive-cuda/src/sampler.h"

#include "harness/timing.h"

namespace qmc::naive {
namespace {

constexpr int kThreads = 256;

__global__ void SampleFp64Kernel(
    const double* r_nodes, const double* r_cdf, int n_radial,
    const double* theta_nodes, const double* theta_cdf, int n_theta,
    curandStateXORWOW_t* states, double* x, double* y, double* z,
    double* density, double* r, double* theta, double* phi, int n,
    int principal, int l, int m_abs, double radial_norm, double y_norm) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  curandStateXORWOW_t rng = states[i];
  double xi = 0.0;
  double yi = 0.0;
  double zi = 0.0;
  double wi = 0.0;
  double ri = 0.0;
  double thi = 0.0;
  double phi_i = 0.0;
  DrawOne<double, MathMode::Ieee>(
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

__global__ void InvertRadialKernel(const double* nodes, const double* cdf,
                                   int n_bins, const double* u, double* r,
                                   int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    r[i] = InvertCdf(nodes, cdf, n_bins, u[i]);
  }
}

__global__ void EvalRadialKernel(const int* n, const int* l, const double* r,
                                 const double* norm, double* out, int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    out[i] = RadialRnl(n[i], l[i], r[i], norm[i]);
  }
}

__global__ void EvalLegendreKernel(const int* l, const int* m, const double* x,
                                   double* out, int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    out[i] = AssociatedLegendre(l[i], m[i], x[i]);
  }
}

int Blocks(int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

void LaunchSampleFp64(const double* r_nodes, const double* r_cdf, int n_radial,
                      const double* theta_nodes, const double* theta_cdf,
                      int n_theta, void* states, double* x, double* y,
                      double* z, double* density, double* r, double* theta,
                      double* phi, int n, int principal, int l, int m_abs,
                      double radial_norm, double y_norm) {
  SampleFp64Kernel<<<Blocks(n), kThreads>>>(
      r_nodes, r_cdf, n_radial, theta_nodes, theta_cdf, n_theta,
      static_cast<curandStateXORWOW_t*>(states), x, y, z, density, r, theta,
      phi, n, principal, l, m_abs, radial_norm, y_norm);
  qmc::harness::CheckCuda("SampleFp64Kernel");
}

void InvertRadialGpu(const double* nodes, const double* cdf, int n_bins,
                     const double* u, double* r, int n) {
  InvertRadialKernel<<<Blocks(n), kThreads>>>(nodes, cdf, n_bins, u, r, n);
  qmc::harness::CheckCuda("InvertRadialKernel");
}

void EvaluateRadialGpu(const int* n, const int* l, const double* r,
                       const double* norm, double* out, int count) {
  EvalRadialKernel<<<Blocks(count), kThreads>>>(n, l, r, norm, out, count);
  qmc::harness::CheckCuda("EvalRadialKernel");
}

void EvaluateLegendreGpu(const int* l, const int* m, const double* x,
                         double* out, int count) {
  EvalLegendreKernel<<<Blocks(count), kThreads>>>(l, m, x, out, count);
  qmc::harness::CheckCuda("EvalLegendreKernel");
}

}  // namespace qmc::naive
