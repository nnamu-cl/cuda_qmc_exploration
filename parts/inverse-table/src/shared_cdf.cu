#include "parts/inverse-table/src/sampler.h"

#include "parts/naive-cuda/src/device_math.cuh"

#include "harness/timing.h"

namespace qmc::inverse {
namespace {

__global__ void SampleSharedCdfKernel(
    const float* r_nodes, const float* r_cdf, int n_radial,
    const float* theta_nodes, const float* theta_cdf, int n_theta,
    curandStateXORWOW_t* states, float* x, float* y, float* z, float* density,
    float* r, float* theta, float* phi, int n, int principal, int l, int m_abs,
    float radial_norm, float y_norm, int cache_nodes) {
  extern __shared__ float smem[];
  float* r_cdf_s = smem;
  float* theta_cdf_s = r_cdf_s + n_radial;
  float* r_nodes_s = theta_cdf_s + n_theta;
  float* theta_nodes_s = r_nodes_s + n_radial;
  for (int i = static_cast<int>(threadIdx.x); i < n_radial; i += blockDim.x) {
    r_cdf_s[i] = r_cdf[i];
    if (cache_nodes) {
      r_nodes_s[i] = r_nodes[i];
    }
  }
  for (int i = static_cast<int>(threadIdx.x); i < n_theta; i += blockDim.x) {
    theta_cdf_s[i] = theta_cdf[i];
    if (cache_nodes) {
      theta_nodes_s[i] = theta_nodes[i];
    }
  }
  __syncthreads();
  const float* r_n = cache_nodes ? r_nodes_s : r_nodes;
  const float* th_n = cache_nodes ? theta_nodes_s : theta_nodes;
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
  qmc::naive::DrawOne<float, qmc::naive::MathMode::Ieee>(
      &rng, r_n, r_cdf_s, n_radial, th_n, theta_cdf_s, n_theta, principal, l,
      m_abs, radial_norm, y_norm, &xi, &yi, &zi, &wi, &ri, &thi, &phi_i);
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

int SharedBytes(int n_radial, int n_theta, bool cache_nodes) {
  const int floats = cache_nodes ? 2 * (n_radial + n_theta) : (n_radial + n_theta);
  return floats * static_cast<int>(sizeof(float));
}

}  // namespace

void LaunchSharedCdf(const float* r_nodes, const float* r_cdf, int n_radial,
                     const float* theta_nodes, const float* theta_cdf,
                     int n_theta, void* states, float* x, float* y, float* z,
                     float* density, float* r, float* theta, float* phi, int n,
                     int principal, int l, int m_abs, float radial_norm,
                     float y_norm, int threads, bool cache_nodes) {
  const int blocks = (n + threads - 1) / threads;
  const int smem = SharedBytes(n_radial, n_theta, cache_nodes);
  qmc::harness::CheckCudaCall(
      cudaFuncSetAttribute(SampleSharedCdfKernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
      "shared cdf max dynamic smem");
  SampleSharedCdfKernel<<<blocks, threads, static_cast<size_t>(smem)>>>(
      r_nodes, r_cdf, n_radial, theta_nodes, theta_cdf, n_theta,
      static_cast<curandStateXORWOW_t*>(states), x, y, z, density, r, theta,
      phi, n, principal, l, m_abs, radial_norm, y_norm, cache_nodes ? 1 : 0);
  qmc::harness::CheckCuda("SampleSharedCdfKernel");
}

int SharedCdfOccupancyBlocks(int threads, bool cache_nodes, int n_radial,
                             int n_theta) {
  const int smem = SharedBytes(n_radial, n_theta, cache_nodes);
  int blocks = 0;
  qmc::harness::CheckCudaCall(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, SampleSharedCdfKernel, threads, static_cast<size_t>(smem)),
      "shared cdf occupancy");
  return blocks;
}

}  // namespace qmc::inverse
