#include "parts/alias-method/src/alias_draw.cuh"
#include "parts/alias-method/src/sampler.h"

#include "harness/timing.h"

namespace qmc::alias {
namespace {

__global__ void SampleSplitCoordsKernel(
    const AliasBin* radial, int n_radial, const AliasBin* theta, int n_theta,
    curandStateXORWOW_t* states, float* r, float* theta_out, float* phi, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  curandStateXORWOW_t rng = states[i];
  const int j_r = DrawAliasBin(&rng, radial, n_radial);
  const float u_r = qmc::naive::Uniform(&rng, 0.0f);
  const int j_th = DrawAliasBin(&rng, theta, n_theta);
  const float u_th = qmc::naive::Uniform(&rng, 0.0f);
  const float u_phi = qmc::naive::Uniform(&rng, 0.0f);
  r[i] = SampleInterior(radial[j_r], u_r, true);
  theta_out[i] = SampleInterior(theta[j_th], u_th, true);
  phi[i] = 2.0f * static_cast<float>(qmc::naive::kPi) * u_phi;
  states[i] = rng;
}

__global__ void WeightSplitKernel(const float* r, const float* theta,
                                  const float* phi, float* x, float* y,
                                  float* z, float* density, int n,
                                  int principal, int l, int m_abs,
                                  float radial_norm, float y_norm) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  const float radius = r[i];
  const float th = theta[i];
  const float ph = phi[i];
  float sin_th = 0.0f;
  float cos_th = 0.0f;
  float sin_ph = 0.0f;
  float cos_ph = 0.0f;
  qmc::naive::DeviceSinCos(th, &sin_th, &cos_th);
  qmc::naive::DeviceSinCos(ph, &sin_ph, &cos_ph);
  x[i] = radius * sin_th * cos_ph;
  y[i] = radius * sin_th * sin_ph;
  z[i] = radius * cos_th;
  density[i] = qmc::naive::WavefunctionDensity(principal, l, m_abs, radius, th,
                                               radial_norm, y_norm);
}

}  // namespace

void LaunchAliasSplit(const AliasBin* radial, int n_radial,
                      const AliasBin* theta, int n_theta, void* states,
                      float* x, float* y, float* z, float* density, float* r,
                      float* theta_out, float* phi, int n, int principal,
                      int l, int m_abs, float radial_norm, float y_norm,
                      int threads) {
  const int blocks = (n + threads - 1) / threads;
  auto* xorwow = static_cast<curandStateXORWOW_t*>(states);
  SampleSplitCoordsKernel<<<blocks, threads>>>(radial, n_radial, theta, n_theta,
                                               xorwow, r, theta_out, phi, n);
  qmc::harness::CheckCuda("SampleSplitCoordsKernel");
  WeightSplitKernel<<<blocks, threads>>>(r, theta_out, phi, x, y, z, density, n,
                                         principal, l, m_abs, radial_norm,
                                         y_norm);
  qmc::harness::CheckCuda("WeightSplitKernel");
}

}  // namespace qmc::alias
