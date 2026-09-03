#ifndef PARTS_INVERSE_TABLE_SRC_LOOKUP_CUH_
#define PARTS_INVERSE_TABLE_SRC_LOOKUP_CUH_

#include <cstdint>

#include "parts/naive-cuda/src/device_math.cuh"

namespace qmc::inverse {

__device__ inline float QuantileLerp(const float* table, int k, float u) {
  const float x = u * static_cast<float>(k - 1);
  int idx = static_cast<int>(x);
  if (idx < 0) {
    idx = 0;
  }
  if (idx > k - 2) {
    idx = k - 2;
  }
  const float frac = x - static_cast<float>(idx);
  return table[idx] + frac * (table[idx + 1] - table[idx]);
}

__device__ inline float QuantileSnap(const float* table, const std::uint8_t* flags,
                                     int k, float u) {
  const float x = u * static_cast<float>(k - 1);
  int idx = static_cast<int>(x);
  if (idx < 0) {
    idx = 0;
  }
  if (idx > k - 2) {
    idx = k - 2;
  }
  const float frac = x - static_cast<float>(idx);
  const float a = table[idx];
  const float b = table[idx + 1];
  if (flags != nullptr && flags[idx] != 0) {
    return (frac < 0.5f) ? a : b;
  }
  return a + frac * (b - a);
}

template <bool Snap>
__device__ inline void DrawQuantile(
    curandStateXORWOW_t* rng, const float* r_table, int k_radial,
    const float* theta_table, int k_theta, const std::uint8_t* r_flags,
    const std::uint8_t* theta_flags, int n, int l, int m_abs, float radial_norm,
    float y_norm, float* x, float* y, float* z, float* density, float* r,
    float* theta, float* phi) {
  const float u_r = qmc::naive::Uniform(rng, 0.0f);
  const float u_th = qmc::naive::Uniform(rng, 0.0f);
  const float u_phi = qmc::naive::Uniform(rng, 0.0f);
  float radius = 0.0f;
  float th = 0.0f;
  if constexpr (Snap) {
    radius = QuantileSnap(r_table, r_flags, k_radial, u_r);
    th = QuantileSnap(theta_table, theta_flags, k_theta, u_th);
  } else {
    radius = QuantileLerp(r_table, k_radial, u_r);
    th = QuantileLerp(theta_table, k_theta, u_th);
  }
  const float ph = 2.0f * static_cast<float>(qmc::naive::kPi) * u_phi;
  float sin_th = 0.0f;
  float cos_th = 0.0f;
  float sin_ph = 0.0f;
  float cos_ph = 0.0f;
  qmc::naive::DeviceSinCos(th, &sin_th, &cos_th);
  qmc::naive::DeviceSinCos(ph, &sin_ph, &cos_ph);
  *r = radius;
  *theta = th;
  *phi = ph;
  *x = radius * sin_th * cos_ph;
  *y = radius * sin_th * sin_ph;
  *z = radius * cos_th;
  *density = qmc::naive::WavefunctionDensity(n, l, m_abs, radius, th,
                                             radial_norm, y_norm);
}

}  // namespace qmc::inverse

#endif
