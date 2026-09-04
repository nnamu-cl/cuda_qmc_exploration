#ifndef PARTS_ENDGAME_SRC_PHILOX_DRAW_CUH_
#define PARTS_ENDGAME_SRC_PHILOX_DRAW_CUH_

#include "harness/rng/philox.h"
#include "parts/alias-method/src/alias_draw.cuh"

namespace qmc::endgame {

__device__ inline int DrawAliasBinUniforms(const qmc::alias::AliasBin* bins,
                                           int k, float u1, float u2) {
  int j = static_cast<int>(u1 * static_cast<float>(k));
  if (j < 0) {
    j = 0;
  }
  if (j >= k) {
    j = k - 1;
  }
  const qmc::alias::AliasBin rec = bins[j];
  return (u2 < rec.prob) ? j : rec.alias;
}

__device__ inline void FillPhiloxUniforms(std::uint32_t sample_index,
                                          std::uint64_t seed, float u[8]) {
  const qmc::rng::Philox4x32Key key = qmc::rng::SeedToKey(seed);
  const qmc::rng::Philox4x32Ctr a = qmc::rng::Philox4x32TenRounds(
      qmc::rng::MakeCounter(sample_index, 0), key);
  const qmc::rng::Philox4x32Ctr b = qmc::rng::Philox4x32TenRounds(
      qmc::rng::MakeCounter(sample_index, 1), key);
  u[0] = qmc::rng::Uint32ToUnitFloat(a.v[0]);
  u[1] = qmc::rng::Uint32ToUnitFloat(a.v[1]);
  u[2] = qmc::rng::Uint32ToUnitFloat(a.v[2]);
  u[3] = qmc::rng::Uint32ToUnitFloat(a.v[3]);
  u[4] = qmc::rng::Uint32ToUnitFloat(b.v[0]);
  u[5] = qmc::rng::Uint32ToUnitFloat(b.v[1]);
  u[6] = qmc::rng::Uint32ToUnitFloat(b.v[2]);
  u[7] = qmc::rng::Uint32ToUnitFloat(b.v[3]);
}

__device__ inline qmc::alias::DrawnSample DrawAliasFromSeven(
    const qmc::alias::AliasBin* radial, int n_radial,
    const qmc::alias::AliasBin* theta, int n_theta, const float* u, int n,
    int l, int m_abs, float radial_norm, float y_norm) {
  const int j_r = DrawAliasBinUniforms(radial, n_radial, u[0], u[1]);
  const int j_th = DrawAliasBinUniforms(theta, n_theta, u[3], u[4]);
  const qmc::alias::AliasBin rec_r = radial[j_r];
  const qmc::alias::AliasBin rec_th = theta[j_th];
  const float radius = qmc::alias::SampleInterior(rec_r, u[2], true);
  const float th = qmc::alias::SampleInterior(rec_th, u[5], true);
  const float ph = 2.0f * static_cast<float>(qmc::naive::kPi) * u[6];
  float sin_th = 0.0f;
  float cos_th = 0.0f;
  float sin_ph = 0.0f;
  float cos_ph = 0.0f;
  qmc::naive::DeviceSinCos(th, &sin_th, &cos_th);
  qmc::naive::DeviceSinCos(ph, &sin_ph, &cos_ph);
  qmc::alias::DrawnSample sample;
  sample.r = radius;
  sample.theta = th;
  sample.phi = ph;
  sample.x = radius * sin_th * cos_ph;
  sample.y = radius * sin_th * sin_ph;
  sample.z = radius * cos_th;
  sample.density = qmc::naive::WavefunctionDensity(n, l, m_abs, radius, th,
                                                   radial_norm, y_norm);
  return sample;
}

__device__ inline qmc::alias::DrawnSample DrawPhiloxSample(
    std::uint32_t sample_index, std::uint64_t seed,
    const qmc::alias::AliasBin* radial, int n_radial,
    const qmc::alias::AliasBin* theta, int n_theta, int n, int l, int m_abs,
    float radial_norm, float y_norm) {
  float u8[8];
  FillPhiloxUniforms(sample_index, seed, u8);
  return DrawAliasFromSeven(radial, n_radial, theta, n_theta, u8, n, l, m_abs,
                            radial_norm, y_norm);
}

__device__ inline void StoreSample(const qmc::alias::DrawnSample& s, int i,
                                   qmc::alias::PackedXyzw* packed, float* x,
                                   float* y, float* z, float* density, float* r,
                                   float* theta_out, float* phi) {
  if (packed != nullptr) {
    packed[i] = qmc::alias::PackedXyzw{s.x, s.y, s.z, s.density};
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

}  // namespace qmc::endgame

#endif
