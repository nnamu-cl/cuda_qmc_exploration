#ifndef PARTS_ALIAS_METHOD_SRC_ALIAS_DRAW_CUH_
#define PARTS_ALIAS_METHOD_SRC_ALIAS_DRAW_CUH_

#include "parts/alias-method/src/sampler.h"
#include "parts/naive-cuda/src/device_math.cuh"

namespace qmc::alias {

__device__ inline int DrawAliasBin(curandStateXORWOW_t* rng, const AliasBin* bins,
                                   int k) {
  const float u1 = qmc::naive::Uniform(rng, 0.0f);
  int j = static_cast<int>(u1 * static_cast<float>(k));
  if (j < 0) {
    j = 0;
  }
  if (j >= k) {
    j = k - 1;
  }
  const AliasBin rec = bins[j];
  const float u2 = qmc::naive::Uniform(rng, 0.0f);
  return (u2 < rec.prob) ? j : rec.alias;
}

__device__ inline float SampleInterior(const AliasBin& rec, float u,
                                       bool linear) {
  if (rec.width <= 0.0f) {
    return rec.lo;
  }
  if (!linear) {
    return rec.lo + u * rec.width;
  }
  const float y0 = rec.y0;
  const float y1 = rec.y1;
  const float sum = y0 + y1;
  if (sum <= 0.0f) {
    return rec.lo;
  }
  const float root = sqrtf((1.0f - u) * y0 * y0 + u * y1 * y1);
  const float t = u * sum / (y0 + root);
  return rec.lo + t * rec.width;
}

struct DrawnSample {
  float x;
  float y;
  float z;
  float density;
  float r;
  float theta;
  float phi;
};

template <bool Linear>
__device__ inline DrawnSample DrawAliasSample(
    curandStateXORWOW_t* rng, const AliasBin* radial, int n_radial,
    const AliasBin* theta, int n_theta, int n, int l, int m_abs,
    float radial_norm, float y_norm) {


  const int j_r = DrawAliasBin(rng, radial, n_radial);
  const float u_r = qmc::naive::Uniform(rng, 0.0f);
  const int j_th = DrawAliasBin(rng, theta, n_theta);
  const float u_th = qmc::naive::Uniform(rng, 0.0f);
  const float u_phi = qmc::naive::Uniform(rng, 0.0f);
  const AliasBin rec_r = radial[j_r];
  const AliasBin rec_th = theta[j_th];
  const float radius = SampleInterior(rec_r, u_r, Linear);
  const float th = SampleInterior(rec_th, u_th, Linear);
  const float ph = 2.0f * static_cast<float>(qmc::naive::kPi) * u_phi;
  float sin_th = 0.0f;
  float cos_th = 0.0f;
  float sin_ph = 0.0f;
  float cos_ph = 0.0f;
  qmc::naive::DeviceSinCos(th, &sin_th, &cos_th);
  qmc::naive::DeviceSinCos(ph, &sin_ph, &cos_ph);
  DrawnSample sample;
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

}  // namespace qmc::alias

#endif
