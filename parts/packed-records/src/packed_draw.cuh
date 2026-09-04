#ifndef PARTS_PACKED_RECORDS_SRC_PACKED_DRAW_CUH_
#define PARTS_PACKED_RECORDS_SRC_PACKED_DRAW_CUH_

#include "parts/endgame/src/philox_draw.cuh"
#include "parts/packed-records/src/sampler.h"

namespace qmc::packed {

// One aligned 64-bit load. Both fields arrive together, so the predicated
// second load K7 emitted for `alias` does not exist here.
__device__ inline int DrawPackedBin(const AliasDraw* __restrict__ draw, int k,
                                    float u1, float u2) {
  int j = static_cast<int>(u1 * static_cast<float>(k));
  if (j < 0) {
    j = 0;
  }
  if (j >= k) {
    j = k - 1;
  }
  const AliasDraw rec = draw[j];
  return (u2 < rec.prob) ? j : static_cast<int>(rec.alias);
}

__device__ inline float SampleInteriorUniform(const BinUniform& rec, float u) {
  if (rec.width <= 0.0f) {
    return rec.lo;
  }
  return rec.lo + u * rec.width;
}

// Byte-for-byte the expression qmc::alias::SampleInterior(rec, u, true) uses.
// Keeping the statement order identical is what lets packed_linear be checked
// bitwise against K7 instead of only statistically.
__device__ inline float SampleInteriorLinear(const BinLinear& rec, float u) {
  if (rec.width <= 0.0f) {
    return rec.lo;
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

__device__ inline float SampleInteriorOf(const BinUniform& rec, float u) {
  return SampleInteriorUniform(rec, u);
}
__device__ inline float SampleInteriorOf(const BinLinear& rec, float u) {
  return SampleInteriorLinear(rec, u);
}

// Same seven uniforms, same slots, same counter scheme as K7. u[7] is still
// the wasted lane of the second Philox call.
template <class BinT>
__device__ inline qmc::alias::DrawnSample DrawFromSeven(
    const AliasDraw* __restrict__ radial_draw,
    const BinT* __restrict__ radial_bin, int n_radial,
    const AliasDraw* __restrict__ theta_draw,
    const BinT* __restrict__ theta_bin, int n_theta, const float* u, int n,
    int l, int m_abs, float radial_norm, float y_norm) {
  const int j_r = DrawPackedBin(radial_draw, n_radial, u[0], u[1]);
  const int j_th = DrawPackedBin(theta_draw, n_theta, u[3], u[4]);
  const BinT rec_r = radial_bin[j_r];
  const BinT rec_th = theta_bin[j_th];
  const float radius = SampleInteriorOf(rec_r, u[2]);
  const float th = SampleInteriorOf(rec_th, u[5]);
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

template <class BinT>
__device__ inline qmc::alias::DrawnSample DrawPackedSample(
    std::uint32_t sample_index, std::uint64_t seed,
    const AliasDraw* __restrict__ radial_draw,
    const BinT* __restrict__ radial_bin, int n_radial,
    const AliasDraw* __restrict__ theta_draw,
    const BinT* __restrict__ theta_bin, int n_theta, int n, int l, int m_abs,
    float radial_norm, float y_norm) {
  float u8[8];
  qmc::endgame::FillPhiloxUniforms(sample_index, seed, u8);
  return DrawFromSeven<BinT>(radial_draw, radial_bin, n_radial, theta_draw,
                             theta_bin, n_theta, u8, n, l, m_abs, radial_norm,
                             y_norm);
}

}  // namespace qmc::packed

#endif
