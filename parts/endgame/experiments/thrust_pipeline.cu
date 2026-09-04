#include "parts/endgame/src/philox_draw.cuh"
#include "parts/endgame/src/sampler.h"

#include <curand.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdio>
#include <cstdlib>

#include "harness/timing.h"

namespace qmc::endgame {
namespace {

void CheckCurand(curandStatus_t status, const char* what) {
  if (status != CURAND_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuRAND error at %s: %d\n", what,
                 static_cast<int>(status));
    std::abort();
  }
}

struct ThrustAliasFunctor {
  const qmc::alias::AliasBin* radial;
  int n_radial;
  const qmc::alias::AliasBin* theta;
  int n_theta;
  const float* u0;
  const float* u1;
  const float* u2;
  const float* u3;
  const float* u4;
  const float* u5;
  const float* u6;
  qmc::alias::PackedXyzw* packed;
  float* x;
  float* y;
  float* z;
  float* density;
  float* r;
  float* theta_out;
  float* phi;
  int principal;
  int l;
  int m_abs;
  float radial_norm;
  float y_norm;

  __device__ void operator()(int i) const {
    const float u[7] = {u0[i], u1[i], u2[i], u3[i], u4[i], u5[i], u6[i]};
    const qmc::alias::DrawnSample s =
        DrawAliasFromSeven(radial, n_radial, theta, n_theta, u, principal, l,
                           m_abs, radial_norm, y_norm);
    StoreSample(s, i, packed, x, y, z, density, r, theta_out, phi);
  }
};

}  // namespace

void LaunchThrustAlias(const qmc::alias::AliasBin* radial, int n_radial,
                       const qmc::alias::AliasBin* theta, int n_theta,
                       float* uniforms, qmc::alias::PackedXyzw* packed, float* x,
                       float* y, float* z, float* density, float* r,
                       float* theta_out, float* phi, int n, int principal,
                       int l, int m_abs, float radial_norm, float y_norm,
                       std::uint64_t seed) {
  curandGenerator_t gen;
  CheckCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10),
              "create");
  CheckCurand(curandSetPseudoRandomGeneratorSeed(gen, seed), "seed");
  CheckCurand(curandGenerateUniform(gen, uniforms, static_cast<size_t>(n) * 7),
              "generate");
  CheckCurand(curandDestroyGenerator(gen), "destroy");
  const ThrustAliasFunctor fn{
      radial,     n_radial, theta, n_theta, uniforms,
      uniforms + n, uniforms + 2 * n, uniforms + 3 * n, uniforms + 4 * n,
      uniforms + 5 * n, uniforms + 6 * n, packed, x, y, z, density, r,
      theta_out,  phi,      principal, l, m_abs, radial_norm, y_norm};
  thrust::for_each(thrust::cuda::par, thrust::counting_iterator<int>(0),
                   thrust::counting_iterator<int>(n), fn);
  qmc::harness::CheckCuda("thrust::for_each alias");
}

}  // namespace qmc::endgame
