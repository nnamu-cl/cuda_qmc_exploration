#ifndef PARTS_NAIVE_CUDA_SRC_DEVICE_MATH_CUH_
#define PARTS_NAIVE_CUDA_SRC_DEVICE_MATH_CUH_

#include <math.h>

#include <cuda/std/algorithm>
#include <curand_kernel.h>

namespace qmc::naive {

constexpr double kPi = 3.14159265358979323846;

enum class MathMode { Ieee, Fast };

template <class Real>
__host__ __device__ inline Real InvertCdf(const Real* nodes, const Real* cdf,
                                          int n, Real u) {
  if (u <= cdf[0]) {
    return nodes[0];
  }
  if (u >= cdf[n - 1]) {
    return nodes[n - 1];
  }
  const Real* it = cuda::std::lower_bound(cdf, cdf + n, u);
  const int i = static_cast<int>(it - cdf);
  if (i == 0) {
    return nodes[0];
  }
  const Real c0 = cdf[i - 1];
  const Real c1 = cdf[i];
  const Real t = (c1 > c0) ? (u - c0) / (c1 - c0) : Real(0);
  return nodes[i - 1] + t * (nodes[i] - nodes[i - 1]);
}

template <class Real>
__host__ __device__ inline Real AssociatedLaguerre(int k, int alpha, Real x) {
  if (k <= 0) {
    return Real(1);
  }
  Real lm2 = Real(1);
  Real lm1 = Real(1) + static_cast<Real>(alpha) - x;
  if (k == 1) {
    return lm1;
  }
  Real L = lm1;
  for (int j = 2; j <= k; ++j) {
    L = ((Real(2) * static_cast<Real>(j) - Real(1) + static_cast<Real>(alpha) -
          x) *
             lm1 -
         (static_cast<Real>(j) - Real(1) + static_cast<Real>(alpha)) * lm2) /
        static_cast<Real>(j);
    lm2 = lm1;
    lm1 = L;
  }
  return L;
}

template <class Real>
__host__ __device__ inline Real AssociatedLegendrePositiveM(int l, int m,
                                                            Real x) {
  Real pmm = Real(1);
  if (m > 0) {
    Real somx2 = (Real(1) - x) * (Real(1) + x);
    if (somx2 < Real(0)) {
      somx2 = Real(0);
    }
    somx2 = sqrt(somx2);
    Real fact = Real(1);
    for (int j = 1; j <= m; ++j) {
      pmm *= -fact * somx2;
      fact += Real(2);
    }
  }
  if (l == m) {
    return pmm;
  }
  Real pm1m = x * (Real(2) * static_cast<Real>(m) + Real(1)) * pmm;
  if (l == m + 1) {
    return pm1m;
  }
  Real pll = pm1m;
  for (int ll = m + 2; ll <= l; ++ll) {
    pll = ((Real(2) * static_cast<Real>(ll) - Real(1)) * x * pm1m -
           (static_cast<Real>(ll) + static_cast<Real>(m) - Real(1)) * pmm) /
          static_cast<Real>(ll - m);
    pmm = pm1m;
    pm1m = pll;
  }
  return pm1m;
}

template <class Real>
__host__ __device__ inline Real Factorial(int n) {
  Real value = Real(1);
  for (int i = 2; i <= n; ++i) {
    value *= static_cast<Real>(i);
  }
  return value;
}

template <class Real>
__host__ __device__ inline Real AssociatedLegendre(int l, int m, Real x) {
  const int m_abs = m < 0 ? -m : m;
  const Real plm = AssociatedLegendrePositiveM(l, m_abs, x);
  if (m >= 0) {
    return plm;
  }
  const Real phase = (m_abs % 2 == 0) ? Real(1) : Real(-1);
  return phase * Factorial<Real>(l - m_abs) / Factorial<Real>(l + m_abs) * plm;
}

template <class Real>
__host__ __device__ inline Real IntegerPower(Real base, int p) {
  Real value = Real(1);
  for (int i = 0; i < p; ++i) {
    value *= base;
  }
  return value;
}

__host__ __device__ inline double DeviceExp(double x) { return exp(x); }
__host__ __device__ inline float DeviceExp(float x) { return expf(x); }
__host__ __device__ inline float DeviceExpFast(float x) {
#ifdef __CUDA_ARCH__
  return __expf(x);
#else
  return expf(x);
#endif
}

__host__ __device__ inline void DeviceSinCos(double a, double* s, double* c) {
  sincos(a, s, c);
}
__host__ __device__ inline void DeviceSinCos(float a, float* s, float* c) {
  sincosf(a, s, c);
}
__host__ __device__ inline void DeviceSinCosFast(float a, float* s, float* c) {
#ifdef __CUDA_ARCH__
  __sincosf(a, s, c);
#else
  sincosf(a, s, c);
#endif
}

__host__ __device__ inline double DeviceCos(double x) { return cos(x); }
__host__ __device__ inline float DeviceCos(float x) { return cosf(x); }

template <class Real>
__host__ __device__ inline Real RadialRnl(int n, int l, Real r, Real norm) {
  const Real rho = Real(2) * r / static_cast<Real>(n);
  const Real laguerre = AssociatedLaguerre<Real>(n - l - 1, 2 * l + 1, rho);
  const Real rho_l = IntegerPower(rho, l);
  return norm * DeviceExp(Real(-0.5) * rho) * rho_l * laguerre;
}

template <class Real>
__host__ __device__ inline Real RadialRnlFast(int n, int l, Real r, Real norm) {
  const Real rho = Real(2) * r / static_cast<Real>(n);
  const Real laguerre = AssociatedLaguerre<Real>(n - l - 1, 2 * l + 1, rho);
  const Real rho_l = IntegerPower(rho, l);
  return norm * DeviceExpFast(Real(-0.5) * rho) * rho_l * laguerre;
}

template <class Real>
__host__ __device__ inline Real
WavefunctionDensity(int n, int l, int m_abs, Real r, Real theta,
                    Real radial_norm, Real y_norm) {
  const Real radial = RadialRnl(n, l, r, radial_norm);
  const Real plm = AssociatedLegendrePositiveM(l, m_abs, DeviceCos(theta));
  return radial * radial * y_norm * plm * plm;
}

template <class Real>
__host__ __device__ inline Real
WavefunctionDensityFast(int n, int l, int m_abs, Real r, Real theta,
                        Real radial_norm, Real y_norm) {
  const Real radial = RadialRnlFast(n, l, r, radial_norm);
  const Real plm = AssociatedLegendrePositiveM(l, m_abs, DeviceCos(theta));
  return radial * radial * y_norm * plm * plm;
}

__device__ inline double Uniform(curandStateXORWOW_t* rng, double) {
  return curand_uniform_double(rng);
}
__device__ inline float Uniform(curandStateXORWOW_t* rng, float) {
  return curand_uniform(rng);
}

template <class Real, MathMode mode>
__device__ inline void DrawOne(curandStateXORWOW_t* rng, const Real* r_nodes,
                               const Real* r_cdf, int n_radial,
                               const Real* theta_nodes, const Real* theta_cdf,
                               int n_theta, int n, int l, int m_abs,
                               Real radial_norm, Real y_norm, Real* x, Real* y,
                               Real* z, Real* density, Real* r, Real* theta,
                               Real* phi) {
  const Real u_r = Uniform(rng, Real{});
  const Real u_th = Uniform(rng, Real{});
  const Real u_phi = Uniform(rng, Real{});
  const Real radius = InvertCdf(r_nodes, r_cdf, n_radial, u_r);
  const Real th = InvertCdf(theta_nodes, theta_cdf, n_theta, u_th);
  const Real ph = Real(2) * Real(kPi) * u_phi;
  Real sin_th = Real(0);
  Real cos_th = Real(0);
  Real sin_ph = Real(0);
  Real cos_ph = Real(0);
  if constexpr (mode == MathMode::Fast) {
    DeviceSinCosFast(th, &sin_th, &cos_th);
    DeviceSinCosFast(ph, &sin_ph, &cos_ph);
  } else {
    DeviceSinCos(th, &sin_th, &cos_th);
    DeviceSinCos(ph, &sin_ph, &cos_ph);
  }
  *r = radius;
  *theta = th;
  *phi = ph;
  *x = radius * sin_th * cos_ph;
  *y = radius * sin_th * sin_ph;
  *z = radius * cos_th;
  if constexpr (mode == MathMode::Fast) {
    *density = WavefunctionDensityFast(n, l, m_abs, radius, th, radial_norm,
                                       y_norm);
  } else {
    *density =
        WavefunctionDensity(n, l, m_abs, radius, th, radial_norm, y_norm);
  }
}

}  // namespace qmc::naive

#endif
