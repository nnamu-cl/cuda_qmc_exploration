#ifndef CPU_REFERENCE_SPECIAL_H_
#define CPU_REFERENCE_SPECIAL_H_

#include <cmath>
#include <numbers>

namespace qmc::cpu {

inline constexpr double kBohrRadius = 1.0;

[[nodiscard]] inline double Factorial(int n) {
  double value = 1.0;
  for (int i = 2; i <= n; ++i) {
    value *= static_cast<double>(i);
  }
  return value;
}

[[nodiscard]] inline double AssociatedLaguerre(int k, int alpha, double x) {
  if (k <= 0) {
    return 1.0;
  }
  double lm2 = 1.0;
  double lm1 = 1.0 + static_cast<double>(alpha) - x;
  if (k == 1) {
    return lm1;
  }
  double L = lm1;
  for (int j = 2; j <= k; ++j) {
    L = ((2.0 * j - 1.0 + alpha - x) * lm1 - (j - 1.0 + alpha) * lm2) /
        static_cast<double>(j);
    lm2 = lm1;
    lm1 = L;
  }
  return L;
}

[[nodiscard]] inline double AssociatedLegendrePositiveM(int l, int m,
                                                        double x) {
  double pmm = 1.0;
  if (m > 0) {
    const double somx2 = std::sqrt(std::max(0.0, (1.0 - x) * (1.0 + x)));
    double fact = 1.0;
    for (int j = 1; j <= m; ++j) {
      pmm *= -fact * somx2;
      fact += 2.0;
    }
  }
  if (l == m) {
    return pmm;
  }
  double pm1m = x * (2.0 * m + 1.0) * pmm;
  if (l == m + 1) {
    return pm1m;
  }
  double pll = pm1m;
  for (int ll = m + 2; ll <= l; ++ll) {
    pll = ((2.0 * ll - 1.0) * x * pm1m - (ll + m - 1.0) * pmm) /
          static_cast<double>(ll - m);
    pmm = pm1m;
    pm1m = pll;
  }
  return pm1m;
}

[[nodiscard]] inline double AssociatedLegendre(int l, int m, double x) {
  const int m_abs = m < 0 ? -m : m;
  const double plm = AssociatedLegendrePositiveM(l, m_abs, x);
  if (m >= 0) {
    return plm;
  }
  const double phase = (m_abs % 2 == 0) ? 1.0 : -1.0;
  return phase * Factorial(l - m_abs) / Factorial(l + m_abs) * plm;
}

[[nodiscard]] inline double RadialNorm(int n, int l, double a0 = kBohrRadius) {
  return std::sqrt(std::pow(2.0 / (static_cast<double>(n) * a0), 3.0) *
                   Factorial(n - l - 1) / (2.0 * n * Factorial(n + l)));
}

[[nodiscard]] inline double SphericalHarmonicNorm(int l, int m) {
  const int m_abs = m < 0 ? -m : m;
  return ((2.0 * l + 1.0) / (4.0 * std::numbers::pi)) *
         (Factorial(l - m_abs) / Factorial(l + m_abs));
}

[[nodiscard]] inline double RadialRnl(int n, int l, double r,
                                      double a0 = kBohrRadius) {
  const double rho = 2.0 * r / (static_cast<double>(n) * a0);
  const double laguerre = AssociatedLaguerre(n - l - 1, 2 * l + 1, rho);
  const double rho_l = (l == 0) ? 1.0 : std::pow(rho, static_cast<double>(l));
  return RadialNorm(n, l, a0) * std::exp(-0.5 * rho) * rho_l * laguerre;
}

[[nodiscard]] inline double SphericalHarmonicDensity(int l, int m,
                                                     double theta) {
  const int m_abs = m < 0 ? -m : m;
  const double plm = AssociatedLegendrePositiveM(l, m_abs, std::cos(theta));
  return SphericalHarmonicNorm(l, m) * plm * plm;
}

[[nodiscard]] inline double WavefunctionDensity(int n, int l, int m, double r,
                                                double theta) {
  const double radial = RadialRnl(n, l, r);
  return radial * radial * SphericalHarmonicDensity(l, m, theta);
}

[[nodiscard]] inline double ExactMeanR(int n, int l,
                                       double a0 = kBohrRadius) {
  return 0.5 * a0 * (3.0 * n * n - l * (l + 1));
}

[[nodiscard]] inline double ExactMeanR2(int n, int l,
                                        double a0 = kBohrRadius) {
  return (a0 * a0 * n * n / 2.0) * (5.0 * n * n + 1.0 - 3.0 * l * (l + 1));
}

[[nodiscard]] inline double ExactMeanInvR(int n, double a0 = kBohrRadius) {
  return 1.0 / (static_cast<double>(n) * n * a0);
}

}  // namespace qmc::cpu

#endif
