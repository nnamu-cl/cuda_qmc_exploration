#include "cpu-reference/original_buggy.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace qmc::cpu::buggy {
namespace {

constexpr double kA0 = 1.0;

std::vector<double> r_cdf;
bool r_built = false;
std::vector<double> theta_cdf;
bool theta_built = false;
std::mt19937 phi_gen{1};
std::uniform_real_distribution<float> phi_dis{0.0f, 1.0f};

}  // namespace

void ResetTables() {
  r_cdf.clear();
  r_built = false;
  theta_cdf.clear();
  theta_built = false;
  phi_gen.seed(1);
}

double SampleRadius(int n, int l, std::mt19937& gen) {
  const int N = 4096;
  const double r_max = 10.0 * n * n * kA0;

  if (!r_built) {
    r_cdf.resize(static_cast<size_t>(N));
    const double dr = r_max / (N - 1);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      const double r = i * dr;
      const double rho = 2.0 * r / (n * kA0);
      const int k = n - l - 1;
      const int alpha = 2 * l + 1;
      double L = 1.0;
      double lm1 = 1.0 + alpha - rho;
      if (k == 1) {
        L = lm1;
      } else if (k > 1) {
        double lm2 = 1.0;
        for (int j = 2; j <= k; ++j) {
          L = ((2 * j - 1 + alpha - rho) * lm1 - (j - 1 + alpha) * lm2) / j;
          lm2 = lm1;
          lm1 = L;
        }
      }
      const double norm =
          std::pow(2.0 / (n * kA0), 3) * std::tgamma(n - l) /
          (2.0 * n * std::tgamma(n + l + 1));
      const double R = std::sqrt(norm) * std::exp(-rho / 2.0) *
                       std::pow(rho, l) * L;
      const double pdf = r * r * R * R;
      sum += pdf;
      r_cdf[static_cast<size_t>(i)] = sum;
    }
    for (double& v : r_cdf) {
      v /= sum;
    }
    r_built = true;
  }

  std::uniform_real_distribution<double> dis(0.0, 1.0);
  const double u = dis(gen);
  const int idx =
      static_cast<int>(std::lower_bound(r_cdf.begin(), r_cdf.end(), u) -
                       r_cdf.begin());
  return idx * (r_max / (N - 1));
}

double SampleTheta(int l, int m, std::mt19937& gen) {
  const int N = 2048;

  if (!theta_built) {
    theta_cdf.resize(static_cast<size_t>(N));
    const double dtheta = std::numbers::pi / (N - 1);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      const double theta = i * dtheta;
      const double x = std::cos(theta);
      double pmm = 1.0;
      if (m > 0) {
        const double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int j = 1; j <= m; ++j) {
          pmm *= -fact * somx2;
          fact += 2.0;
        }
      }
      double plm = 0.0;
      if (l == m) {
        plm = pmm;
      } else {
        double pm1m = x * (2 * m + 1) * pmm;
        if (l == m + 1) {
          plm = pm1m;
        } else {
          for (int ll = m + 2; ll <= l; ++ll) {
            const double pll = ((2 * ll - 1) * x * pm1m - (ll + m - 1) * pmm) /
                               (ll - m);
            pmm = pm1m;
            pm1m = pll;
          }
          plm = pm1m;
        }
      }
      const double pdf = std::sin(theta) * plm * plm;
      sum += pdf;
      theta_cdf[static_cast<size_t>(i)] = sum;
    }
    for (double& v : theta_cdf) {
      v /= sum;
    }
    theta_built = true;
  }

  std::uniform_real_distribution<double> dis(0.0, 1.0);
  const double u = dis(gen);
  const int idx =
      static_cast<int>(std::lower_bound(theta_cdf.begin(), theta_cdf.end(), u) -
                       theta_cdf.begin());
  return idx * (std::numbers::pi / (N - 1));
}

float SamplePhi() {
  return static_cast<float>(2.0 * std::numbers::pi * phi_dis(phi_gen));
}

}  // namespace qmc::cpu::buggy
