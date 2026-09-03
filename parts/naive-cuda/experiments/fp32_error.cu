#include "parts/naive-cuda/src/device_math.cuh"
#include "parts/naive-cuda/src/sampler.h"

#include <cmath>
#include <fstream>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "cpu-reference/special.h"

namespace qmc::naive {
namespace {

double RelErr(double got, double want) {
  if (!std::isfinite(got) || !std::isfinite(want)) {
    return 1.0;
  }
  if (std::abs(want) < 1e-18) {
    return std::abs(got);
  }
  return std::abs(got - want) / std::abs(want);
}

}  // namespace

int RunFp32ErrorAnalysis(const std::string& out_path) {
  double max_r = 0.0;
  double sum_r = 0.0;
  int n_r = 0;
  double max_p = 0.0;
  double sum_p = 0.0;
  int n_p = 0;
  double max_pow = 0.0;
  double max_explog = 0.0;

  nlohmann::json radial = nlohmann::json::array();
  for (int n = 1; n <= 6; ++n) {
    for (int l = 0; l < n; ++l) {
      const double r_max = 10.0 * n * n;
      const double norm64 = qmc::cpu::RadialNorm(n, l);
      const float norm32 = static_cast<float>(norm64);
      double row_max = 0.0;
      for (int i = 1; i <= 200; ++i) {
        const double r = r_max * static_cast<double>(i) / 200.0;
        const double want = RadialRnl(n, l, r, norm64);
        const float got = RadialRnl(n, l, static_cast<float>(r), norm32);
        const double err = RelErr(static_cast<double>(got), want);
        row_max = std::max(row_max, err);
        max_r = std::max(max_r, err);
        sum_r += err;
        ++n_r;
        const double rho = 2.0 * r / static_cast<double>(n);
        if (l > 0 && rho > 0.0) {
          const double iterated = IntegerPower(rho, l);
          const double explog = std::exp(static_cast<double>(l) * std::log(rho));
          const double want_pow = std::pow(rho, static_cast<double>(l));
          max_pow = std::max(max_pow, RelErr(iterated, want_pow));
          max_explog = std::max(max_explog, RelErr(explog, want_pow));
        }
      }
      radial.push_back({{"n", n}, {"l", l}, {"max_rel_error", row_max}});
    }
  }

  nlohmann::json legendre = nlohmann::json::array();
  for (int l = 0; l <= 5; ++l) {
    for (int m = -l; m <= l; ++m) {
      double row_max = 0.0;
      for (int i = 0; i <= 200; ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / 200.0;
        const double want = AssociatedLegendre(l, m, x);
        const float got = AssociatedLegendre(l, m, static_cast<float>(x));
        const double err = RelErr(static_cast<double>(got), want);
        row_max = std::max(row_max, err);
        max_p = std::max(max_p, err);
        sum_p += err;
        ++n_p;
      }
      legendre.push_back({{"l", l}, {"m", m}, {"max_rel_error", row_max}});
    }
  }

  const nlohmann::json body = {
      {"radial_max_rel_error", max_r},
      {"radial_mean_rel_error", sum_r / static_cast<double>(n_r)},
      {"legendre_max_rel_error", max_p},
      {"legendre_mean_rel_error", sum_p / static_cast<double>(n_p)},
      {"rho_l_iterated_max_rel_error", max_pow},
      {"rho_l_explog_max_rel_error", max_explog},
      {"monte_carlo_1e9_scale", 3.0e-5},
      {"radial", radial},
      {"legendre", legendre},
  };

  fmt::print(
      "fp32 error R_nl max={:.3e} mean={:.3e} P_l^m max={:.3e} mean={:.3e} "
      "rho^l iterated={:.3e} explog={:.3e}\n",
      max_r, sum_r / static_cast<double>(n_r), max_p,
      sum_p / static_cast<double>(n_p), max_pow, max_explog);

  if (!out_path.empty()) {
    std::ofstream out(out_path);
    if (!out) {
      fmt::print(stderr, "could not write {}\n", out_path);
      return 1;
    }
    out << body.dump(2) << '\n';
  }
  return 0;
}

}  // namespace qmc::naive
