#include "parts/inverse-table/src/sampler.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "cpu-reference/hydrogen.h"
#include "harness/env.h"
#include "harness/stats/chi2.h"

namespace qmc::inverse {
namespace {

double Chi2Limit(int dof) {
  return static_cast<double>(dof) + 5.0 * std::sqrt(2.0 * dof);
}

double RadialChi2Value(const GpuDraw& draw,
                       const qmc::cpu::HydrogenSampler& sampler, int bins,
                       double r_lo, double r_hi) {
  const int n = static_cast<int>(draw.r.size());
  const double width = r_hi - r_lo;
  std::vector<double> obs(static_cast<size_t>(bins), 0.0);
  std::vector<double> exp(static_cast<size_t>(bins), 0.0);
  for (double r : draw.r) {
    if (r < r_lo || r >= r_hi) {
      continue;
    }
    int bin = static_cast<int>((r - r_lo) / width * bins);
    bin = std::clamp(bin, 0, bins - 1);
    obs[static_cast<size_t>(bin)] += 1.0;
  }
  for (int i = 0; i < bins; ++i) {
    const double lo = sampler.RadialCdf(r_lo + width * i / bins);
    const double hi = sampler.RadialCdf(r_lo + width * (i + 1) / bins);
    exp[static_cast<size_t>(i)] = static_cast<double>(n) * (hi - lo);
  }
  return qmc::stats::ChiSquare(obs, exp, Chi2Limit(bins - 1)).chi2;
}

}  // namespace

int RunNodeTrap(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  const qmc::cpu::QuantumNumbers qn{3, 1, -1};
  const qmc::cpu::HydrogenSampler sampler(qn);
  const int ks[] = {256, 1024, 4096, 16384, 65536};
  nlohmann::json rows = nlohmann::json::array();
  for (int k : ks) {
    LaunchConfig cfg;
    cfg.table_k_radial = k;
    cfg.table_k_theta = std::max(k / 2, 2);
    for (KernelKind kind :
         {KernelKind::InverseTable, KernelKind::InverseNodes}) {
      GpuDraw draw;
      DrawGpu(kind, qn, 0xC0FFEEULL, n, &draw, cfg);
      const double chi_all =
          RadialChi2Value(draw, sampler, 256, 0.0, sampler.RadiusMax());
      const double chi_node = RadialChi2Value(draw, sampler, 40, 4.5, 7.5);
      const nlohmann::json row = {
          {"kernel", KernelName(kind)},
          {"k_radial", k},
          {"k_theta", cfg.table_k_theta},
          {"chi2_radial_256", chi_all},
          {"chi2_node_window", chi_node},
      };
      rows.push_back(row);
      fmt::print("{} K={} chi2_r={:.1f} chi2_node={:.1f}\n", KernelName(kind),
                 k, chi_all, chi_node);
    }
  }
  if (!out_path.empty()) {
    nlohmann::json body = {
        {"part", "inverse-table"},
        {"experiment", "node_trap_k_sweep"},
        {"n_samples", n},
        {"quantum_numbers", {qn.n, qn.l, qn.m}},
        {"env",
         {{"gpu", qmc::harness::CaptureEnv().gpu},
          {"clocks_sm_mhz", qmc::harness::CaptureEnv().clocks_sm_mhz}}},
        {"rows", rows},
    };
    std::ofstream out(out_path);
    if (!out) {
      fmt::print(stderr, "could not write {}\n", out_path);
      return 1;
    }
    out << body.dump(2) << '\n';
  }
  return 0;
}

}  // namespace qmc::inverse
