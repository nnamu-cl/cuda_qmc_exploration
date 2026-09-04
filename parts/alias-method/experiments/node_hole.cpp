#include "parts/alias-method/src/sampler.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "cpu-reference/hydrogen.h"
#include "harness/env.h"
#include "harness/stats/chi2.h"
#include "parts/inverse-table/src/sampler.h"

namespace qmc::alias {
namespace {

double Chi2Limit(int dof) {
  return static_cast<double>(dof) + 5.0 * std::sqrt(2.0 * dof);
}

double RadialChi2Value(const std::vector<double>& r,
                       const qmc::cpu::HydrogenSampler& sampler, int bins,
                       double r_lo, double r_hi) {
  const int n = static_cast<int>(r.size());
  const double width = r_hi - r_lo;
  std::vector<double> obs(static_cast<size_t>(bins), 0.0);
  std::vector<double> exp(static_cast<size_t>(bins), 0.0);
  for (double value : r) {
    if (value < r_lo || value >= r_hi) {
      continue;
    }
    int bin = static_cast<int>((value - r_lo) / width * bins);
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

int RunNodeHole(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  const qmc::cpu::QuantumNumbers qn{3, 1, -1};
  const qmc::cpu::HydrogenSampler sampler(qn);
  nlohmann::json rows = nlohmann::json::array();

  qmc::inverse::GpuDraw inverse;
  qmc::inverse::DrawGpu(qmc::inverse::KernelKind::InverseTable, qn, 0xC0FFEEULL,
                        n, &inverse);
  const double inv_all =
      RadialChi2Value(inverse.r, sampler, 256, 0.0, sampler.RadiusMax());
  const double inv_node = RadialChi2Value(inverse.r, sampler, 40, 4.5, 7.5);
  rows.push_back({
      {"kernel", "inverse_table"},
      {"chi2_radial_256", inv_all},
      {"chi2_node_window", inv_node},
  });
  fmt::print("inverse_table chi2_r={:.1f} chi2_node={:.1f}\n", inv_all, inv_node);

  for (KernelKind kind : {KernelKind::Uniform, KernelKind::Linear}) {
    GpuDraw draw;
    DrawGpu(kind, qn, 0xC0FFEEULL, n, &draw);
    const double chi_all =
        RadialChi2Value(draw.r, sampler, 256, 0.0, sampler.RadiusMax());
    const double chi_node = RadialChi2Value(draw.r, sampler, 40, 4.5, 7.5);
    rows.push_back({
        {"kernel", KernelName(kind)},
        {"chi2_radial_256", chi_all},
        {"chi2_node_window", chi_node},
    });
    fmt::print("{} chi2_r={:.1f} chi2_node={:.1f}\n", KernelName(kind), chi_all,
               chi_node);
  }

  if (!out_path.empty()) {
    const auto env = qmc::harness::CaptureEnv();
    nlohmann::json body = {
        {"part", "alias-method"},
        {"experiment", "node_hole"},
        {"n_samples", n},
        {"quantum_numbers", {qn.n, qn.l, qn.m}},
        {"env",
         {{"gpu", env.gpu},
          {"clocks_sm_mhz", env.clocks_sm_mhz}}},
        {"rows", rows},
    };
    std::ofstream out(out_path);
    out << body.dump(2) << '\n';
  }
  return 0;
}

}  // namespace qmc::alias
