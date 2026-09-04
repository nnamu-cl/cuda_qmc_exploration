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

namespace qmc::alias {
namespace {

double Chi2Limit(int dof) {
  return static_cast<double>(dof) + 5.0 * std::sqrt(2.0 * dof);
}

double RadialChi2Value(const GpuDraw& draw,
                       const qmc::cpu::HydrogenSampler& sampler, int bins) {
  const int n = static_cast<int>(draw.r.size());
  const double r_hi = sampler.RadiusMax();
  std::vector<double> obs(static_cast<size_t>(bins), 0.0);
  std::vector<double> exp(static_cast<size_t>(bins), 0.0);
  for (double r : draw.r) {
    if (r < 0.0 || r >= r_hi) {
      continue;
    }
    int bin = static_cast<int>(r / r_hi * bins);
    bin = std::clamp(bin, 0, bins - 1);
    obs[static_cast<size_t>(bin)] += 1.0;
  }
  for (int i = 0; i < bins; ++i) {
    const double lo = sampler.RadialCdf(r_hi * i / bins);
    const double hi = sampler.RadialCdf(r_hi * (i + 1) / bins);
    exp[static_cast<size_t>(i)] = static_cast<double>(n) * (hi - lo);
  }
  return qmc::stats::ChiSquare(obs, exp, Chi2Limit(bins - 1)).chi2;
}

}  // namespace

int RunInteriorCompare(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  const qmc::cpu::QuantumNumbers qn{1, 0, 0};
  const qmc::cpu::HydrogenSampler sampler(qn);
  nlohmann::json rows = nlohmann::json::array();
  for (KernelKind kind : {KernelKind::Uniform, KernelKind::Linear}) {
    GpuDraw draw;
    DrawGpu(kind, qn, 0xC0FFEEULL, n, &draw);
    const double chi = RadialChi2Value(draw, sampler, 256);
    rows.push_back({
        {"kernel", KernelName(kind)},
        {"chi2_radial_256", chi},
    });
    fmt::print("{} chi2_r={:.1f}\n", KernelName(kind), chi);
  }
  if (!out_path.empty()) {
    nlohmann::json body = {
        {"part", "alias-method"},
        {"experiment", "uniform_vs_linear"},
        {"n_samples", n},
        {"env",
         {{"gpu", qmc::harness::CaptureEnv().gpu},
          {"clocks_sm_mhz", qmc::harness::CaptureEnv().clocks_sm_mhz}}},
        {"rows", rows},
    };
    std::ofstream out(out_path);
    out << body.dump(2) << '\n';
  }
  return 0;
}

}  // namespace qmc::alias
