#include "harness/stats_smoke.h"

#include <algorithm>
#include <vector>

#include <fmt/format.h>

#include "harness/rng/philox.h"
#include "harness/stats/chi2.h"
#include "harness/stats/ks.h"
#include "harness/stats/moments.h"

namespace qmc::harness {
namespace {

std::vector<double> UniformPhilox(int n, std::uint64_t seed) {
  std::vector<double> out(static_cast<size_t>(n));
  const qmc::rng::Philox4x32Key key = qmc::rng::SeedToKey(seed);
  for (int i = 0; i < n; ++i) {
    const qmc::rng::Philox4x32Ctr ctr = qmc::rng::Philox4x32TenRounds(
        qmc::rng::MakeCounter(static_cast<std::uint32_t>(i), 0), key);
    out[static_cast<size_t>(i)] = qmc::rng::Uint32ToUnitFloat(ctr.v[0]);
  }
  return out;
}

}  // namespace

int RunStatsSmoke() {
  constexpr int kN = 100000;
  auto samples = UniformPhilox(kN, 0xC0FFEEULL);
  std::sort(samples.begin(), samples.end());

  const auto ks = qmc::stats::KolmogorovSmirnovD(
      samples, [](double x) { return x; }, 0.01);
  fmt::print("KS D={:.6f} {}\n", ks.D, ks.passed ? "pass" : "FAIL");

  constexpr int kBins = 20;
  std::vector<double> observed(kBins, 0.0);
  for (double x : samples) {
    int bin = static_cast<int>(x * kBins);
    if (bin >= kBins) {
      bin = kBins - 1;
    }
    if (bin < 0) {
      bin = 0;
    }
    observed[static_cast<size_t>(bin)] += 1.0;
  }
  const double expected_count = static_cast<double>(kN) / kBins;
  std::vector<double> expected(kBins, expected_count);
  const auto chi2 =
      qmc::stats::ChiSquare(observed, expected, 2.0 * (kBins - 1) + 20.0);
  fmt::print("chi2={:.3f} dof={} {}\n", chi2.chi2, chi2.dof,
             chi2.passed ? "pass" : "FAIL");

  const auto moment = qmc::stats::MomentVsExact(samples, 0.5, 5.0);
  fmt::print("mean={:.6f} se={:.6f} {}\n", moment.sample_mean,
             moment.standard_error, moment.passed ? "pass" : "FAIL");

  auto biased = samples;
  for (double& x : biased) {
    x *= 0.5;
  }
  std::sort(biased.begin(), biased.end());
  const auto ks_fail = qmc::stats::KolmogorovSmirnovD(
      biased, [](double x) { return x; }, 0.01);
  fmt::print("biased KS D={:.6f} {}\n", ks_fail.D,
             !ks_fail.passed ? "pass (caught)" : "FAIL (missed bias)");

  const bool ok = ks.passed && chi2.passed && moment.passed && !ks_fail.passed;
  return ok ? 0 : 1;
}

}  // namespace qmc::harness
