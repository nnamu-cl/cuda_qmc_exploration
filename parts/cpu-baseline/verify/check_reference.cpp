#include "parts/cpu-baseline/verify/check_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "cpu-reference/hydrogen.h"
#include "cpu-reference/original_buggy.h"
#include "harness/stats/chi2.h"
#include "harness/stats/ks.h"
#include "harness/stats/moments.h"
#include "harness/stats_smoke.h"
#include "harness/verify_gate.h"

namespace qmc::cpu::verify {
namespace {

#ifndef QMC_SOURCE_DIR
#define QMC_SOURCE_DIR "."
#endif

constexpr std::array<QuantumNumbers, 6> kOrbitals = {{
    {1, 0, 0},
    {2, 0, 0},
    {2, 1, 1},
    {3, 1, -1},
    {4, 2, 0},
    {5, 0, 0},
}};

[[nodiscard]] bool CloseRel(double got, double want, double tol = 1e-12) {
  if (!std::isfinite(got) || !std::isfinite(want)) {
    return false;
  }
  if (std::abs(want) < 1e-18) {
    return std::abs(got) < 1e-12;
  }
  return std::abs(got - want) / std::abs(want) < tol;
}

[[nodiscard]] std::filesystem::path GoldensPath() {
  return std::filesystem::path(QMC_SOURCE_DIR) / "cpu-reference" / "goldens" /
         "special_values.json";
}

[[nodiscard]] double Chi2Limit(int dof) {
  return static_cast<double>(dof) + 5.0 * std::sqrt(2.0 * dof);
}

bool CheckGoldens() {
  const auto path = GoldensPath();
  std::ifstream in(path);
  if (!in) {
    fmt::print(stderr, "missing goldens {}\n", path.string());
    return false;
  }
  nlohmann::json body;
  in >> body;
  int radial_fail = 0;
  for (const auto& row : body.at("radial")) {
    const int n = row.at("n").get<int>();
    const int l = row.at("l").get<int>();
    const double r = row.at("r").get<double>();
    const double want = row.at("R").get<double>();
    const double got = RadialRnl(n, l, r);
    if (!CloseRel(got, want)) {
      if (radial_fail < 5) {
        fmt::print(stderr, "R_nl n={} l={} r={:.6g} got={:.16g} want={:.16g}\n",
                   n, l, r, got, want);
      }
      ++radial_fail;
    }
  }
  int legendre_fail = 0;
  for (const auto& row : body.at("legendre")) {
    const int l = row.at("l").get<int>();
    const int m = row.at("m").get<int>();
    const double x = row.at("x").get<double>();
    const double want = row.at("P").get<double>();
    const double got = AssociatedLegendre(l, m, x);
    if (!CloseRel(got, want)) {
      if (legendre_fail < 5) {
        fmt::print(stderr,
                   "P_l^m l={} m={} x={:.6g} got={:.16g} want={:.16g}\n", l, m,
                   x, got, want);
      }
      ++legendre_fail;
    }
  }
  const bool ok = radial_fail == 0 && legendre_fail == 0;
  fmt::print("goldens radial_fail={} legendre_fail={} {}\n", radial_fail,
             legendre_fail, ok ? "pass" : "FAIL");
  return ok;
}

struct Draw {
  std::vector<double> r;
  std::vector<double> theta;
  std::vector<double> phi;
};

Draw DrawCorrected(const HydrogenSampler& sampler, std::uint64_t seed, int n) {
  Draw draw;
  draw.r.resize(static_cast<size_t>(n));
  draw.theta.resize(static_cast<size_t>(n));
  draw.phi.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const HydrogenSample s = sampler.Sample(seed, static_cast<std::uint32_t>(i));
    draw.r[static_cast<size_t>(i)] = s.r;
    draw.theta[static_cast<size_t>(i)] = s.theta;
    draw.phi[static_cast<size_t>(i)] = s.phi;
  }
  return draw;
}

bool CheckMoments(const Draw& draw, QuantumNumbers qn) {
  std::vector<double> r2(draw.r.size());
  std::vector<double> inv_r(draw.r.size());
  for (size_t i = 0; i < draw.r.size(); ++i) {
    r2[i] = draw.r[i] * draw.r[i];
    inv_r[i] = 1.0 / draw.r[i];
  }
  const auto m_r =
      qmc::stats::MomentVsExact(draw.r, ExactMeanR(qn.n, qn.l));
  const auto m_r2 =
      qmc::stats::MomentVsExact(r2, ExactMeanR2(qn.n, qn.l));
  const auto m_inv =
      qmc::stats::MomentVsExact(inv_r, ExactMeanInvR(qn.n));
  fmt::print("  moments <r>={:.6f} <r2>={:.6f} <1/r>={:.6f} {}\n",
             m_r.sample_mean, m_r2.sample_mean, m_inv.sample_mean,
             (m_r.passed && m_r2.passed && m_inv.passed) ? "pass" : "FAIL");
  return m_r.passed && m_r2.passed && m_inv.passed;
}

bool CheckKs(const Draw& draw, const HydrogenSampler& sampler, double max_d) {
  auto r_sorted = draw.r;
  std::ranges::sort(r_sorted);
  auto th_sorted = draw.theta;
  std::ranges::sort(th_sorted);
  const auto ks_r = qmc::stats::KolmogorovSmirnovD(
      r_sorted, [&](double r) { return sampler.RadialCdf(r); }, max_d);
  const auto ks_th = qmc::stats::KolmogorovSmirnovD(
      th_sorted, [&](double th) { return sampler.ThetaCdf(th); }, max_d);
  fmt::print("  KS D_r={:.6g} D_th={:.6g} {}\n", ks_r.D, ks_th.D,
             (ks_r.passed && ks_th.passed) ? "pass" : "FAIL");
  return ks_r.passed && ks_th.passed;
}

bool CheckChi2(const Draw& draw, const HydrogenSampler& sampler) {
  constexpr int kPhiBins = 36;
  constexpr int kThetaBins = 64;
  constexpr int kRadialBins = 128;
  const int n = static_cast<int>(draw.r.size());

  std::vector<double> phi_obs(kPhiBins, 0.0);
  for (double phi : draw.phi) {
    int bin = static_cast<int>(phi / (2.0 * std::acos(-1.0)) * kPhiBins);
    bin = std::clamp(bin, 0, kPhiBins - 1);
    phi_obs[static_cast<size_t>(bin)] += 1.0;
  }
  std::vector<double> phi_exp(kPhiBins, static_cast<double>(n) / kPhiBins);
  const auto chi_phi =
      qmc::stats::ChiSquare(phi_obs, phi_exp, Chi2Limit(kPhiBins - 1));

  std::vector<double> th_obs(kThetaBins, 0.0);
  std::vector<double> th_exp(kThetaBins, 0.0);
  const double dtheta = std::acos(-1.0) / kThetaBins;
  for (double theta : draw.theta) {
    int bin = static_cast<int>(theta / dtheta);
    bin = std::clamp(bin, 0, kThetaBins - 1);
    th_obs[static_cast<size_t>(bin)] += 1.0;
  }
  for (int i = 0; i < kThetaBins; ++i) {
    const double lo = sampler.ThetaCdf(i * dtheta);
    const double hi = sampler.ThetaCdf((i + 1) * dtheta);
    th_exp[static_cast<size_t>(i)] = static_cast<double>(n) * (hi - lo);
  }
  const auto chi_th =
      qmc::stats::ChiSquare(th_obs, th_exp, Chi2Limit(kThetaBins - 1));

  const double r_cut = sampler.RadiusMax();
  const double dr = r_cut / kRadialBins;
  std::vector<double> r_obs(kRadialBins, 0.0);
  std::vector<double> r_exp(kRadialBins, 0.0);
  for (double r : draw.r) {
    int bin = static_cast<int>(r / dr);
    bin = std::clamp(bin, 0, kRadialBins - 1);
    r_obs[static_cast<size_t>(bin)] += 1.0;
  }
  for (int i = 0; i < kRadialBins; ++i) {
    const double lo = sampler.RadialCdf(i * dr);
    const double hi = sampler.RadialCdf((i + 1) * dr);
    r_exp[static_cast<size_t>(i)] = static_cast<double>(n) * (hi - lo);
  }
  const auto chi_r =
      qmc::stats::ChiSquare(r_obs, r_exp, Chi2Limit(kRadialBins - 1));
  fmt::print("  chi2 phi={:.1f} th={:.1f} r={:.1f} {}\n", chi_phi.chi2,
             chi_th.chi2, chi_r.chi2,
             (chi_phi.passed && chi_th.passed && chi_r.passed) ? "pass"
                                                                : "FAIL");
  return chi_phi.passed && chi_th.passed && chi_r.passed;
}

bool CheckDeterminism(const HydrogenSampler& sampler) {
  constexpr int kN = 1024;
  bool ok = true;
  for (int i = 0; i < kN; ++i) {
    const auto a = sampler.Sample(0xC0FFEEULL, static_cast<std::uint32_t>(i));
    const auto b = sampler.Sample(0xC0FFEEULL, static_cast<std::uint32_t>(i));
    if (a.r != b.r || a.theta != b.theta || a.phi != b.phi) {
      ok = false;
      break;
    }
  }
  fmt::print("  determinism {}\n", ok ? "pass" : "FAIL");
  return ok;
}

int UniqueCount(const std::vector<double>& values) {
  auto copy = values;
  std::ranges::sort(copy);
  copy.erase(std::unique(copy.begin(), copy.end()), copy.end());
  return static_cast<int>(copy.size());
}

bool CheckBuggyCaught(int n_fast) {
  bool ok = true;
  const HydrogenSampler h1s({1, 0, 0});
  const HydrogenSampler h2s({2, 0, 0});
  const HydrogenSampler h31m({3, 1, -1});

  qmc::cpu::buggy::ResetTables();
  std::mt19937 gen(1);
  for (int i = 0; i < 256; ++i) {
    (void)qmc::cpu::buggy::SampleRadius(1, 0, gen);
  }
  std::vector<double> stale(static_cast<size_t>(n_fast));
  for (int i = 0; i < n_fast; ++i) {
    stale[static_cast<size_t>(i)] = qmc::cpu::buggy::SampleRadius(2, 0, gen);
  }
  auto stale_sorted = stale;
  std::ranges::sort(stale_sorted);
  const auto ks_stale = qmc::stats::KolmogorovSmirnovD(
      stale_sorted, [&](double r) { return h2s.RadialCdf(r); }, 0.02);
  const bool caught_stale = !ks_stale.passed;
  fmt::print("buggy stale-CDF KS D={:.6g} {}\n", ks_stale.D,
             caught_stale ? "pass (caught)" : "FAIL (missed)");
  ok = ok && caught_stale;

  qmc::cpu::buggy::ResetTables();
  gen.seed(2);
  std::vector<double> edges(static_cast<size_t>(n_fast));
  for (int i = 0; i < n_fast; ++i) {
    edges[static_cast<size_t>(i)] = qmc::cpu::buggy::SampleRadius(1, 0, gen);
  }
  const int unique_r = UniqueCount(edges);
  const bool caught_edges = unique_r <= 4096;
  fmt::print("buggy unique radii={} {}\n", unique_r,
             caught_edges ? "pass (caught)" : "FAIL (missed)");
  ok = ok && caught_edges;

  qmc::cpu::buggy::ResetTables();
  gen.seed(3);
  std::vector<double> neg_m(static_cast<size_t>(n_fast));
  for (int i = 0; i < n_fast; ++i) {
    neg_m[static_cast<size_t>(i)] = qmc::cpu::buggy::SampleTheta(1, -1, gen);
  }
  auto neg_sorted = neg_m;
  std::ranges::sort(neg_sorted);
  const auto ks_neg = qmc::stats::KolmogorovSmirnovD(
      neg_sorted, [&](double th) { return h31m.ThetaCdf(th); }, 0.02);
  const bool caught_neg = !ks_neg.passed;
  fmt::print("buggy negative-m KS D={:.6g} {}\n", ks_neg.D,
             caught_neg ? "pass (caught)" : "FAIL (missed)");
  ok = ok && caught_neg;

  const Draw fixed = DrawCorrected(h1s, 42, n_fast);
  const int unique_fixed = UniqueCount(fixed.r);
  const bool interpolated = unique_fixed > 4096;
  fmt::print("corrected unique radii={} {}\n", unique_fixed,
             interpolated ? "pass" : "FAIL");
  ok = ok && interpolated;
  (void)h1s;
  return ok;
}

bool CheckCorrected(int n, double max_d, bool full_orbitals) {
  bool ok = true;
  const std::array<QuantumNumbers, 2> smoke = {{{1, 0, 0}, {3, 1, -1}}};
  const int n_orbitals =
      full_orbitals ? static_cast<int>(kOrbitals.size()) : 2;
  for (int i = 0; i < n_orbitals; ++i) {
    const QuantumNumbers qn = full_orbitals ? kOrbitals[static_cast<size_t>(i)]
                                            : smoke[static_cast<size_t>(i)];
    fmt::print("orbital n={} l={} m={}\n", qn.n, qn.l, qn.m);
    const HydrogenSampler sampler(qn);
    const Draw draw = DrawCorrected(sampler, 0xC0FFEEULL, n);
    ok = CheckDeterminism(sampler) && ok;
    ok = CheckMoments(draw, qn) && ok;
    ok = CheckKs(draw, sampler, max_d) && ok;
    ok = CheckChi2(draw, sampler) && ok;
  }
  return ok;
}

}  // namespace

int RunVerify(bool full) {
  int smoke = 0;
  if (!full) {
    smoke = qmc::harness::RunStatsSmoke();
  }
  const bool goldens = CheckGoldens();
  const int n_fast = 100000;
  const bool buggy = CheckBuggyCaught(n_fast);
  const bool corrected =
      full ? CheckCorrected(10000000, 5.0e-4, true)
           : CheckCorrected(n_fast, 0.01, false);
  const bool ok = smoke == 0 && goldens && buggy && corrected;
  if (full && ok) {
    if (!qmc::harness::RecordVerifyFullOk()) {
      fmt::print(stderr, "could not write verify-full gate\n");
      return 1;
    }
    fmt::print("verify-full gate written\n");
  }
  fmt::print("verify-{} {}\n", full ? "full" : "fast", ok ? "pass" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace qmc::cpu::verify
