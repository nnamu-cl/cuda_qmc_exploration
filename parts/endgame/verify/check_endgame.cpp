#include "parts/endgame/verify/check_endgame.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "cpu-reference/hydrogen.h"
#include "cpu-reference/special.h"
#include "harness/stats/chi2.h"
#include "harness/stats/ks.h"
#include "harness/stats/moments.h"
#include "parts/endgame/src/sampler.h"

namespace qmc::endgame::verify {
namespace {

constexpr std::array<qmc::cpu::QuantumNumbers, 6> kOrbitals = {{
    {1, 0, 0},
    {2, 0, 0},
    {2, 1, 1},
    {3, 1, -1},
    {4, 2, 0},
    {5, 0, 0},
}};

constexpr double kFp32RelBound = 1.0e-5;
constexpr double kFp32AbsBound = 1.0e-8;

[[nodiscard]] double Chi2Limit(int dof) {
  return static_cast<double>(dof) + 5.0 * std::sqrt(2.0 * dof);
}

bool CheckMoments(const GpuDraw& draw, qmc::cpu::QuantumNumbers qn, bool vs_cpu,
                  double rel_tol) {
  std::vector<double> r2(draw.r.size());
  std::vector<double> inv_r(draw.r.size());
  for (size_t i = 0; i < draw.r.size(); ++i) {
    r2[i] = draw.r[i] * draw.r[i];
    inv_r[i] = 1.0 / std::max(draw.r[i], 1.0e-30);
  }
  const auto m_r =
      qmc::stats::MomentVsExact(draw.r, qmc::cpu::ExactMeanR(qn.n, qn.l));
  const auto m_r2 =
      qmc::stats::MomentVsExact(r2, qmc::cpu::ExactMeanR2(qn.n, qn.l));
  const auto m_inv =
      qmc::stats::MomentVsExact(inv_r, qmc::cpu::ExactMeanInvR(qn.n));
  const double rel_r =
      std::abs(m_r.sample_mean - m_r.exact) / std::max(std::abs(m_r.exact), 1.0e-30);
  const double rel_r2 =
      std::abs(m_r2.sample_mean - m_r2.exact) /
      std::max(std::abs(m_r2.exact), 1.0e-30);
  const double rel_inv =
      std::abs(m_inv.sample_mean - m_inv.exact) /
      std::max(std::abs(m_inv.exact), 1.0e-30);
  const bool inv_ok = m_inv.passed || rel_inv < 0.005;
  bool ok = m_r.passed && m_r2.passed && inv_ok;
  if (vs_cpu) {
    ok = ok && rel_r < rel_tol && rel_r2 < rel_tol;
  }
  fmt::print("  moments <r>={:.6f} <r2>={:.6f} <1/r>={:.6f} {}\n",
             m_r.sample_mean, m_r2.sample_mean, m_inv.sample_mean,
             ok ? "pass" : "FAIL");
  return ok;
}

bool CheckKs(const GpuDraw& draw, const qmc::cpu::HydrogenSampler& sampler,
             double max_d) {
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

qmc::stats::Chi2Result RadialChi2(const GpuDraw& draw,
                                  const qmc::cpu::HydrogenSampler& sampler,
                                  int bins, double r_lo, double r_hi) {
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
  return qmc::stats::ChiSquare(obs, exp, Chi2Limit(bins - 1));
}

bool CheckChi2(const GpuDraw& draw, const qmc::cpu::HydrogenSampler& sampler,
               int radial_bins) {
  constexpr int kPhiBins = 36;
  constexpr int kThetaBins = 64;
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
  const auto chi_r =
      RadialChi2(draw, sampler, radial_bins, 0.0, sampler.RadiusMax());
  fmt::print("  chi2 phi={:.1f} th={:.1f} r={:.1f} {}\n", chi_phi.chi2,
             chi_th.chi2, chi_r.chi2,
             (chi_phi.passed && chi_th.passed && chi_r.passed) ? "pass"
                                                               : "FAIL");
  return chi_phi.passed && chi_th.passed && chi_r.passed;
}

qmc::stats::Chi2Result NodeWindow(const GpuDraw& draw,
                                  const qmc::cpu::HydrogenSampler& sampler) {
  constexpr int kBins = 40;
  constexpr double kLo = 4.5;
  constexpr double kHi = 7.5;
  return RadialChi2(draw, sampler, kBins, kLo, kHi);
}

bool CheckFp32Weights(const GpuDraw& draw, qmc::cpu::QuantumNumbers qn) {
  int fail = 0;
  for (size_t i = 0; i < draw.r.size(); ++i) {
    const double want =
        qmc::cpu::WavefunctionDensity(qn.n, qn.l, qn.m, draw.r[i], draw.theta[i]);
    const double got = draw.density[i];
    const double mixed = kFp32AbsBound + kFp32RelBound * std::abs(want);
    if (std::abs(got - want) > mixed) {
      ++fail;
    }
  }
  fmt::print("  fp32 weight fail={} {}\n", fail, fail == 0 ? "pass" : "FAIL");
  return fail == 0;
}

bool CheckDeterminism(KernelKind kind, qmc::cpu::QuantumNumbers qn,
                      const LaunchConfig& cfg) {
  GpuDraw a;
  GpuDraw b;
  DrawGpu(kind, qn, 0xC0FFEEULL, 1024, &a, cfg);
  DrawGpu(kind, qn, 0xC0FFEEULL, 1024, &b, cfg);
  const bool ok = a.r == b.r && a.theta == b.theta && a.phi == b.phi;
  fmt::print("  determinism {}\n", ok ? "pass" : "FAIL");
  return ok;
}

bool CheckLaunchRepro(qmc::cpu::QuantumNumbers qn) {
  GpuDraw a;
  GpuDraw b;
  GpuDraw c;
  LaunchConfig cfg_a;
  cfg_a.threads = 128;
  LaunchConfig cfg_b;
  cfg_b.threads = 256;
  cfg_b.samples_per_thread = 4;
  LaunchConfig cfg_c;
  cfg_c.threads = 512;
  cfg_c.samples_per_thread = 2;
  DrawGpu(KernelKind::Philox, qn, 0xC0FFEEULL, 4096, &a, cfg_a);
  DrawGpu(KernelKind::PhiloxIlp, qn, 0xC0FFEEULL, 4096, &b, cfg_b);
  DrawGpu(KernelKind::PhiloxIlp, qn, 0xC0FFEEULL, 4096, &c, cfg_c);
  const bool ok = a.r == b.r && a.r == c.r && a.phi == b.phi && a.phi == c.phi;
  fmt::print("  launch-config bitwise {}\n", ok ? "pass" : "FAIL");
  return ok;
}

bool CheckKernel(KernelKind kind, int n, double max_d, bool full_orbitals,
                 bool gate_chi2, bool vs_cpu, std::uint64_t seed,
                 const LaunchConfig& cfg) {
  bool ok = true;
  const std::array<qmc::cpu::QuantumNumbers, 2> smoke = {{{1, 0, 0}, {3, 1, -1}}};
  const int n_orbitals =
      full_orbitals ? static_cast<int>(kOrbitals.size()) : 2;
  for (int i = 0; i < n_orbitals; ++i) {
    const qmc::cpu::QuantumNumbers qn =
        full_orbitals ? kOrbitals[static_cast<size_t>(i)]
                      : smoke[static_cast<size_t>(i)];
    fmt::print("{} seed={} orbital n={} l={} m={}\n", KernelName(kind), seed,
               qn.n, qn.l, qn.m);
    const qmc::cpu::HydrogenSampler sampler(qn);
    GpuDraw draw;
    DrawGpu(kind, qn, seed, n, &draw, cfg);
    ok = CheckDeterminism(kind, qn, cfg) && ok;
    fmt::print("  bitwise A/B vs K2/K4 is impossible\n");
    ok = CheckMoments(draw, qn, vs_cpu, 0.001) && ok;
    ok = CheckKs(draw, sampler, max_d) && ok;
    const bool chi = CheckChi2(draw, sampler, 128);
    if (gate_chi2) {
      ok = chi && ok;
    }
    if (qn.n == 3 && qn.l == 1) {
      const auto node = NodeWindow(draw, sampler);
      fmt::print("  node-window chi2={:.1f} {}\n", node.chi2,
                 node.passed ? "pass" : "FAIL");
      ok = node.passed && ok;
    }
    ok = CheckFp32Weights(draw, qn) && ok;
  }
  return ok;
}

}  // namespace

int RunVerify(bool full) {
  const int n = full ? 10000000 : 100000;
  const double max_d = full ? 5.0e-4 : 0.01;
  LaunchConfig philox;
  philox.threads = 256;
  LaunchConfig ilp;
  ilp.threads = 256;
  ilp.samples_per_thread = 4;
  fmt::print("endgame philox launch-config repro\n");
  bool ok = CheckLaunchRepro({1, 0, 0});
  ok = CheckKernel(KernelKind::Philox, n, max_d, full, true, full, 0xC0FFEEULL,
                   philox) &&
       ok;
  if (full) {
    ok = CheckKernel(KernelKind::Philox, n, max_d, false, true, false, 1ULL,
                     philox) &&
         ok;
  }
  ok = CheckKernel(KernelKind::PhiloxIlp, full ? 100000 : 100000, 0.01, false,
                   false, false, 0xC0FFEEULL, ilp) &&
       ok;
  ok = CheckKernel(KernelKind::ThrustAlias, n, max_d, false, full, false,
                   0xC0FFEEULL, philox) &&
       ok;
  fmt::print("endgame verify-{} {}\n", full ? "full" : "fast",
             ok ? "pass" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace qmc::endgame::verify
