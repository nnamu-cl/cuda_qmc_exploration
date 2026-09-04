#include "parts/packed-records/verify/check_packed.h"

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
#include "parts/packed-records/src/sampler.h"

namespace qmc::packed::verify {
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

// The split must be a transpose, not a rebuild. Same prob, same alias, same
// geometry, and the Vose masses reconstructed from the 8 B draw array must
// equal those reconstructed from the 24 B records K5/K7 shipped.
bool CheckSplitIsFaithful(qmc::cpu::QuantumNumbers qn) {
  const qmc::cpu::HydrogenSampler sampler(qn);
  const auto radial = qmc::alias::BuildRadialAlias(sampler);
  const auto theta = qmc::alias::BuildThetaAlias(sampler);
  bool ok = true;
  double worst_mass = 0.0;
  for (const auto* bins : {&radial, &theta}) {
    const PackedTable table = SplitAliasBins(*bins);
    if (table.draw.size() != bins->size()) {
      ok = false;
      continue;
    }
    for (size_t i = 0; i < bins->size(); ++i) {
      const qmc::alias::AliasBin& src = (*bins)[i];
      ok = ok && table.draw[i].prob == src.prob &&
           static_cast<int>(table.draw[i].alias) == src.alias &&
           table.uniform[i].lo == src.lo && table.uniform[i].width == src.width &&
           table.linear[i].lo == src.lo && table.linear[i].width == src.width &&
           table.linear[i].y0 == src.y0 && table.linear[i].y1 == src.y1;
    }
    const auto want = qmc::alias::ReconstructMasses(*bins);
    std::vector<qmc::alias::AliasBin> rebuilt(bins->size());
    for (size_t i = 0; i < bins->size(); ++i) {
      rebuilt[i].prob = table.draw[i].prob;
      rebuilt[i].alias = static_cast<int>(table.draw[i].alias);
    }
    const auto got = qmc::alias::ReconstructMasses(rebuilt);
    for (size_t i = 0; i < want.size(); ++i) {
      worst_mass = std::max(worst_mass, std::abs(want[i] - got[i]));
    }
  }
  ok = ok && worst_mass == 0.0;
  fmt::print("  split faithful (worst mass delta {:.3g}) {}\n", worst_mass,
             ok ? "pass" : "FAIL");
  return ok;
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
  const double rel_r = std::abs(m_r.sample_mean - m_r.exact) /
                       std::max(std::abs(m_r.exact), 1.0e-30);
  const double rel_r2 = std::abs(m_r2.sample_mean - m_r2.exact) /
                        std::max(std::abs(m_r2.exact), 1.0e-30);
  const double rel_inv = std::abs(m_inv.sample_mean - m_inv.exact) /
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

// Same seed, same sample index, four launch shapes including the 48 KB shared
// block and two ILP depths. The Philox counter is the sample index or this
// fails.
bool CheckLaunchRepro(qmc::cpu::QuantumNumbers qn) {
  constexpr int kN = 4096;
  GpuDraw base;
  LaunchConfig flat;
  flat.threads = 128;
  DrawGpu(KernelKind::Packed, qn, 0xC0FFEEULL, kN, &base, flat);

  bool ok = true;
  const auto same_as_base = [&](const char* label, KernelKind kind,
                                const LaunchConfig& cfg) {
    GpuDraw other;
    DrawGpu(kind, qn, 0xC0FFEEULL, kN, &other, cfg);
    const bool match = base.r == other.r && base.theta == other.theta &&
                       base.phi == other.phi && base.x == other.x;
    fmt::print("  launch-config bitwise {} {}\n", label,
               match ? "pass" : "FAIL");
    ok = match && ok;
  };

  LaunchConfig flat512;
  flat512.threads = 512;
  same_as_base("packed/512", KernelKind::Packed, flat512);

  LaunchConfig shared256;
  shared256.threads = 256;
  same_as_base("packed_shared/256", KernelKind::PackedShared, shared256);

  LaunchConfig shared1024;
  shared1024.threads = 1024;
  same_as_base("packed_shared/1024", KernelKind::PackedShared, shared1024);

  LaunchConfig ilp2;
  ilp2.threads = 256;
  ilp2.samples_per_thread = 2;
  same_as_base("packed_ilp/256/S=2", KernelKind::PackedIlp, ilp2);

  LaunchConfig ilp4;
  ilp4.threads = 128;
  ilp4.samples_per_thread = 4;
  same_as_base("packed_ilp/128/S=4", KernelKind::PackedIlp, ilp4);
  return ok;
}

// The linear variant reads the same values through a different layout and runs
// the same arithmetic in the same order, so it must agree with K7 bit for bit.
// Anything else means the transpose changed the numbers, not just the loads.
bool CheckMatchesK7(qmc::cpu::QuantumNumbers qn) {
  constexpr int kN = 65536;
  GpuDraw k9;
  LaunchConfig cfg;
  DrawGpu(KernelKind::PackedLinear, qn, 0xC0FFEEULL, kN, &k9, cfg);
  qmc::endgame::GpuDraw k7;
  qmc::endgame::LaunchConfig k7_cfg;
  qmc::endgame::DrawGpu(qmc::endgame::KernelKind::Philox, qn, 0xC0FFEEULL, kN,
                        &k7, k7_cfg);
  const bool ok = k9.r == k7.r && k9.theta == k7.theta && k9.phi == k7.phi &&
                  k9.x == k7.x && k9.y == k7.y && k9.z == k7.z &&
                  k9.density == k7.density;
  int mismatch = 0;
  for (size_t i = 0; i < k9.r.size(); ++i) {
    if (k9.r[i] != k7.r[i] || k9.theta[i] != k7.theta[i]) {
      ++mismatch;
    }
  }
  fmt::print("  packed_linear vs K7 philox bitwise mismatch={} {}\n", mismatch,
             ok ? "pass" : "FAIL");
  return ok;
}

// The uniform interior is a different sampling rule, not a different layout,
// so it is checked against the analytic CDF rather than against K7.
bool CheckUniformVsLinearAgree(qmc::cpu::QuantumNumbers qn, int n) {
  GpuDraw uni;
  GpuDraw lin;
  LaunchConfig cfg;
  DrawGpu(KernelKind::Packed, qn, 0xC0FFEEULL, n, &uni, cfg);
  DrawGpu(KernelKind::PackedLinear, qn, 0xC0FFEEULL, n, &lin, cfg);
  const qmc::cpu::HydrogenSampler sampler(qn);
  const auto chi_uni = RadialChi2(uni, sampler, 256, 0.0, sampler.RadiusMax());
  const auto chi_lin = RadialChi2(lin, sampler, 256, 0.0, sampler.RadiusMax());
  const bool ok = chi_uni.passed && chi_lin.passed;
  fmt::print("  interior rule chi2_radial_256 uniform={:.2f} linear={:.2f} {}\n",
             chi_uni.chi2, chi_lin.chi2, ok ? "pass" : "FAIL");
  return ok;
}

bool CheckKernel(KernelKind kind, int n, double max_d, bool full_orbitals,
                 bool gate_chi2, bool vs_cpu, std::uint64_t seed,
                 const LaunchConfig& cfg) {
  bool ok = true;
  const std::array<qmc::cpu::QuantumNumbers, 2> smoke = {{{1, 0, 0}, {3, 1, -1}}};
  const int n_orbitals = full_orbitals ? static_cast<int>(kOrbitals.size()) : 2;
  for (int i = 0; i < n_orbitals; ++i) {
    const qmc::cpu::QuantumNumbers qn = full_orbitals
                                            ? kOrbitals[static_cast<size_t>(i)]
                                            : smoke[static_cast<size_t>(i)];
    fmt::print("{} seed={} orbital n={} l={} m={}\n", KernelName(kind), seed,
               qn.n, qn.l, qn.m);
    const qmc::cpu::HydrogenSampler sampler(qn);
    GpuDraw draw;
    DrawGpu(kind, qn, seed, n, &draw, cfg);
    ok = CheckDeterminism(kind, qn, cfg) && ok;
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
  LaunchConfig flat;
  LaunchConfig shared;
  shared.threads = 1024;
  LaunchConfig ilp;
  ilp.threads = 256;
  ilp.samples_per_thread = 4;

  fmt::print("packed-records table split\n");
  bool ok = CheckSplitIsFaithful({1, 0, 0});
  ok = CheckSplitIsFaithful({3, 1, -1}) && ok;
  ok = CheckSplitIsFaithful({4, 2, 0}) && ok;

  fmt::print("packed-records layout equivalence\n");
  ok = CheckMatchesK7({1, 0, 0}) && ok;
  ok = CheckMatchesK7({3, 1, -1}) && ok;
  ok = CheckLaunchRepro({1, 0, 0}) && ok;
  ok = CheckUniformVsLinearAgree({3, 1, -1}, full ? 1000000 : 100000) && ok;

  ok = CheckKernel(KernelKind::Packed, n, max_d, full, true, full, 0xC0FFEEULL,
                   flat) &&
       ok;
  if (full) {
    ok = CheckKernel(KernelKind::Packed, n, max_d, false, true, false, 1ULL,
                     flat) &&
         ok;
  }
  ok = CheckKernel(KernelKind::PackedLinear, n, max_d, false, true, false,
                   0xC0FFEEULL, flat) &&
       ok;
  ok = CheckKernel(KernelKind::PackedShared, full ? 1000000 : 100000, 0.01,
                   false, false, false, 0xC0FFEEULL, shared) &&
       ok;
  ok = CheckKernel(KernelKind::PackedIlp, full ? 1000000 : 100000, 0.01, false,
                   false, false, 0xC0FFEEULL, ilp) &&
       ok;
  fmt::print("packed-records verify-{} {}\n", full ? "full" : "fast",
             ok ? "pass" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace qmc::packed::verify
