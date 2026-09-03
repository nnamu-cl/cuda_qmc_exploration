#include "parts/naive-cuda/verify/check_naive.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "cpu-reference/hydrogen.h"
#include "cpu-reference/special.h"
#include "harness/device_memory.h"
#include "harness/stats/chi2.h"
#include "harness/stats/ks.h"
#include "harness/stats/moments.h"
#include "harness/timing.h"
#include "parts/naive-cuda/src/sampler.h"

namespace qmc::naive::verify {
namespace {

#ifndef QMC_SOURCE_DIR
#define QMC_SOURCE_DIR "."
#endif

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

[[nodiscard]] bool CloseRel(double got, double want, double tol) {
  if (!std::isfinite(got) || !std::isfinite(want)) {
    return false;
  }
  if (std::abs(want) < 1e-18) {
    return std::abs(got) < tol;
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

bool CheckDeviceGoldens() {
  const auto path = GoldensPath();
  std::ifstream in(path);
  if (!in) {
    fmt::print(stderr, "missing goldens {}\n", path.string());
    return false;
  }
  nlohmann::json body;
  in >> body;

  std::vector<int> n;
  std::vector<int> l;
  std::vector<double> r;
  std::vector<double> want_r;
  std::vector<double> norm;
  for (const auto& row : body.at("radial")) {
    n.push_back(row.at("n").get<int>());
    l.push_back(row.at("l").get<int>());
    r.push_back(row.at("r").get<double>());
    want_r.push_back(row.at("R").get<double>());
    norm.push_back(qmc::cpu::RadialNorm(n.back(), l.back()));
  }
  const int n_rad = static_cast<int>(n.size());
  auto d_n = qmc::harness::DeviceAlloc<int>(static_cast<size_t>(n_rad));
  auto d_l = qmc::harness::DeviceAlloc<int>(static_cast<size_t>(n_rad));
  auto d_r = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n_rad));
  auto d_norm = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n_rad));
  auto d_out = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n_rad));
  qmc::harness::CopyHostToDevice(d_n.get(), n.data(), n.size() * sizeof(int));
  qmc::harness::CopyHostToDevice(d_l.get(), l.data(), l.size() * sizeof(int));
  qmc::harness::CopyHostToDevice(d_r.get(), r.data(), r.size() * sizeof(double));
  qmc::harness::CopyHostToDevice(d_norm.get(), norm.data(), norm.size() * sizeof(double));
  EvaluateRadialGpu(d_n.get(), d_l.get(), d_r.get(), d_norm.get(), d_out.get(),
                    n_rad);
  std::vector<double> got_r(static_cast<size_t>(n_rad));
  qmc::harness::CopyDeviceToHost(got_r.data(), d_out.get(), got_r.size() * sizeof(double));

  int radial64_fail = 0;
  for (int i = 0; i < n_rad; ++i) {
    if (!CloseRel(got_r[static_cast<size_t>(i)], want_r[static_cast<size_t>(i)],
                  1e-12)) {
      if (radial64_fail < 5) {
        fmt::print(stderr, "device R64 n={} l={} r={:.6g} got={:.16g} want={:.16g}\n",
                   n[static_cast<size_t>(i)], l[static_cast<size_t>(i)],
                   r[static_cast<size_t>(i)], got_r[static_cast<size_t>(i)],
                   want_r[static_cast<size_t>(i)]);
      }
      ++radial64_fail;
    }
  }

  std::vector<float> r32(r.begin(), r.end());
  std::vector<float> norm32(norm.begin(), norm.end());
  auto d_r32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n_rad));
  auto d_norm32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n_rad));
  auto d_out32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n_rad));
  qmc::harness::CopyHostToDevice(d_r32.get(), r32.data(), r32.size() * sizeof(float));
  qmc::harness::CopyHostToDevice(d_norm32.get(), norm32.data(), norm32.size() * sizeof(float));
  EvaluateRadialGpuF32(d_n.get(), d_l.get(), d_r32.get(), d_norm32.get(),
                       d_out32.get(), n_rad);
  std::vector<float> got_r32(static_cast<size_t>(n_rad));
  qmc::harness::CopyDeviceToHost(got_r32.data(), d_out32.get(), got_r32.size() * sizeof(float));
  int radial32_fail = 0;
  double max_r32 = 0.0;
  for (int i = 0; i < n_rad; ++i) {
    const double got = static_cast<double>(got_r32[static_cast<size_t>(i)]);
    const double want = want_r[static_cast<size_t>(i)];
    const double err = (std::abs(want) < 1e-18) ? std::abs(got)
                                                : std::abs(got - want) / std::abs(want);
    max_r32 = std::max(max_r32, err);
    if (!CloseRel(got, want, kFp32RelBound)) {
      ++radial32_fail;
    }
  }

  std::vector<int> pl;
  std::vector<int> pm;
  std::vector<double> x;
  std::vector<double> want_p;
  for (const auto& row : body.at("legendre")) {
    pl.push_back(row.at("l").get<int>());
    pm.push_back(row.at("m").get<int>());
    x.push_back(row.at("x").get<double>());
    want_p.push_back(row.at("P").get<double>());
  }
  const int n_p = static_cast<int>(pl.size());
  auto d_pl = qmc::harness::DeviceAlloc<int>(static_cast<size_t>(n_p));
  auto d_pm = qmc::harness::DeviceAlloc<int>(static_cast<size_t>(n_p));
  auto d_x = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n_p));
  auto d_pout = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n_p));
  qmc::harness::CopyHostToDevice(d_pl.get(), pl.data(), pl.size() * sizeof(int));
  qmc::harness::CopyHostToDevice(d_pm.get(), pm.data(), pm.size() * sizeof(int));
  qmc::harness::CopyHostToDevice(d_x.get(), x.data(), x.size() * sizeof(double));
  EvaluateLegendreGpu(d_pl.get(), d_pm.get(), d_x.get(), d_pout.get(), n_p);
  std::vector<double> got_p(static_cast<size_t>(n_p));
  qmc::harness::CopyDeviceToHost(got_p.data(), d_pout.get(), got_p.size() * sizeof(double));
  int leg64_fail = 0;
  for (int i = 0; i < n_p; ++i) {
    if (!CloseRel(got_p[static_cast<size_t>(i)], want_p[static_cast<size_t>(i)],
                  1e-12)) {
      ++leg64_fail;
    }
  }

  std::vector<float> x32(x.begin(), x.end());
  auto d_x32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n_p));
  auto d_pout32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n_p));
  qmc::harness::CopyHostToDevice(d_x32.get(), x32.data(), x32.size() * sizeof(float));
  EvaluateLegendreGpuF32(d_pl.get(), d_pm.get(), d_x32.get(), d_pout32.get(),
                         n_p);
  std::vector<float> got_p32(static_cast<size_t>(n_p));
  qmc::harness::CopyDeviceToHost(got_p32.data(), d_pout32.get(), got_p32.size() * sizeof(float));
  int leg32_fail = 0;
  double max_p32 = 0.0;
  for (int i = 0; i < n_p; ++i) {
    const double got = static_cast<double>(got_p32[static_cast<size_t>(i)]);
    const double want = want_p[static_cast<size_t>(i)];
    const double err = (std::abs(want) < 1e-18) ? std::abs(got)
                                                : std::abs(got - want) / std::abs(want);
    max_p32 = std::max(max_p32, err);
    if (!CloseRel(got, want, kFp32RelBound)) {
      ++leg32_fail;
    }
  }

  const bool ok = radial64_fail == 0 && leg64_fail == 0 && radial32_fail == 0 &&
                  leg32_fail == 0;
  fmt::print(
      "device goldens R64_fail={} P64_fail={} R32_fail={} P32_fail={} "
      "R32_maxrel={:.3e} P32_maxrel={:.3e} {}\n",
      radial64_fail, leg64_fail, radial32_fail, leg32_fail, max_r32, max_p32,
      ok ? "pass" : "FAIL");
  return ok;
}

bool CheckInvertAb() {
  const qmc::cpu::HydrogenSampler sampler({1, 0, 0});
  constexpr int kN = 4096;
  std::vector<double> u(kN);
  std::vector<double> want(kN);
  for (int i = 0; i < kN; ++i) {
    u[static_cast<size_t>(i)] = (static_cast<double>(i) + 0.5) / kN;
    want[static_cast<size_t>(i)] =
        sampler.InvertRadial(u[static_cast<size_t>(i)]);
  }
  const int n_bins = static_cast<int>(sampler.RadialNodes().size());
  auto d_nodes = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n_bins));
  auto d_cdf = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n_bins));
  auto d_u = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(kN));
  auto d_r = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(kN));
  qmc::harness::CopyHostToDevice(d_nodes.get(), sampler.RadialNodes().data(),
          static_cast<size_t>(n_bins) * sizeof(double));
  qmc::harness::CopyHostToDevice(d_cdf.get(), sampler.RadialCdf().data(),
          static_cast<size_t>(n_bins) * sizeof(double));
  qmc::harness::CopyHostToDevice(d_u.get(), u.data(), u.size() * sizeof(double));
  InvertRadialGpu(d_nodes.get(), d_cdf.get(), n_bins, d_u.get(), d_r.get(), kN);
  std::vector<double> got(kN);
  qmc::harness::CopyDeviceToHost(got.data(), d_r.get(), got.size() * sizeof(double));
  int fail = 0;
  for (int i = 0; i < kN; ++i) {
    if (got[static_cast<size_t>(i)] != want[static_cast<size_t>(i)]) {
      if (!CloseRel(got[static_cast<size_t>(i)], want[static_cast<size_t>(i)],
                    1e-14)) {
        ++fail;
      }
    }
  }
  fmt::print("invert A/B fail={} {}\n", fail, fail == 0 ? "pass" : "FAIL");
  return fail == 0;
}

bool CheckMoments(const GpuDraw& draw, qmc::cpu::QuantumNumbers qn) {
  std::vector<double> r2(draw.r.size());
  std::vector<double> inv_r(draw.r.size());
  for (size_t i = 0; i < draw.r.size(); ++i) {
    r2[i] = draw.r[i] * draw.r[i];
    inv_r[i] = 1.0 / draw.r[i];
  }
  const auto m_r =
      qmc::stats::MomentVsExact(draw.r, qmc::cpu::ExactMeanR(qn.n, qn.l));
  const auto m_r2 =
      qmc::stats::MomentVsExact(r2, qmc::cpu::ExactMeanR2(qn.n, qn.l));
  const auto m_inv =
      qmc::stats::MomentVsExact(inv_r, qmc::cpu::ExactMeanInvR(qn.n));
  fmt::print("  moments <r>={:.6f} <r2>={:.6f} <1/r>={:.6f} {}\n",
             m_r.sample_mean, m_r2.sample_mean, m_inv.sample_mean,
             (m_r.passed && m_r2.passed && m_inv.passed) ? "pass" : "FAIL");
  return m_r.passed && m_r2.passed && m_inv.passed;
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

bool CheckChi2(const GpuDraw& draw, const qmc::cpu::HydrogenSampler& sampler) {
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

bool CheckDeterminism(KernelKind kind, qmc::cpu::QuantumNumbers qn) {
  GpuDraw a;
  GpuDraw b;
  DrawGpu(kind, qn, 0xC0FFEEULL, 1024, &a);
  DrawGpu(kind, qn, 0xC0FFEEULL, 1024, &b);
  bool ok = a.r == b.r && a.theta == b.theta && a.phi == b.phi;
  fmt::print("  determinism {}\n", ok ? "pass" : "FAIL");
  return ok;
}

bool CheckFp32Weights(const GpuDraw& draw, qmc::cpu::QuantumNumbers qn) {
  double max_rel = 0.0;
  int fail = 0;
  for (size_t i = 0; i < draw.r.size(); ++i) {
    const double want =
        qmc::cpu::WavefunctionDensity(qn.n, qn.l, qn.m, draw.r[i], draw.theta[i]);
    const double got = draw.density[i];
    const double mixed = kFp32AbsBound + kFp32RelBound * std::abs(want);
    if (std::abs(got - want) > mixed) {
      if (fail < 3) {
        fmt::print(stderr, "  fp32 weight r={:.6g} th={:.6g} got={:.6g} want={:.6g}\n",
                   draw.r[i], draw.theta[i], got, want);
      }
      ++fail;
    }
    if (std::abs(want) > 1e-8) {
      max_rel = std::max(max_rel, std::abs(got - want) / std::abs(want));
    }
  }
  fmt::print("  fp32 weight max_rel(|w|>1e-8)={:.3e} fail={} {}\n", max_rel, fail,
             fail == 0 ? "pass" : "FAIL");
  return fail == 0;
}

bool CheckKernel(KernelKind kind, int n, double max_d, bool full_orbitals) {
  bool ok = true;
  const std::array<qmc::cpu::QuantumNumbers, 2> smoke = {{{1, 0, 0}, {3, 1, -1}}};
  const int n_orbitals =
      full_orbitals ? static_cast<int>(kOrbitals.size()) : 2;
  for (int i = 0; i < n_orbitals; ++i) {
    const qmc::cpu::QuantumNumbers qn =
        full_orbitals ? kOrbitals[static_cast<size_t>(i)]
                      : smoke[static_cast<size_t>(i)];
    fmt::print("{} orbital n={} l={} m={}\n", KernelName(kind), qn.n, qn.l,
               qn.m);
    const qmc::cpu::HydrogenSampler sampler(qn);
    GpuDraw draw;
    DrawGpu(kind, qn, 0xC0FFEEULL, n, &draw);
    ok = CheckDeterminism(kind, qn) && ok;
    ok = CheckMoments(draw, qn) && ok;
    ok = CheckKs(draw, sampler, max_d) && ok;
    ok = CheckChi2(draw, sampler) && ok;
    if (kind != KernelKind::Fp64) {
      ok = CheckFp32Weights(draw, qn) && ok;
    } else {
      int dens_fail = 0;
      for (size_t j = 0; j < draw.r.size(); ++j) {
        const double want = qmc::cpu::WavefunctionDensity(
            qn.n, qn.l, qn.m, draw.r[j], draw.theta[j]);
        if (!CloseRel(draw.density[j], want, 1e-12)) {
          ++dens_fail;
        }
      }
      fmt::print("  fp64 weight fail={} {}\n", dens_fail,
                 dens_fail == 0 ? "pass" : "FAIL");
      ok = dens_fail == 0 && ok;
    }
  }
  return ok;
}

}  // namespace

int RunVerify(bool full) {
  const bool goldens = CheckDeviceGoldens();
  const bool invert = CheckInvertAb();
  const int n = full ? 10000000 : 100000;
  const double max_d = full ? 5.0e-4 : 0.01;
  const bool fp64 = CheckKernel(KernelKind::Fp64, n, max_d, full);
  const bool fp32 = CheckKernel(KernelKind::Fp32, n, max_d, full);
  const bool fast = CheckKernel(KernelKind::Fp32Fast, full ? 100000 : n, 0.01,
                                false);
  const bool ok = goldens && invert && fp64 && fp32 && fast;
  fmt::print("naive-cuda verify-{} {}\n", full ? "full" : "fast",
             ok ? "pass" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace qmc::naive::verify
