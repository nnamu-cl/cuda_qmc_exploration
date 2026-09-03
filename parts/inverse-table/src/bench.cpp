#include "parts/inverse-table/src/sampler.h"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#include <fmt/format.h>

#include "cpu-reference/special.h"
#include "harness/device_memory.h"
#include "harness/env.h"
#include "harness/results.h"
#include "harness/timing.h"
#include "parts/naive-cuda/src/sampler.h"

namespace qmc::inverse {
namespace {

constexpr std::uint64_t kSeed = 0xC0FFEEULL;

template <class T>
std::vector<T> Narrow(std::span<const double> src) {
  std::vector<T> out(src.size());
  std::ranges::transform(src, out.begin(),
                         [](double value) { return static_cast<T>(value); });
  return out;
}

int MAbs(int m) { return m < 0 ? -m : m; }

double BytesPerSample() {
  return 2.0 * static_cast<double>(qmc::naive::XorwowStateBytes()) + 16.0;
}

struct DeviceRun {
  qmc::harness::DeviceUnique<unsigned char> states;
  qmc::harness::DeviceUnique<float> r_nodes;
  qmc::harness::DeviceUnique<float> r_cdf;
  qmc::harness::DeviceUnique<float> theta_nodes;
  qmc::harness::DeviceUnique<float> theta_cdf;
  qmc::harness::DeviceUnique<float> r_table;
  qmc::harness::DeviceUnique<float> theta_table;
  qmc::harness::DeviceUnique<std::uint8_t> r_flags;
  qmc::harness::DeviceUnique<std::uint8_t> theta_flags;
  qmc::harness::DeviceUnique<float> x;
  qmc::harness::DeviceUnique<float> y;
  qmc::harness::DeviceUnique<float> z;
  qmc::harness::DeviceUnique<float> w;
  qmc::harness::DeviceUnique<float> r;
  qmc::harness::DeviceUnique<float> th;
  qmc::harness::DeviceUnique<float> phi;
  int n_radial = 0;
  int n_theta = 0;
  int k_radial = 0;
  int k_theta = 0;
  int n = 0;
  int principal = 0;
  int l = 0;
  int m_abs = 0;
  float radial_norm = 0.0f;
  float y_norm = 0.0f;
  KernelKind kind = KernelKind::SharedCdf;
  LaunchConfig cfg;
};

DeviceRun Prepare(KernelKind kind, qmc::cpu::QuantumNumbers qn, int n,
                  bool with_spherical, const LaunchConfig& cfg) {
  DeviceRun run;
  run.kind = kind;
  run.cfg = cfg;
  run.n = n;
  run.principal = qn.n;
  run.l = qn.l;
  run.m_abs = MAbs(qn.m);
  run.radial_norm = static_cast<float>(qmc::cpu::RadialNorm(qn.n, qn.l));
  run.y_norm = static_cast<float>(qmc::cpu::SphericalHarmonicNorm(qn.l, qn.m));
  const qmc::cpu::HydrogenSampler sampler(qn);
  run.n_radial = static_cast<int>(sampler.RadialNodes().size());
  run.n_theta = static_cast<int>(sampler.ThetaNodes().size());
  run.k_radial = cfg.table_k_radial;
  run.k_theta = cfg.table_k_theta;
  run.states = qmc::harness::DeviceAlloc<unsigned char>(
      static_cast<size_t>(n) * qmc::naive::XorwowStateBytes());
  run.x = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
  run.y = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
  run.z = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
  run.w = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
  if (with_spherical) {
    run.r = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.th = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.phi = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
  }
  if (kind == KernelKind::SharedCdf) {
    const auto r_nodes = Narrow<float>(sampler.RadialNodes());
    const auto r_cdf = Narrow<float>(sampler.RadialCdf());
    const auto th_nodes = Narrow<float>(sampler.ThetaNodes());
    const auto th_cdf = Narrow<float>(sampler.ThetaCdf());
    run.r_nodes = qmc::harness::DeviceFromHost(std::span<const float>(r_nodes));
    run.r_cdf = qmc::harness::DeviceFromHost(std::span<const float>(r_cdf));
    run.theta_nodes =
        qmc::harness::DeviceFromHost(std::span<const float>(th_nodes));
    run.theta_cdf = qmc::harness::DeviceFromHost(std::span<const float>(th_cdf));
  } else {
    const auto r_table = BuildRadialQuantile(sampler, run.k_radial);
    const auto th_table = BuildThetaQuantile(sampler, run.k_theta);
    run.r_table = qmc::harness::DeviceFromHost(std::span<const float>(r_table));
    run.theta_table =
        qmc::harness::DeviceFromHost(std::span<const float>(th_table));
    if (kind == KernelKind::InverseNodes) {
      const auto r_flags =
          BuildJumpFlags(sampler.RadialNodes(), sampler.RadialCdf(), r_table);
      const auto th_flags =
          BuildJumpFlags(sampler.ThetaNodes(), sampler.ThetaCdf(), th_table);
      run.r_flags =
          qmc::harness::DeviceFromHost(std::span<const std::uint8_t>(r_flags));
      run.theta_flags =
          qmc::harness::DeviceFromHost(std::span<const std::uint8_t>(th_flags));
    }
  }
  return run;
}

void Launch(const DeviceRun& run) {
  if (run.kind == KernelKind::SharedCdf) {
    LaunchSharedCdf(run.r_nodes.get(), run.r_cdf.get(), run.n_radial,
                    run.theta_nodes.get(), run.theta_cdf.get(), run.n_theta,
                    run.states.get(), run.x.get(), run.y.get(), run.z.get(),
                    run.w.get(), run.r.get(), run.th.get(), run.phi.get(),
                    run.n, run.principal, run.l, run.m_abs, run.radial_norm,
                    run.y_norm, run.cfg.threads, run.cfg.cache_nodes);
    return;
  }
  LaunchInverseLookup(run.r_table.get(), run.k_radial, run.theta_table.get(),
                      run.k_theta, run.r_flags.get(), run.theta_flags.get(),
                      run.states.get(), run.x.get(), run.y.get(), run.z.get(),
                      run.w.get(), run.r.get(), run.th.get(), run.phi.get(),
                      run.n, run.principal, run.l, run.m_abs, run.radial_norm,
                      run.y_norm, run.kind == KernelKind::InverseNodes,
                      run.cfg.threads);
}

template <class T>
void Pull(std::vector<double>* dst, const qmc::harness::DeviceUnique<T>& src,
          int n) {
  const std::vector<T> tmp =
      qmc::harness::HostFromDevice(src, static_cast<size_t>(n));
  dst->resize(tmp.size());
  std::ranges::transform(tmp, dst->begin(),
                         [](T value) { return static_cast<double>(value); });
}

}  // namespace

const char* KernelName(KernelKind kind) {
  switch (kind) {
    case KernelKind::SharedCdf:
      return "shared_cdf";
    case KernelKind::InverseTable:
      return "inverse_table";
    case KernelKind::InverseNodes:
      return "inverse_nodes";
  }
  return "unknown";
}

void DrawGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn, std::uint64_t seed,
             int n, GpuDraw* out, const LaunchConfig& cfg) {
  DeviceRun run = Prepare(kind, qn, n, true, cfg);
  qmc::naive::SetupXorwowStates(run.states.get(), seed, n);
  qmc::harness::CheckCudaCall(cudaDeviceSynchronize(), "setup sync");
  Launch(run);
  qmc::harness::CheckCudaCall(cudaDeviceSynchronize(), "sample sync");
  Pull(&out->x, run.x, n);
  Pull(&out->y, run.y, n);
  Pull(&out->z, run.z, n);
  Pull(&out->density, run.w, n);
  Pull(&out->r, run.r, n);
  Pull(&out->theta, run.th, n);
  Pull(&out->phi, run.phi, n);
}

KernelTiming TimeGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn,
                     std::uint64_t seed, int n, const LaunchConfig& cfg) {
  DeviceRun run = Prepare(kind, qn, n, false, cfg);
  const auto setup = qmc::harness::TimeCudaLaunch({1, 1}, [&] {
    qmc::naive::SetupXorwowStates(run.states.get(), seed, n);
  });
  qmc::naive::SetupXorwowStates(run.states.get(), seed, n);
  qmc::harness::CheckCudaCall(cudaDeviceSynchronize(), "setup before timed");
  const auto sample = qmc::harness::TimeCudaLaunch({}, [&] { Launch(run); });
  KernelTiming timing;
  timing.setup_ms = setup.median_ms;
  timing.reps_ms = sample.reps_ms;
  timing.median_ms = sample.median_ms;
  timing.bytes_per_sample = BytesPerSample();
  if (kind == KernelKind::SharedCdf) {
    timing.occupancy_blocks = SharedCdfOccupancyBlocks(
        cfg.threads, cfg.cache_nodes, run.n_radial, run.n_theta);
  }
  return timing;
}

int RunInverseBench(const BenchOptions& opt) {
  const qmc::cpu::QuantumNumbers qn{opt.principal, opt.l, opt.m};
  const KernelTiming timing =
      TimeGpu(opt.kernel, qn, kSeed, opt.n, opt.launch);
  qmc::harness::BenchmarkRecord rec;
  rec.part = "inverse-table";
  rec.kernel = KernelName(opt.kernel);
  rec.workload = "hydrogen_samples";
  rec.count = opt.n;
  rec.quantum_numbers = {opt.principal, opt.l, opt.m};
  rec.reps_ms = timing.reps_ms;
  rec.median_ms = timing.median_ms;
  rec.bytes_per_unit = timing.bytes_per_sample;
  rec.units_per_s = static_cast<double>(opt.n) / (timing.median_ms * 1e-3);
  rec.achieved_gbps = (static_cast<double>(opt.n) * rec.bytes_per_unit) /
                      (timing.median_ms * 1e-3) / 1e9;
  rec.env = qmc::harness::CaptureEnv();

  fmt::print(
      "{} n={} threads={} cache_nodes={} median_ms={:.3f} Msamples/s={:.3f} "
      "GB/s={:.3f} setup_ms={:.3f} occ_blocks={} locked={}\n",
      rec.kernel, opt.n, opt.launch.threads, opt.launch.cache_nodes ? 1 : 0,
      rec.median_ms, rec.units_per_s / 1e6, rec.achieved_gbps, timing.setup_ms,
      timing.occupancy_blocks, rec.env.clocks_locked ? 1 : 0);

  if (!opt.out_path.empty()) {
    if (const auto written =
            qmc::harness::WriteJson(opt.out_path, rec, !opt.scratch);
        !written) {
      fmt::print(stderr, "{}\n", written.error());
      return 1;
    }
  }
  return 0;
}

}  // namespace qmc::inverse
