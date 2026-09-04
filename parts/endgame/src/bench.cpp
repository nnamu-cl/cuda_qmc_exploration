#include "parts/endgame/src/sampler.h"

#include <algorithm>
#include <ranges>
#include <vector>

#include <fmt/format.h>

#include "cpu-reference/special.h"
#include "harness/device_memory.h"
#include "harness/dump.h"
#include "harness/env.h"
#include "harness/results.h"
#include "harness/timing.h"

namespace qmc::endgame {
namespace {

constexpr std::uint64_t kSeed = 0xC0FFEEULL;

int MAbs(int m) { return m < 0 ? -m : m; }

double BytesPerSample(KernelKind kind) {
  if (kind == KernelKind::ThrustAlias) {
    return 44.0;
  }
  return 16.0;
}

void SetPersistWindow(void* pointer, std::size_t bytes) {
  cudaAccessPolicyWindow window{};
  window.base_ptr = pointer;
  window.num_bytes = bytes;
  window.hitRatio = 1.0f;
  window.hitProp = cudaAccessPropertyPersisting;
  window.missProp = cudaAccessPropertyStreaming;
  cudaStreamAttrValue value;
  value.accessPolicyWindow = window;
  const cudaError_t err = cudaStreamSetAttribute(
      static_cast<cudaStream_t>(0), cudaStreamAttributeAccessPolicyWindow,
      &value);
  if (err != cudaSuccess) {
    fmt::print(stderr, "persist window skipped: {}\n", cudaGetErrorString(err));
    static_cast<void>(cudaGetLastError());
  }
}

struct DeviceRun {
  qmc::harness::DeviceUnique<qmc::alias::AliasBin> radial;
  qmc::harness::DeviceUnique<qmc::alias::AliasBin> theta;
  qmc::harness::DeviceUnique<qmc::alias::PackedXyzw> packed;
  qmc::harness::DeviceUnique<float> x;
  qmc::harness::DeviceUnique<float> y;
  qmc::harness::DeviceUnique<float> z;
  qmc::harness::DeviceUnique<float> w;
  qmc::harness::DeviceUnique<float> r;
  qmc::harness::DeviceUnique<float> th;
  qmc::harness::DeviceUnique<float> phi;
  qmc::harness::DeviceUnique<float> uniforms;
  int n_radial = 0;
  int n_theta = 0;
  int n = 0;
  int principal = 0;
  int l = 0;
  int m_abs = 0;
  float radial_norm = 0.0f;
  float y_norm = 0.0f;
  KernelKind kind = KernelKind::Philox;
  LaunchConfig cfg;
  std::uint64_t seed = kSeed;
};

DeviceRun Prepare(KernelKind kind, qmc::cpu::QuantumNumbers qn, int n,
                  bool with_spherical, const LaunchConfig& cfg,
                  std::uint64_t seed) {
  DeviceRun run;
  run.kind = kind;
  run.cfg = cfg;
  run.n = n;
  run.seed = seed;
  run.principal = qn.n;
  run.l = qn.l;
  run.m_abs = MAbs(qn.m);
  run.radial_norm = static_cast<float>(qmc::cpu::RadialNorm(qn.n, qn.l));
  run.y_norm = static_cast<float>(qmc::cpu::SphericalHarmonicNorm(qn.l, qn.m));
  const qmc::cpu::HydrogenSampler sampler(qn);
  const auto radial = qmc::alias::BuildRadialAlias(sampler);
  const auto theta = qmc::alias::BuildThetaAlias(sampler);
  run.n_radial = static_cast<int>(radial.size());
  run.n_theta = static_cast<int>(theta.size());
  run.radial =
      qmc::harness::DeviceFromHost(std::span<const qmc::alias::AliasBin>(radial));
  run.theta =
      qmc::harness::DeviceFromHost(std::span<const qmc::alias::AliasBin>(theta));
  if (kind == KernelKind::ThrustAlias) {
    run.uniforms = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n) * 7);
  }
  if (with_spherical) {
    run.x = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.y = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.z = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.w = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.r = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.th = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.phi = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
  } else {
    run.packed =
        qmc::harness::DeviceAlloc<qmc::alias::PackedXyzw>(static_cast<size_t>(n));
  }
  return run;
}

void Launch(const DeviceRun& run) {
  if (run.cfg.persist_tables) {
    SetPersistWindow(run.radial.get(),
                     static_cast<std::size_t>(run.n_radial) *
                         sizeof(qmc::alias::AliasBin));
  }
  if (run.kind == KernelKind::ThrustAlias) {
    LaunchThrustAlias(run.radial.get(), run.n_radial, run.theta.get(),
                      run.n_theta, run.uniforms.get(), run.packed.get(),
                      run.x.get(), run.y.get(), run.z.get(), run.w.get(),
                      run.r.get(), run.th.get(), run.phi.get(), run.n,
                      run.principal, run.l, run.m_abs, run.radial_norm,
                      run.y_norm, run.seed);
    return;
  }
  if (run.kind == KernelKind::PhiloxIlp) {
    LaunchPhiloxIlp(run.radial.get(), run.n_radial, run.theta.get(),
                    run.n_theta, run.packed.get(), run.x.get(), run.y.get(),
                    run.z.get(), run.w.get(), run.r.get(), run.th.get(),
                    run.phi.get(), run.n, run.principal, run.l, run.m_abs,
                    run.radial_norm, run.y_norm, run.seed, run.cfg);
    return;
  }
  LaunchPhiloxSample(run.radial.get(), run.n_radial, run.theta.get(),
                     run.n_theta, run.packed.get(), run.x.get(), run.y.get(),
                     run.z.get(), run.w.get(), run.r.get(), run.th.get(),
                     run.phi.get(), run.n, run.principal, run.l, run.m_abs,
                     run.radial_norm, run.y_norm, run.seed, run.cfg.threads);
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
    case KernelKind::Philox:
      return "philox";
    case KernelKind::PhiloxIlp:
      return "philox_ilp";
    case KernelKind::ThrustAlias:
      return "thrust_alias";
  }
  return "unknown";
}

void DrawGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn, std::uint64_t seed,
             int n, GpuDraw* out, const LaunchConfig& cfg) {
  DeviceRun run = Prepare(kind, qn, n, true, cfg, seed);
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
  DeviceRun run = Prepare(kind, qn, n, false, cfg, seed);
  const auto sample = qmc::harness::TimeCudaLaunch({}, [&] { Launch(run); });
  KernelTiming timing;
  timing.setup_ms = 0.0;
  timing.reps_ms = sample.reps_ms;
  timing.median_ms = sample.median_ms;
  timing.bytes_per_sample = BytesPerSample(kind);
  if (kind == KernelKind::PhiloxIlp) {
    timing.occupancy_blocks =
        IlpOccupancyBlocks(cfg.samples_per_thread, cfg.threads);
    timing.grid_blocks = cfg.grid_blocks > 0
                             ? cfg.grid_blocks
                             : IlpGridBlocks(cfg.samples_per_thread, cfg.threads);
  } else {
    timing.grid_blocks = (n + cfg.threads - 1) / cfg.threads;
  }
  return timing;
}

int RunEndgameBench(const BenchOptions& opt) {
  const qmc::cpu::QuantumNumbers qn{opt.principal, opt.l, opt.m};
  const KernelTiming timing = TimeGpu(opt.kernel, qn, kSeed, opt.n, opt.launch);
  qmc::harness::BenchmarkRecord rec;
  rec.part = "endgame";
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
      "{} n={} threads={} S={} median_ms={:.3f} Msamples/s={:.3f} GB/s={:.3f} "
      "setup_ms={:.3f} occ_blocks={} locked={}\n",
      rec.kernel, opt.n, opt.launch.threads, opt.launch.samples_per_thread,
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

int RunSampleDump(const std::string& prefix, int n, qmc::cpu::QuantumNumbers qn,
                  std::uint64_t seed) {
  GpuDraw draw;
  DrawGpu(KernelKind::Philox, qn, seed, n, &draw);
  std::vector<float> xyzw(static_cast<size_t>(n) * 4);
  for (int i = 0; i < n; ++i) {
    xyzw[static_cast<size_t>(i) * 4 + 0] = static_cast<float>(draw.x[static_cast<size_t>(i)]);
    xyzw[static_cast<size_t>(i) * 4 + 1] = static_cast<float>(draw.y[static_cast<size_t>(i)]);
    xyzw[static_cast<size_t>(i) * 4 + 2] = static_cast<float>(draw.z[static_cast<size_t>(i)]);
    xyzw[static_cast<size_t>(i) * 4 + 3] =
        static_cast<float>(draw.density[static_cast<size_t>(i)]);
  }
  qmc::harness::SampleDumpMeta meta;
  meta.n = static_cast<std::uint64_t>(n);
  meta.nlm = {qn.n, qn.l, qn.m};
  meta.seed = seed;
  if (const auto written = qmc::harness::WriteSampleDump(prefix, xyzw, meta);
      !written) {
    fmt::print(stderr, "{}\n", written.error());
    return 1;
  }
  fmt::print("dumped {} samples to {}\n", n, prefix);
  return 0;
}

}  // namespace qmc::endgame
