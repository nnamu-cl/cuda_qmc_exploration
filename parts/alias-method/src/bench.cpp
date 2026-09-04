#include "parts/alias-method/src/sampler.h"

#include <ranges>
#include <vector>

#include <fmt/format.h>

#include "cpu-reference/special.h"
#include "harness/device_memory.h"
#include "harness/env.h"
#include "harness/results.h"
#include "harness/timing.h"
#include "parts/naive-cuda/src/sampler.h"

namespace qmc::alias {
namespace {

constexpr std::uint64_t kSeed = 0xC0FFEEULL;

int MAbs(int m) { return m < 0 ? -m : m; }

double BytesPerSample(KernelKind kind) {
  if (kind == KernelKind::Split) {
    return 124.0;
  }
  return 2.0 * static_cast<double>(qmc::naive::XorwowStateBytes()) + 16.0;
}

struct DeviceRun {
  qmc::harness::DeviceUnique<unsigned char> states;
  qmc::harness::DeviceUnique<AliasBin> radial;
  qmc::harness::DeviceUnique<AliasBin> theta;
  qmc::harness::DeviceUnique<float> x;
  qmc::harness::DeviceUnique<float> y;
  qmc::harness::DeviceUnique<float> z;
  qmc::harness::DeviceUnique<float> w;
  qmc::harness::DeviceUnique<float> r;
  qmc::harness::DeviceUnique<float> th;
  qmc::harness::DeviceUnique<float> phi;
  qmc::harness::DeviceUnique<PackedXyzw> packed;
  int n_radial = 0;
  int n_theta = 0;
  int n = 0;
  int principal = 0;
  int l = 0;
  int m_abs = 0;
  float radial_norm = 0.0f;
  float y_norm = 0.0f;
  KernelKind kind = KernelKind::Linear;
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
  const auto radial = BuildRadialAlias(sampler);
  const auto theta = BuildThetaAlias(sampler);
  run.n_radial = static_cast<int>(radial.size());
  run.n_theta = static_cast<int>(theta.size());
  run.radial = qmc::harness::DeviceFromHost(std::span<const AliasBin>(radial));
  run.theta = qmc::harness::DeviceFromHost(std::span<const AliasBin>(theta));
  run.states = qmc::harness::DeviceAlloc<unsigned char>(
      static_cast<size_t>(n) * qmc::naive::XorwowStateBytes());
  const bool split = kind == KernelKind::Split;
  const bool packed = kind == KernelKind::Float4;
  if (packed) {
    run.packed = qmc::harness::DeviceAlloc<PackedXyzw>(static_cast<size_t>(n));
  } else {
    run.x = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.y = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.z = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.w = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
  }
  if (with_spherical || split) {
    run.r = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.th = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.phi = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
  }
  return run;
}

void Launch(const DeviceRun& run) {
  if (run.kind == KernelKind::Split) {
    LaunchAliasSplit(run.radial.get(), run.n_radial, run.theta.get(),
                     run.n_theta, run.states.get(), run.x.get(), run.y.get(),
                     run.z.get(), run.w.get(), run.r.get(), run.th.get(),
                     run.phi.get(), run.n, run.principal, run.l, run.m_abs,
                     run.radial_norm, run.y_norm, run.cfg.threads);
    return;
  }
  const bool linear = run.kind != KernelKind::Uniform;
  LaunchAliasSample(run.radial.get(), run.n_radial, run.theta.get(),
                    run.n_theta, run.states.get(), run.x.get(), run.y.get(),
                    run.z.get(), run.w.get(), run.r.get(), run.th.get(),
                    run.phi.get(), run.packed.get(), run.n, run.principal,
                    run.l, run.m_abs, run.radial_norm, run.y_norm, linear,
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

void PullPacked(GpuDraw* out, const qmc::harness::DeviceUnique<PackedXyzw>& src,
                int n) {
  const std::vector<PackedXyzw> tmp =
      qmc::harness::HostFromDevice(src, static_cast<size_t>(n));
  out->x.resize(tmp.size());
  out->y.resize(tmp.size());
  out->z.resize(tmp.size());
  out->density.resize(tmp.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    out->x[i] = static_cast<double>(tmp[i].x);
    out->y[i] = static_cast<double>(tmp[i].y);
    out->z[i] = static_cast<double>(tmp[i].z);
    out->density[i] = static_cast<double>(tmp[i].w);
  }
}

}  // namespace

const char* KernelName(KernelKind kind) {
  switch (kind) {
    case KernelKind::Uniform:
      return "alias_uniform";
    case KernelKind::Linear:
      return "alias_linear";
    case KernelKind::Float4:
      return "alias_float4";
    case KernelKind::Split:
      return "alias_split";
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
  if (kind == KernelKind::Float4) {
    PullPacked(out, run.packed, n);
  } else {
    Pull(&out->x, run.x, n);
    Pull(&out->y, run.y, n);
    Pull(&out->z, run.z, n);
    Pull(&out->density, run.w, n);
  }
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
  timing.bytes_per_sample = BytesPerSample(kind);
  return timing;
}

int RunAliasBench(const BenchOptions& opt) {
  const qmc::cpu::QuantumNumbers qn{opt.principal, opt.l, opt.m};
  const KernelTiming timing = TimeGpu(opt.kernel, qn, kSeed, opt.n, opt.launch);
  qmc::harness::BenchmarkRecord rec;
  rec.part = "alias-method";
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
      "{} n={} threads={} median_ms={:.3f} Msamples/s={:.3f} GB/s={:.3f} "
      "setup_ms={:.3f} locked={}\n",
      rec.kernel, opt.n, opt.launch.threads, rec.median_ms,
      rec.units_per_s / 1e6, rec.achieved_gbps, timing.setup_ms,
      rec.env.clocks_locked ? 1 : 0);

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

}  // namespace qmc::alias
