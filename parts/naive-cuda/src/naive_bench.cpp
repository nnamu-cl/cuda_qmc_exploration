#include "parts/naive-cuda/src/sampler.h"

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

namespace qmc::naive {
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

double BytesPerSample(KernelKind kind) {
  const double state = 2.0 * static_cast<double>(XorwowStateBytes());
  if (kind == KernelKind::Fp64) {
    return state + 23.0 * 8.0 + 32.0 + 32.0;
  }
  return state + 23.0 * 4.0 + 16.0 + 16.0;
}

struct DeviceRun {
  qmc::harness::DeviceUnique<unsigned char> states;
  qmc::harness::DeviceUnique<double> r_nodes64;
  qmc::harness::DeviceUnique<double> r_cdf64;
  qmc::harness::DeviceUnique<double> theta_nodes64;
  qmc::harness::DeviceUnique<double> theta_cdf64;
  qmc::harness::DeviceUnique<float> r_nodes32;
  qmc::harness::DeviceUnique<float> r_cdf32;
  qmc::harness::DeviceUnique<float> theta_nodes32;
  qmc::harness::DeviceUnique<float> theta_cdf32;
  qmc::harness::DeviceUnique<double> x64;
  qmc::harness::DeviceUnique<double> y64;
  qmc::harness::DeviceUnique<double> z64;
  qmc::harness::DeviceUnique<double> w64;
  qmc::harness::DeviceUnique<double> r64;
  qmc::harness::DeviceUnique<double> th64;
  qmc::harness::DeviceUnique<double> phi64;
  qmc::harness::DeviceUnique<float> x32;
  qmc::harness::DeviceUnique<float> y32;
  qmc::harness::DeviceUnique<float> z32;
  qmc::harness::DeviceUnique<float> w32;
  qmc::harness::DeviceUnique<float> r32;
  qmc::harness::DeviceUnique<float> th32;
  qmc::harness::DeviceUnique<float> phi32;
  int n_radial = 0;
  int n_theta = 0;
  int n = 0;
  int principal = 0;
  int l = 0;
  int m_abs = 0;
  double radial_norm = 0.0;
  double y_norm = 0.0;
  KernelKind kind = KernelKind::Fp64;
};

DeviceRun Prepare(KernelKind kind, qmc::cpu::QuantumNumbers qn, int n,
                  bool with_spherical) {
  DeviceRun run;
  run.kind = kind;
  run.n = n;
  run.principal = qn.n;
  run.l = qn.l;
  run.m_abs = MAbs(qn.m);
  run.radial_norm = qmc::cpu::RadialNorm(qn.n, qn.l);
  run.y_norm = qmc::cpu::SphericalHarmonicNorm(qn.l, qn.m);
  const qmc::cpu::HydrogenSampler sampler(qn);
  run.n_radial = static_cast<int>(sampler.RadialNodes().size());
  run.n_theta = static_cast<int>(sampler.ThetaNodes().size());
  run.states = qmc::harness::DeviceAlloc<unsigned char>(
      static_cast<size_t>(n) * XorwowStateBytes());

  if (kind == KernelKind::Fp64) {
    run.r_nodes64 = qmc::harness::DeviceFromHost(sampler.RadialNodes());
    run.r_cdf64 = qmc::harness::DeviceFromHost(sampler.RadialCdf());
    run.theta_nodes64 = qmc::harness::DeviceFromHost(sampler.ThetaNodes());
    run.theta_cdf64 = qmc::harness::DeviceFromHost(sampler.ThetaCdf());
    run.x64 = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n));
    run.y64 = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n));
    run.z64 = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n));
    run.w64 = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n));
    if (with_spherical) {
      run.r64 = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n));
      run.th64 = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n));
      run.phi64 = qmc::harness::DeviceAlloc<double>(static_cast<size_t>(n));
    }
  } else {
    const auto r_nodes = Narrow<float>(sampler.RadialNodes());
    const auto r_cdf = Narrow<float>(sampler.RadialCdf());
    const auto th_nodes = Narrow<float>(sampler.ThetaNodes());
    const auto th_cdf = Narrow<float>(sampler.ThetaCdf());
    run.r_nodes32 = qmc::harness::DeviceFromHost(std::span<const float>(r_nodes));
    run.r_cdf32 = qmc::harness::DeviceFromHost(std::span<const float>(r_cdf));
    run.theta_nodes32 =
        qmc::harness::DeviceFromHost(std::span<const float>(th_nodes));
    run.theta_cdf32 =
        qmc::harness::DeviceFromHost(std::span<const float>(th_cdf));
    run.x32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.y32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.z32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.w32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    if (with_spherical) {
      run.r32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
      run.th32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
      run.phi32 = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    }
  }
  return run;
}

void Launch(const DeviceRun& run) {
  if (run.kind == KernelKind::Fp64) {
    LaunchSampleFp64(run.r_nodes64.get(), run.r_cdf64.get(), run.n_radial,
                     run.theta_nodes64.get(), run.theta_cdf64.get(),
                     run.n_theta, run.states.get(), run.x64.get(),
                     run.y64.get(), run.z64.get(), run.w64.get(),
                     run.r64.get(), run.th64.get(), run.phi64.get(), run.n,
                     run.principal, run.l, run.m_abs, run.radial_norm,
                     run.y_norm);
    return;
  }
  const float radial_norm = static_cast<float>(run.radial_norm);
  const float y_norm = static_cast<float>(run.y_norm);
  if (run.kind == KernelKind::Fp32) {
    LaunchSampleFp32(run.r_nodes32.get(), run.r_cdf32.get(), run.n_radial,
                     run.theta_nodes32.get(), run.theta_cdf32.get(),
                     run.n_theta, run.states.get(), run.x32.get(),
                     run.y32.get(), run.z32.get(), run.w32.get(),
                     run.r32.get(), run.th32.get(), run.phi32.get(), run.n,
                     run.principal, run.l, run.m_abs, radial_norm, y_norm);
    return;
  }
  LaunchSampleFp32Fast(run.r_nodes32.get(), run.r_cdf32.get(), run.n_radial,
                       run.theta_nodes32.get(), run.theta_cdf32.get(),
                       run.n_theta, run.states.get(), run.x32.get(),
                       run.y32.get(), run.z32.get(), run.w32.get(),
                       run.r32.get(), run.th32.get(), run.phi32.get(), run.n,
                       run.principal, run.l, run.m_abs, radial_norm, y_norm);
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
    case KernelKind::Fp64:
      return "naive_fp64";
    case KernelKind::Fp32:
      return "naive_fp32";
    case KernelKind::Fp32Fast:
      return "naive_fp32_fast";
  }
  return "unknown";
}

void DrawGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn, std::uint64_t seed,
             int n, GpuDraw* out) {
  DeviceRun run = Prepare(kind, qn, n, true);
  SetupXorwowStates(run.states.get(), seed, n);
  qmc::harness::CheckCudaCall(cudaDeviceSynchronize(), "setup sync");
  Launch(run);
  qmc::harness::CheckCudaCall(cudaDeviceSynchronize(), "sample sync");
  if (kind == KernelKind::Fp64) {
    Pull(&out->x, run.x64, n);
    Pull(&out->y, run.y64, n);
    Pull(&out->z, run.z64, n);
    Pull(&out->density, run.w64, n);
    Pull(&out->r, run.r64, n);
    Pull(&out->theta, run.th64, n);
    Pull(&out->phi, run.phi64, n);
  } else {
    Pull(&out->x, run.x32, n);
    Pull(&out->y, run.y32, n);
    Pull(&out->z, run.z32, n);
    Pull(&out->density, run.w32, n);
    Pull(&out->r, run.r32, n);
    Pull(&out->theta, run.th32, n);
    Pull(&out->phi, run.phi32, n);
  }
}

KernelTiming TimeGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn,
                     std::uint64_t seed, int n) {
  DeviceRun run = Prepare(kind, qn, n, false);
  const auto setup = qmc::harness::TimeCudaLaunch({1, 1}, [&] {
    SetupXorwowStates(run.states.get(), seed, n);
  });
  SetupXorwowStates(run.states.get(), seed, n);
  qmc::harness::CheckCudaCall(cudaDeviceSynchronize(), "setup before timed");
  const auto sample = qmc::harness::TimeCudaLaunch({}, [&] { Launch(run); });
  KernelTiming timing;
  timing.setup_ms = setup.median_ms;
  timing.reps_ms = sample.reps_ms;
  timing.median_ms = sample.median_ms;
  timing.bytes_per_sample = BytesPerSample(kind);
  return timing;
}

int RunNaiveBench(const NaiveBenchOptions& opt) {
  const qmc::cpu::QuantumNumbers qn{opt.principal, opt.l, opt.m};
  const KernelTiming timing = TimeGpu(opt.kernel, qn, kSeed, opt.n);
  qmc::harness::BenchmarkRecord rec;
  rec.part = "naive-cuda";
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
      "{} n={} median_ms={:.3f} Msamples/s={:.3f} GB/s={:.3f} setup_ms={:.3f} "
      "locked={}\n",
      rec.kernel, opt.n, rec.median_ms, rec.units_per_s / 1e6, rec.achieved_gbps,
      timing.setup_ms, rec.env.clocks_locked ? 1 : 0);

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

}  // namespace qmc::naive
