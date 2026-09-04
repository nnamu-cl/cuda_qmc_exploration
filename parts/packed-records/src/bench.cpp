#include "parts/packed-records/src/sampler.h"

#include <algorithm>
#include <ranges>
#include <vector>

#include <fmt/format.h>

#include "cpu-reference/special.h"
#include "harness/device_memory.h"
#include "harness/env.h"
#include "harness/results.h"
#include "harness/timing.h"

namespace qmc::packed {
namespace {

constexpr std::uint64_t kSeed = 0xC0FFEEULL;

int MAbs(int m) { return m < 0 ? -m : m; }

// Only the bytes that actually reach DRAM: one 16 B float4 per sample. Table
// traffic is L1/L2 resident and is not modelled here — an earlier part in this
// repo published modelled GB/s above the copy roof and that must not recur.
constexpr double kBytesPerSample = 16.0;

struct DeviceRun {
  qmc::harness::DeviceUnique<AliasDraw> radial_draw;
  qmc::harness::DeviceUnique<AliasDraw> theta_draw;
  qmc::harness::DeviceUnique<BinUniform> radial_uniform;
  qmc::harness::DeviceUnique<BinUniform> theta_uniform;
  qmc::harness::DeviceUnique<BinLinear> radial_linear;
  qmc::harness::DeviceUnique<BinLinear> theta_linear;
  qmc::harness::DeviceUnique<qmc::alias::PackedXyzw> packed;
  qmc::harness::DeviceUnique<float> x;
  qmc::harness::DeviceUnique<float> y;
  qmc::harness::DeviceUnique<float> z;
  qmc::harness::DeviceUnique<float> w;
  qmc::harness::DeviceUnique<float> r;
  qmc::harness::DeviceUnique<float> th;
  qmc::harness::DeviceUnique<float> phi;
  int n_radial = 0;
  int n_theta = 0;
  KernelKind kind = KernelKind::Packed;
  LaunchConfig cfg;
  Params params;
  Outputs out;
};

DeviceRun Prepare(KernelKind kind, qmc::cpu::QuantumNumbers qn, int n,
                  bool with_spherical, const LaunchConfig& cfg,
                  std::uint64_t seed) {
  DeviceRun run;
  run.kind = kind;
  run.cfg = cfg;
  run.params.n = n;
  run.params.seed = seed;
  run.params.principal = qn.n;
  run.params.l = qn.l;
  run.params.m_abs = MAbs(qn.m);
  run.params.radial_norm = static_cast<float>(qmc::cpu::RadialNorm(qn.n, qn.l));
  run.params.y_norm =
      static_cast<float>(qmc::cpu::SphericalHarmonicNorm(qn.l, qn.m));

  const qmc::cpu::HydrogenSampler sampler(qn);
  const auto radial_bins = qmc::alias::BuildRadialAlias(sampler);
  const auto theta_bins = qmc::alias::BuildThetaAlias(sampler);
  const PackedTable radial = SplitAliasBins(radial_bins);
  const PackedTable theta = SplitAliasBins(theta_bins);
  run.n_radial = static_cast<int>(radial.draw.size());
  run.n_theta = static_cast<int>(theta.draw.size());

  run.radial_draw =
      qmc::harness::DeviceFromHost(std::span<const AliasDraw>(radial.draw));
  run.theta_draw =
      qmc::harness::DeviceFromHost(std::span<const AliasDraw>(theta.draw));
  if (kind == KernelKind::PackedLinear) {
    run.radial_linear =
        qmc::harness::DeviceFromHost(std::span<const BinLinear>(radial.linear));
    run.theta_linear =
        qmc::harness::DeviceFromHost(std::span<const BinLinear>(theta.linear));
  } else {
    run.radial_uniform = qmc::harness::DeviceFromHost(
        std::span<const BinUniform>(radial.uniform));
    run.theta_uniform =
        qmc::harness::DeviceFromHost(std::span<const BinUniform>(theta.uniform));
  }

  if (with_spherical) {
    run.x = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.y = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.z = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.w = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.r = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.th = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.phi = qmc::harness::DeviceAlloc<float>(static_cast<size_t>(n));
    run.out.x = run.x.get();
    run.out.y = run.y.get();
    run.out.z = run.z.get();
    run.out.density = run.w.get();
    run.out.r = run.r.get();
    run.out.theta = run.th.get();
    run.out.phi = run.phi.get();
  } else {
    run.packed =
        qmc::harness::DeviceAlloc<qmc::alias::PackedXyzw>(static_cast<size_t>(n));
    run.out.packed = run.packed.get();
  }
  return run;
}

void Launch(const DeviceRun& run) {
  switch (run.kind) {
    case KernelKind::PackedLinear:
      LaunchPackedLinear(run.radial_draw.get(), run.radial_linear.get(),
                         run.n_radial, run.theta_draw.get(),
                         run.theta_linear.get(), run.n_theta, run.out,
                         run.params, run.cfg);
      return;
    case KernelKind::PackedShared:
      LaunchPackedShared(run.radial_draw.get(), run.radial_uniform.get(),
                         run.n_radial, run.theta_draw.get(),
                         run.theta_uniform.get(), run.n_theta, run.out,
                         run.params, run.cfg);
      return;
    case KernelKind::PackedIlp:
      LaunchPackedIlp(run.radial_draw.get(), run.radial_uniform.get(),
                      run.n_radial, run.theta_draw.get(),
                      run.theta_uniform.get(), run.n_theta, run.out, run.params,
                      run.cfg);
      return;
    case KernelKind::Packed:
      LaunchPacked(run.radial_draw.get(), run.radial_uniform.get(),
                   run.n_radial, run.theta_draw.get(), run.theta_uniform.get(),
                   run.n_theta, run.out, run.params, run.cfg);
      return;
  }
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
  timing.bytes_per_sample = kBytesPerSample;
  if (kind == KernelKind::PackedShared) {
    timing.shared_bytes = SharedBytesFor(run.n_radial, run.n_theta);
    timing.occupancy_blocks =
        SharedOccupancyBlocks(cfg.threads, timing.shared_bytes);
    timing.grid_blocks = GridBlocksFor(kind, cfg, timing.shared_bytes);
  } else if (kind == KernelKind::PackedIlp) {
    timing.occupancy_blocks =
        IlpOccupancyBlocks(cfg.samples_per_thread, cfg.threads);
    timing.grid_blocks = GridBlocksFor(kind, cfg, 0);
  } else {
    timing.grid_blocks = (n + cfg.threads - 1) / cfg.threads;
  }
  return timing;
}

int RunPackedBench(const BenchOptions& opt) {
  const qmc::cpu::QuantumNumbers qn{opt.principal, opt.l, opt.m};
  const KernelTiming timing = TimeGpu(opt.kernel, qn, kSeed, opt.n, opt.launch);
  qmc::harness::BenchmarkRecord rec;
  rec.part = "packed-records";
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
      "{} n={} threads={} S={} median_ms={:.3f} Gsamples/s={:.3f} GB/s={:.3f} "
      "smem={} occ_blocks={} grid={} locked={}\n",
      rec.kernel, opt.n, opt.launch.threads, opt.launch.samples_per_thread,
      rec.median_ms, rec.units_per_s / 1e9, rec.achieved_gbps,
      timing.shared_bytes, timing.occupancy_blocks, timing.grid_blocks,
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

}  // namespace qmc::packed
