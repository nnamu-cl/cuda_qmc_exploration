#include "parts/packed-records/src/sampler.h"

#include <array>
#include <fstream>
#include <string>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "harness/env.h"

namespace qmc::packed {
namespace {

void PushRow(nlohmann::json* rows, KernelKind kind, const LaunchConfig& cfg,
             int n) {
  const KernelTiming timing =
      TimeGpu(kind, {1, 0, 0}, 0xC0FFEEULL, n, cfg);
  const double gsamples =
      static_cast<double>(n) / (timing.median_ms * 1e-3) / 1e9;
  rows->push_back({
      {"kernel", KernelName(kind)},
      {"threads", cfg.threads},
      {"samples_per_thread", cfg.samples_per_thread},
      {"shared_bytes", timing.shared_bytes},
      {"occupancy_blocks", timing.occupancy_blocks},
      {"grid_blocks", timing.grid_blocks},
      {"median_ms", timing.median_ms},
      {"gsamples_per_s", gsamples},
      {"achieved_gbps", gsamples * timing.bytes_per_sample},
  });
  fmt::print("sweep {} threads={} S={} smem={} occ={} {:.3f} Gsamples/s\n",
             KernelName(kind), cfg.threads, cfg.samples_per_thread,
             timing.shared_bytes, timing.occupancy_blocks, gsamples);
}

}  // namespace

// Block size for the flat kernels, then the shared-vs-global question with a
// matched grid-stride control, then the ILP retest K8 said was settled while
// the LG queue was full.
int RunLayoutSweep(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  const qmc::harness::Env env = qmc::harness::CaptureEnv();
  nlohmann::json rows = nlohmann::json::array();
  constexpr std::array<int, 4> kThreads = {128, 256, 512, 1024};

  for (int t : kThreads) {
    LaunchConfig cfg;
    cfg.threads = t;
    PushRow(&rows, KernelKind::Packed, cfg, n);
  }
  for (int t : kThreads) {
    LaunchConfig cfg;
    cfg.threads = t;
    PushRow(&rows, KernelKind::PackedLinear, cfg, n);
  }
  // Shared needs 48 KB/block, so only large blocks recover the occupancy.
  for (int t : {256, 512, 1024}) {
    LaunchConfig cfg;
    cfg.threads = t;
    PushRow(&rows, KernelKind::PackedShared, cfg, n);
  }
  // Matched grid-stride global control at the same block shapes.
  for (int t : {256, 512, 1024}) {
    LaunchConfig cfg;
    cfg.threads = t;
    cfg.samples_per_thread = 1;
    PushRow(&rows, KernelKind::PackedIlp, cfg, n);
  }
  // ILP retest.
  for (int t : {128, 256}) {
    for (int s : {2, 4}) {
      LaunchConfig cfg;
      cfg.threads = t;
      cfg.samples_per_thread = s;
      PushRow(&rows, KernelKind::PackedIlp, cfg, n);
    }
  }

  if (!out_path.empty()) {
    const nlohmann::json body = {
        {"part", "packed-records"},
        {"experiment", "layout_sweep"},
        {"n_samples", n},
        {"bytes_per_sample", 16.0},
        {"env",
         {{"gpu", env.gpu},
          {"clocks_sm_mhz", env.clocks_sm_mhz},
          {"clocks_mem_mhz", env.clocks_mem_mhz},
          {"clocks_locked", env.clocks_locked},
          {"git", env.git}}},
        {"rows", rows},
    };
    std::ofstream out(out_path);
    if (!out) {
      fmt::print(stderr, "could not write {}\n", out_path);
      return 1;
    }
    out << body.dump(2) << '\n';
  }
  return 0;
}

}  // namespace qmc::packed
