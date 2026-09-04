#include "parts/endgame/src/sampler.h"

#include <array>
#include <fstream>
#include <string>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "harness/env.h"

namespace qmc::endgame {

int RunLaunchSweep(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  const qmc::cpu::QuantumNumbers qn{1, 0, 0};
  const std::array<int, 3> threads = {128, 256, 512};
  const std::array<int, 4> samples = {1, 2, 4, 8};
  nlohmann::json rows = nlohmann::json::array();
  for (int t : threads) {
    for (int s : samples) {
      for (int persist = 0; persist < 2; ++persist) {
        if (persist != 0 && !(t == 256 && s == 4)) {
          continue;
        }
        LaunchConfig cfg;
        cfg.threads = t;
        cfg.samples_per_thread = s;
        cfg.persist_tables = persist != 0;
        const KernelTiming timing =
            TimeGpu(KernelKind::PhiloxIlp, qn, 0xC0FFEEULL, n, cfg);
        const double gsamples =
            static_cast<double>(n) / (timing.median_ms * 1e-3) / 1e9;
        const nlohmann::json row = {
            {"threads", t},
            {"samples_per_thread", s},
            {"persist_tables", cfg.persist_tables},
            {"occupancy_blocks", timing.occupancy_blocks},
            {"grid_blocks", timing.grid_blocks},
            {"median_ms", timing.median_ms},
            {"gsamples_per_s", gsamples},
        };
        rows.push_back(row);
        fmt::print(
            "sweep threads={} S={} persist={} occ={} {:.3f} Gsamples/s\n", t, s,
            persist, timing.occupancy_blocks, gsamples);
      }
    }
  }
  {
    LaunchConfig cfg;
    cfg.threads = 256;
    const KernelTiming timing =
        TimeGpu(KernelKind::Philox, qn, 0xC0FFEEULL, n, cfg);
    const double gsamples =
        static_cast<double>(n) / (timing.median_ms * 1e-3) / 1e9;
    rows.push_back({
        {"kernel", "philox"},
        {"threads", 256},
        {"samples_per_thread", 1},
        {"persist_tables", false},
        {"occupancy_blocks", timing.occupancy_blocks},
        {"grid_blocks", timing.grid_blocks},
        {"median_ms", timing.median_ms},
        {"gsamples_per_s", gsamples},
    });
    fmt::print("sweep philox K7 {:.3f} Gsamples/s\n", gsamples);
  }
  if (!out_path.empty()) {
    nlohmann::json body = {
        {"part", "endgame"},
        {"experiment", "launch_sweep"},
        {"n_samples", n},
        {"env",
         {{"gpu", qmc::harness::CaptureEnv().gpu},
          {"clocks_sm_mhz", qmc::harness::CaptureEnv().clocks_sm_mhz},
          {"clocks_locked", qmc::harness::CaptureEnv().clocks_locked}}},
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

}  // namespace qmc::endgame
