#include "parts/inverse-table/src/sampler.h"

#include <array>
#include <fstream>
#include <string>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "harness/env.h"

namespace qmc::inverse {

int RunOccupancySweep(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  const qmc::cpu::QuantumNumbers qn{1, 0, 0};
  const std::array<int, 3> threads = {128, 256, 512};
  nlohmann::json rows = nlohmann::json::array();
  for (int cache = 0; cache < 2; ++cache) {
    for (int t : threads) {
      LaunchConfig cfg;
      cfg.threads = t;
      cfg.cache_nodes = cache != 0;
      const KernelTiming timing =
          TimeGpu(KernelKind::SharedCdf, qn, 0xC0FFEEULL, n, cfg);
      const double gsamples =
          static_cast<double>(n) / (timing.median_ms * 1e-3) / 1e9;
      const nlohmann::json row = {
          {"threads", t},
          {"cache_nodes", cfg.cache_nodes},
          {"occupancy_blocks", timing.occupancy_blocks},
          {"median_ms", timing.median_ms},
          {"gsamples_per_s", gsamples},
          {"shared_bytes",
           (cache ? 2 : 1) * (4096 + 2048) * static_cast<int>(sizeof(float))},
      };
      rows.push_back(row);
      fmt::print(
          "occupancy threads={} cache_nodes={} blocks={} {:.3f} Gsamples/s\n",
          t, cache, timing.occupancy_blocks, gsamples);
    }
  }
  if (!out_path.empty()) {
    nlohmann::json body = {
        {"part", "inverse-table"},
        {"experiment", "shared_cdf_occupancy"},
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

}  // namespace qmc::inverse
