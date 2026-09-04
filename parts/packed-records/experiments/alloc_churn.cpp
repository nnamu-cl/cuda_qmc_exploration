#include "parts/packed-records/src/sampler.h"

#include <fstream>
#include <string>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "harness/env.h"

namespace qmc::packed {

// Not a kernel question — an instrument question. The same cell, timed six
// times in one process, is not the same number if the ~128 MB output buffer is
// freed and re-allocated between cells. Every multi-cell sweep in this repo
// (including endgame's launch sweep) measures its first cell under
// fresh-allocation conditions and every later cell under churned ones.
int RunAllocChurn(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  const qmc::harness::Env env = qmc::harness::CaptureEnv();
  constexpr int kReps = 6;
  nlohmann::json arms = nlohmann::json::array();
  for (int arm = 0; arm < 2; ++arm) {
    const bool reuse = arm == 1;
    SetScratchReuse(reuse);
    nlohmann::json cells = nlohmann::json::array();
    for (int rep = 0; rep < kReps; ++rep) {
      LaunchConfig cfg;
      cfg.threads = 128;
      const KernelTiming timing =
          TimeGpu(KernelKind::Packed, {1, 0, 0}, 0xC0FFEEULL, n, cfg);
      const double gsamples =
          static_cast<double>(n) / (timing.median_ms * 1e-3) / 1e9;
      cells.push_back({{"rep", rep},
                       {"median_ms", timing.median_ms},
                       {"gsamples_per_s", gsamples}});
      fmt::print("churn reuse_scratch={} rep={} {:.3f} Gsamples/s\n",
                 reuse ? 1 : 0, rep, gsamples);
    }
    arms.push_back({{"reuse_scratch", reuse}, {"cells", cells}});
  }
  SetScratchReuse(true);
  if (!out_path.empty()) {
    const nlohmann::json body = {
        {"part", "packed-records"},
        {"experiment", "alloc_churn"},
        {"kernel", KernelName(KernelKind::Packed)},
        {"threads", 128},
        {"n_samples", n},
        {"env",
         {{"gpu", env.gpu},
          {"clocks_sm_mhz", env.clocks_sm_mhz},
          {"clocks_mem_mhz", env.clocks_mem_mhz},
          {"clocks_locked", env.clocks_locked},
          {"git", env.git}}},
        {"arms", arms},
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
