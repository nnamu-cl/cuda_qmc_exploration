#include "parts/packed-records/src/sampler.h"

#include <array>
#include <fstream>
#include <string>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "harness/env.h"

namespace qmc::packed {

// The same six orbitals the verify matrix uses. Feeds the ridge-crossover
// figure: how flat is K9 across n and l once the tables stop scattering?
int RunOrbitalSweep(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  constexpr std::array<qmc::cpu::QuantumNumbers, 6> kOrbitals = {{
      {1, 0, 0}, {2, 0, 0}, {2, 1, 1}, {3, 1, -1}, {4, 2, 0}, {5, 0, 0},
  }};
  const qmc::harness::Env env = qmc::harness::CaptureEnv();
  nlohmann::json rows = nlohmann::json::array();
  for (const qmc::cpu::QuantumNumbers qn : kOrbitals) {
    LaunchConfig cfg;
    const KernelTiming timing =
        TimeGpu(KernelKind::Packed, qn, 0xC0FFEEULL, n, cfg);
    const double gsamples =
        static_cast<double>(n) / (timing.median_ms * 1e-3) / 1e9;
    rows.push_back({
        {"kernel", KernelName(KernelKind::Packed)},
        {"quantum_numbers", {qn.n, qn.l, qn.m}},
        {"median_ms", timing.median_ms},
        {"gsamples_per_s", gsamples},
        {"bytes_per_sample", timing.bytes_per_sample},
        {"achieved_gbps", gsamples * timing.bytes_per_sample},
    });
    fmt::print("orbital ({},{},{}) {:.3f} Gsamples/s {:.1f} GB/s\n", qn.n, qn.l,
               qn.m, gsamples, gsamples * timing.bytes_per_sample);
  }
  if (!out_path.empty()) {
    const nlohmann::json body = {
        {"part", "packed-records"},
        {"experiment", "orbital_sweep"},
        {"n_samples", n},
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
