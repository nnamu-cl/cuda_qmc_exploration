#include "harness/results.h"

#include <fstream>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "harness/verify_gate.h"

namespace qmc::harness {

std::expected<void, std::string> WriteJson(const std::filesystem::path& path,
                                           const BenchmarkRecord& record,
                                           bool require_verify_full) {
  if (require_verify_full && !VerifyFullPassed()) {
    return std::unexpected(
        fmt::format("refusing to write {}: verify-full has not passed",
                    path.string()));
  }
  const nlohmann::json body = {
      {"part", record.part},
      {"kernel", record.kernel},
      {"workload", record.workload},
      {"n_samples", record.count},
      {"quantum_numbers", record.quantum_numbers},
      {"reps_ms", record.reps_ms},
      {"median_ms", record.median_ms},
      {"gsamples_per_s", record.units_per_s / 1e9},
      {"bytes_per_sample", record.bytes_per_unit},
      {"achieved_gbps", record.achieved_gbps},
      {"env",
       {{"gpu", record.env.gpu},
        {"driver", record.env.driver},
        {"cuda", record.env.cuda},
        {"clocks_sm_mhz", record.env.clocks_sm_mhz},
        {"clocks_mem_mhz", record.env.clocks_mem_mhz},
        {"temperature_c", record.env.temperature_c},
        {"power_w", record.env.power_w},
        {"clocks_locked", record.env.clocks_locked},
        {"git", record.env.git}}},
  };
  std::ofstream out(path);
  if (!out) {
    return std::unexpected(fmt::format("could not write {}", path.string()));
  }
  out << body.dump(2) << '\n';
  if (!out) {
    return std::unexpected(fmt::format("failed writing {}", path.string()));
  }
  return {};
}

}  // namespace qmc::harness
