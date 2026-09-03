#ifndef HARNESS_RESULTS_H_
#define HARNESS_RESULTS_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "harness/env.h"

namespace qmc::harness {

struct BenchmarkRecord {
  std::string part;
  std::string kernel;
  std::string workload;
  std::int64_t count = 0;
  std::vector<int> quantum_numbers;
  std::vector<double> reps_ms;
  double median_ms = 0.0;
  double units_per_s = 0.0;
  double bytes_per_unit = 0.0;
  double achieved_gbps = 0.0;
  Env env;
};

[[nodiscard]] std::expected<void, std::string> WriteJson(
    const std::filesystem::path& path, const BenchmarkRecord& record,
    bool require_verify_full);

}  // namespace qmc::harness

#endif
