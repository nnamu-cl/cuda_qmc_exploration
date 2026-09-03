#ifndef HARNESS_DUMP_H_
#define HARNESS_DUMP_H_

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>

namespace qmc::harness {

struct SampleDumpMeta {
  std::uint64_t n = 0;
  std::string dtype = "float32";
  std::string layout = "float4_xyzw";
  std::array<int, 3> nlm = {0, 0, 0};
  std::uint64_t seed = 0;
};

[[nodiscard]] std::expected<void, std::string> WriteSampleDump(
    const std::filesystem::path& prefix, std::span<const float> xyzw,
    const SampleDumpMeta& meta);

}  // namespace qmc::harness

#endif
