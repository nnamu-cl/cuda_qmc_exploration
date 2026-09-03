#include "harness/dump.h"

#include <fstream>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace qmc::harness {

std::expected<void, std::string> WriteSampleDump(
    const std::filesystem::path& prefix, std::span<const float> xyzw,
    const SampleDumpMeta& meta) {
  const auto json_path = prefix.string() + ".json";
  const auto bin_path = prefix.string() + ".bin";
  const nlohmann::json body = {
      {"n", meta.n},
      {"dtype", meta.dtype},
      {"layout", meta.layout},
      {"nlm", meta.nlm},
      {"seed", meta.seed},
      {"bytes", xyzw.size_bytes()},
      {"binary", bin_path},
  };
  std::ofstream json(json_path);
  if (!json) {
    return std::unexpected(fmt::format("could not write {}", json_path));
  }
  json << body.dump(2) << '\n';
  std::ofstream bin(bin_path, std::ios::binary);
  if (!bin) {
    return std::unexpected(fmt::format("could not write {}", bin_path));
  }
  bin.write(reinterpret_cast<const char*>(xyzw.data()),
            static_cast<std::streamsize>(xyzw.size_bytes()));
  if (!json || !bin) {
    return std::unexpected("failed writing sample dump");
  }
  return {};
}

}  // namespace qmc::harness
