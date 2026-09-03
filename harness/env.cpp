#include "harness/env.h"

#include <cuda_runtime.h>

#include <array>
#include <charconv>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>

namespace qmc::harness {
namespace {

std::string FormatCudaVersion(int version) {
  const int major = version / 1000;
  const int minor = (version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

std::string ReadPipe(const char* cmd) {
  FILE* pipe = popen(cmd, "r");
  if (pipe == nullptr) {
    return {};
  }
  std::string out;
  std::array<char, 512> buf{};
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    out += buf.data();
  }
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out;
}

std::string_view TrimView(std::string_view s) {
  const auto start = s.find_first_not_of(" \t");
  if (start == std::string_view::npos) {
    return {};
  }
  const auto end = s.find_last_not_of(" \t");
  return s.substr(start, end - start + 1);
}

int ParseInt(std::string_view s, int fallback = 0) {
  s = TrimView(s);
  int value = 0;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  return ec == std::errc{} ? value : fallback;
}

double ParseDouble(std::string_view s, double fallback = 0.0) {
  s = TrimView(s);
  double value = 0.0;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  return ec == std::errc{} ? value : fallback;
}

}  // namespace

Env CaptureEnv() {
  Env env;
  int driver = 0;
  int runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  env.cuda = FormatCudaVersion(runtime);

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    env.gpu = prop.name;
  }

  env.git = ReadPipe("git rev-parse --short HEAD 2>/dev/null");
  if (env.git.empty()) {
    env.git = "uncommitted";
  }

  const std::string smi = ReadPipe(
      "nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,"
      "temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null");
  std::stringstream ss(smi);
  std::string gpu;
  std::string driver_smi;
  std::string sm;
  std::string mem;
  std::string temp;
  std::string power;
  if (std::getline(ss, gpu, ',') && std::getline(ss, driver_smi, ',') &&
      std::getline(ss, sm, ',') && std::getline(ss, mem, ',') &&
      std::getline(ss, temp, ',') && std::getline(ss, power, ',')) {
    if (env.gpu.empty()) {
      env.gpu = std::string(TrimView(gpu));
    }
    env.driver = std::string(TrimView(driver_smi));
    env.clocks_sm_mhz = ParseInt(sm);
    env.clocks_mem_mhz = ParseInt(mem);
    env.temperature_c = ParseInt(temp);
    env.power_w = ParseDouble(power);
  }

  if (const char* target = std::getenv("QMC_LOCK_SM_MHZ");
      target != nullptr && target[0] != '\0') {
    env.clocks_locked = env.clocks_sm_mhz == ParseInt(target);
  }
  return env;
}

}  // namespace qmc::harness
