#include "parts/endgame/src/sampler.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <curand.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "harness/device_memory.h"
#include "harness/rng/philox.h"
#include "harness/timing.h"

namespace qmc::endgame {
namespace {

void CheckCurand(curandStatus_t status, const char* what) {
  if (status != CURAND_STATUS_SUCCESS) {
    fmt::print(stderr, "cuRAND error at {}: {}\n", what,
               static_cast<int>(status));
    std::abort();
  }
}

}  // namespace

int RunBitCheck(const std::string& out_path, int n, bool scratch) {
  (void)scratch;
  constexpr std::uint64_t kSeed = 0xC0FFEEULL;
  auto device = qmc::harness::DeviceAlloc<std::uint32_t>(static_cast<size_t>(n) * 4);
  LaunchDumpPhilox(device.get(), n, kSeed, 0, 256);
  qmc::harness::CheckCudaCall(cudaDeviceSynchronize(), "dump sync");
  const std::vector<std::uint32_t> got =
      qmc::harness::HostFromDevice(device, static_cast<size_t>(n) * 4);

  int host_mismatch = 0;
  for (int i = 0; i < n; ++i) {
    const qmc::rng::Philox4x32Ctr ctr = qmc::rng::Philox4x32TenRounds(
        qmc::rng::MakeCounter(static_cast<std::uint32_t>(i), 0),
        qmc::rng::SeedToKey(kSeed));
    const std::uint32_t* row = got.data() + static_cast<size_t>(i) * 4;
    if (row[0] != ctr.v[0] || row[1] != ctr.v[1] || row[2] != ctr.v[2] ||
        row[3] != ctr.v[3]) {
      ++host_mismatch;
    }
  }

  auto curand_out = qmc::harness::DeviceAlloc<std::uint32_t>(static_cast<size_t>(n) * 4);
  curandGenerator_t gen;
  CheckCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10),
              "create");
  CheckCurand(curandSetPseudoRandomGeneratorSeed(gen, kSeed), "seed");
  CheckCurand(
      curandGenerate(gen, curand_out.get(), static_cast<size_t>(n) * 4),
      "generate");
  CheckCurand(curandDestroyGenerator(gen), "destroy");
  const std::vector<std::uint32_t> vendor =
      qmc::harness::HostFromDevice(curand_out, static_cast<size_t>(n) * 4);
  int vendor_mismatch = 0;
  for (size_t i = 0; i < vendor.size(); ++i) {
    if (vendor[i] != got[i]) {
      ++vendor_mismatch;
    }
  }

  GpuDraw a;
  GpuDraw b;
  GpuDraw c;
  LaunchConfig cfg_a;
  cfg_a.threads = 128;
  LaunchConfig cfg_b;
  cfg_b.threads = 256;
  cfg_b.samples_per_thread = 4;
  DrawGpu(KernelKind::Philox, {1, 0, 0}, kSeed, 4096, &a, cfg_a);
  DrawGpu(KernelKind::PhiloxIlp, {1, 0, 0}, kSeed, 4096, &b, cfg_b);
  cfg_b.threads = 512;
  cfg_b.samples_per_thread = 2;
  DrawGpu(KernelKind::PhiloxIlp, {1, 0, 0}, kSeed, 4096, &c, cfg_b);
  const bool launch_ok = a.r == b.r && a.r == c.r && a.phi == b.phi && a.phi == c.phi;

  fmt::print(
      "bit-check n={} host_mismatch={} vendor_mismatch={} launch_repro={}\n", n,
      host_mismatch, vendor_mismatch, launch_ok ? 1 : 0);

  if (!out_path.empty()) {
    nlohmann::json body = {
        {"part", "endgame"},
        {"experiment", "philox_bit_check"},
        {"n", n},
        {"host_device_mismatch", host_mismatch},
        {"curand_host_api_mismatch", vendor_mismatch},
        {"launch_config_reproducible", launch_ok},
        {"note",
         "Host vs device uses the same Random123 rounds in harness/rng/"
         "philox.h. cuRAND host API packing is a different counter layout; "
         "mismatch is expected and recorded, not a rewrite trigger."},
    };
    std::ofstream out(out_path);
    if (!out) {
      fmt::print(stderr, "could not write {}\n", out_path);
      return 1;
    }
    out << body.dump(2) << '\n';
  }
  return host_mismatch == 0 && launch_ok ? 0 : 1;
}

}  // namespace qmc::endgame
