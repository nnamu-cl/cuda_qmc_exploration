#include "harness/microbench.h"

#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "harness/device_memory.h"
#include "harness/env.h"
#include "harness/results.h"
#include "harness/timing.h"

namespace qmc::harness {
namespace {

void CheckCurand(curandStatus_t status, const char* what) {
  if (status != CURAND_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuRAND error at %s: %d\n", what,
                 static_cast<int>(status));
    std::abort();
  }
}

__global__ void CopyFloat4(const float4* in, float4* out, int n) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
       i += stride) {
    out[i] = in[i];
  }
}

__global__ void Fma32Loop(float* out, int n, int inner) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
       i += stride) {
    float a = 1.0000001f;
    float b = 1.0000001f;
    for (int k = 0; k < inner; ++k) {
      a = a * b + a;
    }
    out[i] = a;
  }
}

__global__ void Fma64Loop(double* out, int n, int inner) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
       i += stride) {
    double a = 1.0000001;
    double b = 1.0000001;
    for (int k = 0; k < inner; ++k) {
      a = a * b + a;
    }
    out[i] = a;
  }
}

int Grid(int n) {
  return std::min(2048, (n + 255) / 256);
}

void FillRecord(BenchmarkRecord* rec, const MicrobenchOptions& opt,
                const TimingResult& timing, double bytes_per_unit,
                double units) {
  rec->part = opt.part;
  rec->kernel = opt.kind;
  rec->workload = "microbench";
  rec->count = opt.n;
  rec->reps_ms = timing.reps_ms;
  rec->median_ms = timing.median_ms;
  rec->bytes_per_unit = bytes_per_unit;
  rec->units_per_s = units / (timing.median_ms * 1e-3);
  rec->achieved_gbps =
      (units * bytes_per_unit) / (timing.median_ms * 1e-3) / 1e9;
  rec->env = CaptureEnv();
}

}  // namespace

int RunMicrobench(const MicrobenchOptions& opt) {
  const int threads = 256;
  const int blocks = Grid(opt.n);
  TimingResult timing;
  double bytes = 0.0;
  double units = static_cast<double>(opt.n);

  if (opt.kind == "bw_copy") {
    auto in = DeviceAlloc<float4>(static_cast<size_t>(opt.n));
    auto out = DeviceAlloc<float4>(static_cast<size_t>(opt.n));
    DeviceFill(in, 1, static_cast<size_t>(opt.n));
    timing = TimeCudaLaunch({}, [&] {
      CopyFloat4<<<blocks, threads>>>(in.get(), out.get(), opt.n);
    });
    CheckCuda("CopyFloat4");
    bytes = 2.0 * sizeof(float4);
  } else if (opt.kind == "fma32") {
    constexpr int kInner = 1024;
    auto out = DeviceAlloc<float>(static_cast<size_t>(opt.n));
    timing = TimeCudaLaunch({}, [&] {
      Fma32Loop<<<blocks, threads>>>(out.get(), opt.n, kInner);
    });
    CheckCuda("Fma32Loop");
    bytes = sizeof(float);
    units = static_cast<double>(opt.n) * kInner;
  } else if (opt.kind == "fma64") {
    constexpr int kInner = 1024;
    auto out = DeviceAlloc<double>(static_cast<size_t>(opt.n));
    timing = TimeCudaLaunch({}, [&] {
      Fma64Loop<<<blocks, threads>>>(out.get(), opt.n, kInner);
    });
    CheckCuda("Fma64Loop");
    bytes = sizeof(double);
    units = static_cast<double>(opt.n) * kInner;
  } else if (opt.kind == "curand_philox" || opt.kind == "curand_xorwow") {
    auto out = DeviceAlloc<float>(static_cast<size_t>(opt.n));
    curandGenerator_t gen;
    const curandRngType_t rng = (opt.kind == "curand_philox")
                                    ? CURAND_RNG_PSEUDO_PHILOX4_32_10
                                    : CURAND_RNG_PSEUDO_XORWOW;
    CheckCurand(curandCreateGenerator(&gen, rng), "create");
    CheckCurand(curandSetPseudoRandomGeneratorSeed(gen, 42ULL), "seed");
    CheckCurand(curandGenerateUniform(gen, out.get(), static_cast<size_t>(opt.n)),
                "warmup generate");
    timing = TimeCudaLaunch({}, [&] {
      CheckCurand(
          curandGenerateUniform(gen, out.get(), static_cast<size_t>(opt.n)),
          "generate");
    });
    curandDestroyGenerator(gen);
    bytes = sizeof(float);
  } else {
    std::fprintf(stderr,
                 "unknown kind %s (bw_copy|fma32|fma64|curand_philox|"
                 "curand_xorwow)\n",
                 opt.kind.c_str());
    return 2;
  }

  BenchmarkRecord rec;
  FillRecord(&rec, opt, timing, bytes, units);
  std::printf("%s median_ms=%.4f units/s=%.3e GB/s=%.2f locked=%d\n",
              opt.kind.c_str(), rec.median_ms, rec.units_per_s,
              rec.achieved_gbps, rec.env.clocks_locked ? 1 : 0);
  if (!opt.out_path.empty()) {
    if (const auto written =
            WriteJson(opt.out_path, rec, /*require_verify_full=*/!opt.scratch);
        !written) {
      std::fprintf(stderr, "%s\n", written.error().c_str());
      return 1;
    }
  }
  return 0;
}

}  // namespace qmc::harness
