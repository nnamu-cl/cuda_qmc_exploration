#ifndef PARTS_ENDGAME_SRC_SAMPLER_H_
#define PARTS_ENDGAME_SRC_SAMPLER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cpu-reference/hydrogen.h"
#include "parts/alias-method/src/sampler.h"

namespace qmc::endgame {

enum class KernelKind { Philox, PhiloxIlp, ThrustAlias };

struct GpuDraw {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> density;
  std::vector<double> r;
  std::vector<double> theta;
  std::vector<double> phi;
};

struct KernelTiming {
  double setup_ms = 0.0;
  std::vector<double> reps_ms;
  double median_ms = 0.0;
  double bytes_per_sample = 0.0;
  int occupancy_blocks = 0;
  int grid_blocks = 0;
};

struct LaunchConfig {
  int threads = 256;
  int samples_per_thread = 1;
  int grid_blocks = 0;
  bool persist_tables = false;
};

struct BenchOptions {
  int n = 8000000;
  int principal = 1;
  int l = 0;
  int m = 0;
  KernelKind kernel = KernelKind::Philox;
  LaunchConfig launch;
  std::string out_path;
  bool scratch = true;
};

[[nodiscard]] const char* KernelName(KernelKind kind);

void LaunchPhiloxSample(const qmc::alias::AliasBin* radial, int n_radial,
                        const qmc::alias::AliasBin* theta, int n_theta,
                        qmc::alias::PackedXyzw* packed, float* x, float* y,
                        float* z, float* density, float* r, float* theta_out,
                        float* phi, int n, int principal, int l, int m_abs,
                        float radial_norm, float y_norm, std::uint64_t seed,
                        int threads);

void LaunchPhiloxIlp(const qmc::alias::AliasBin* radial, int n_radial,
                     const qmc::alias::AliasBin* theta, int n_theta,
                     qmc::alias::PackedXyzw* packed, float* x, float* y,
                     float* z, float* density, float* r, float* theta_out,
                     float* phi, int n, int principal, int l, int m_abs,
                     float radial_norm, float y_norm, std::uint64_t seed,
                     const LaunchConfig& cfg);

void LaunchDumpPhilox(std::uint32_t* out, int n, std::uint64_t seed,
                      std::uint32_t slot, int threads);

void LaunchThrustAlias(const qmc::alias::AliasBin* radial, int n_radial,
                       const qmc::alias::AliasBin* theta, int n_theta,
                       float* uniforms, qmc::alias::PackedXyzw* packed, float* x,
                       float* y, float* z, float* density, float* r,
                       float* theta_out, float* phi, int n, int principal,
                       int l, int m_abs, float radial_norm, float y_norm,
                       std::uint64_t seed);

void DrawGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn, std::uint64_t seed,
             int n, GpuDraw* out, const LaunchConfig& cfg = {});
[[nodiscard]] KernelTiming TimeGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn,
                                   std::uint64_t seed, int n,
                                   const LaunchConfig& cfg = {});

int RunEndgameBench(const BenchOptions& opt);
int RunLaunchSweep(const std::string& out_path, int n, bool scratch);
int RunBitCheck(const std::string& out_path, int n, bool scratch);
int RunSampleDump(const std::string& prefix, int n, qmc::cpu::QuantumNumbers qn,
                  std::uint64_t seed);

[[nodiscard]] int IlpOccupancyBlocks(int samples_per_thread, int threads);
[[nodiscard]] int IlpGridBlocks(int samples_per_thread, int threads);

}  // namespace qmc::endgame

#endif
