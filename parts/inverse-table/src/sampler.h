#ifndef PARTS_INVERSE_TABLE_SRC_SAMPLER_H_
#define PARTS_INVERSE_TABLE_SRC_SAMPLER_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "cpu-reference/hydrogen.h"

namespace qmc::inverse {

enum class KernelKind { SharedCdf, InverseTable, InverseNodes };

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
};

struct LaunchConfig {
  int threads = 256;
  bool cache_nodes = false;
  int table_k_radial = 4096;
  int table_k_theta = 2048;
};

struct BenchOptions {
  int n = 8000000;
  int principal = 1;
  int l = 0;
  int m = 0;
  KernelKind kernel = KernelKind::SharedCdf;
  LaunchConfig launch;
  std::string out_path;
  bool scratch = true;
};

[[nodiscard]] const char* KernelName(KernelKind kind);

[[nodiscard]] std::vector<float> BuildRadialQuantile(
    const qmc::cpu::HydrogenSampler& sampler, int k);
[[nodiscard]] std::vector<float> BuildThetaQuantile(
    const qmc::cpu::HydrogenSampler& sampler, int k);
[[nodiscard]] std::vector<std::uint8_t> BuildJumpFlags(
    std::span<const double> nodes, std::span<const double> cdf,
    std::span<const float> table);

void LaunchSharedCdf(const float* r_nodes, const float* r_cdf, int n_radial,
                     const float* theta_nodes, const float* theta_cdf,
                     int n_theta, void* states, float* x, float* y, float* z,
                     float* density, float* r, float* theta, float* phi, int n,
                     int principal, int l, int m_abs, float radial_norm,
                     float y_norm, int threads, bool cache_nodes);

void LaunchInverseLookup(const float* r_table, int k_radial,
                         const float* theta_table, int k_theta,
                         const std::uint8_t* r_flags, const std::uint8_t* theta_flags,
                         void* states, float* x, float* y, float* z,
                         float* density, float* r, float* theta, float* phi,
                         int n, int principal, int l, int m_abs,
                         float radial_norm, float y_norm, bool snap_jumps,
                         int threads);

[[nodiscard]] int SharedCdfOccupancyBlocks(int threads, bool cache_nodes,
                                           int n_radial, int n_theta);

void DrawGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn, std::uint64_t seed,
             int n, GpuDraw* out, const LaunchConfig& cfg = {});
[[nodiscard]] KernelTiming TimeGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn,
                                   std::uint64_t seed, int n,
                                   const LaunchConfig& cfg = {});

int RunInverseBench(const BenchOptions& opt);
int RunNodeTrap(const std::string& out_path, int n, bool scratch);
int RunOccupancySweep(const std::string& out_path, int n, bool scratch);

}  // namespace qmc::inverse

#endif
