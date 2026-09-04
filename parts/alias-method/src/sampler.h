#ifndef PARTS_ALIAS_METHOD_SRC_SAMPLER_H_
#define PARTS_ALIAS_METHOD_SRC_SAMPLER_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "cpu-reference/hydrogen.h"

namespace qmc::alias {

enum class KernelKind { Uniform, Linear, Float4, Split };

struct AliasBin {
  float prob = 0.0f;
  int alias = 0;
  float lo = 0.0f;
  float width = 0.0f;
  float y0 = 0.0f;
  float y1 = 0.0f;
};

struct alignas(16) PackedXyzw {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
};

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
};

struct LaunchConfig {
  int threads = 256;
};

struct BenchOptions {
  int n = 8000000;
  int principal = 1;
  int l = 0;
  int m = 0;
  KernelKind kernel = KernelKind::Linear;
  LaunchConfig launch;
  std::string out_path;
  bool scratch = true;
};

[[nodiscard]] const char* KernelName(KernelKind kind);

[[nodiscard]] std::vector<AliasBin> BuildVoseBins(
    std::span<const double> masses, std::span<const double> lo,
    std::span<const double> width, std::span<const double> y0,
    std::span<const double> y1);

[[nodiscard]] std::vector<double> ReconstructMasses(
    std::span<const AliasBin> bins);

[[nodiscard]] std::vector<AliasBin> BuildRadialAlias(
    const qmc::cpu::HydrogenSampler& sampler);
[[nodiscard]] std::vector<AliasBin> BuildThetaAlias(
    const qmc::cpu::HydrogenSampler& sampler);

void LaunchAliasSample(const AliasBin* radial, int n_radial,
                       const AliasBin* theta, int n_theta, void* states,
                       float* x, float* y, float* z, float* density, float* r,
                       float* theta_out, float* phi, void* packed, int n,
                       int principal, int l, int m_abs, float radial_norm,
                       float y_norm, bool linear, int threads);

void LaunchAliasSplit(const AliasBin* radial, int n_radial,
                      const AliasBin* theta, int n_theta, void* states,
                      float* x, float* y, float* z, float* density, float* r,
                      float* theta_out, float* phi, int n, int principal,
                      int l, int m_abs, float radial_norm, float y_norm,
                      int threads);

void DrawGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn, std::uint64_t seed,
             int n, GpuDraw* out, const LaunchConfig& cfg = {});
[[nodiscard]] KernelTiming TimeGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn,
                                   std::uint64_t seed, int n,
                                   const LaunchConfig& cfg = {});

int RunAliasBench(const BenchOptions& opt);
int RunInteriorCompare(const std::string& out_path, int n, bool scratch);
int RunNodeHole(const std::string& out_path, int n, bool scratch);

}  // namespace qmc::alias

#endif
