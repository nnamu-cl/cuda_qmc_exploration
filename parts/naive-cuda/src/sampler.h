#ifndef PARTS_NAIVE_CUDA_SRC_SAMPLER_H_
#define PARTS_NAIVE_CUDA_SRC_SAMPLER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cpu-reference/hydrogen.h"

namespace qmc::naive {

enum class KernelKind { Fp64, Fp32, Fp32Fast };

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

struct NaiveBenchOptions {
  int n = 8000000;
  int principal = 1;
  int l = 0;
  int m = 0;
  KernelKind kernel = KernelKind::Fp64;
  std::string out_path;
  bool scratch = true;
};

[[nodiscard]] std::size_t XorwowStateBytes();
void SetupXorwowStates(void* states, std::uint64_t seed, int n);

void LaunchSampleFp64(const double* r_nodes, const double* r_cdf, int n_radial,
                      const double* theta_nodes, const double* theta_cdf,
                      int n_theta, void* states, double* x, double* y,
                      double* z, double* density, double* r, double* theta,
                      double* phi, int n, int principal, int l, int m_abs,
                      double radial_norm, double y_norm);

void LaunchSampleFp32(const float* r_nodes, const float* r_cdf, int n_radial,
                      const float* theta_nodes, const float* theta_cdf,
                      int n_theta, void* states, float* x, float* y, float* z,
                      float* density, float* r, float* theta, float* phi, int n,
                      int principal, int l, int m_abs, float radial_norm,
                      float y_norm);

void LaunchSampleFp32Fast(const float* r_nodes, const float* r_cdf,
                          int n_radial, const float* theta_nodes,
                          const float* theta_cdf, int n_theta, void* states,
                          float* x, float* y, float* z, float* density,
                          float* r, float* theta, float* phi, int n,
                          int principal, int l, int m_abs, float radial_norm,
                          float y_norm);

void InvertRadialGpu(const double* nodes, const double* cdf, int n_bins,
                     const double* u, double* r, int n);

void EvaluateRadialGpu(const int* n, const int* l, const double* r,
                       const double* norm, double* out, int count);
void EvaluateLegendreGpu(const int* l, const int* m, const double* x,
                         double* out, int count);
void EvaluateRadialGpuF32(const int* n, const int* l, const float* r,
                          const float* norm, float* out, int count);
void EvaluateLegendreGpuF32(const int* l, const int* m, const float* x,
                            float* out, int count);

void DrawGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn, std::uint64_t seed,
             int n, GpuDraw* out);
[[nodiscard]] KernelTiming TimeGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn,
                                   std::uint64_t seed, int n);

int RunNaiveBench(const NaiveBenchOptions& opt);
int RunFp32ErrorAnalysis(const std::string& out_path);

[[nodiscard]] const char* KernelName(KernelKind kind);

}  // namespace qmc::naive

#endif
