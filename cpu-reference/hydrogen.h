#ifndef CPU_REFERENCE_HYDROGEN_H_
#define CPU_REFERENCE_HYDROGEN_H_

#include <cstdint>
#include <span>
#include <vector>

#include "cpu-reference/special.h"

namespace qmc::cpu {

inline constexpr int kRadialBins = 4096;
inline constexpr int kThetaBins = 2048;
inline constexpr int kFineCdfScale = 10;

struct QuantumNumbers {
  int n = 1;
  int l = 0;
  int m = 0;
};

struct HydrogenSample {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double density = 0.0;
  double r = 0.0;
  double theta = 0.0;
  double phi = 0.0;
};

class HydrogenSampler {
 public:
  explicit HydrogenSampler(QuantumNumbers qn);

  [[nodiscard]] QuantumNumbers Numbers() const { return qn_; }
  [[nodiscard]] double RadiusMax() const { return r_nodes_.back(); }
  [[nodiscard]] HydrogenSample Sample(std::uint64_t seed,
                                      std::uint32_t index) const;
  void SampleMany(std::uint64_t seed, std::span<HydrogenSample> out,
                  bool parallel) const;

  [[nodiscard]] double RadialCdf(double r) const;
  [[nodiscard]] double ThetaCdf(double theta) const;

 private:
  QuantumNumbers qn_;
  std::vector<double> r_nodes_;
  std::vector<double> r_cdf_;
  std::vector<double> theta_nodes_;
  std::vector<double> theta_cdf_;
  std::vector<double> r_nodes_fine_;
  std::vector<double> r_cdf_fine_;
  std::vector<double> theta_nodes_fine_;
  std::vector<double> theta_cdf_fine_;
};

}  // namespace qmc::cpu

#endif
