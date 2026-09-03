#ifndef HARNESS_STATS_MOMENTS_H_
#define HARNESS_STATS_MOMENTS_H_

#include <span>

namespace qmc::stats {

struct MomentResult {
  double sample_mean = 0.0;
  double exact = 0.0;
  double standard_error = 0.0;
  bool passed = false;
};

[[nodiscard]] MomentResult MomentVsExact(std::span<const double> values,
                                         double exact, double n_sigma = 5.0);

}  // namespace qmc::stats

#endif
