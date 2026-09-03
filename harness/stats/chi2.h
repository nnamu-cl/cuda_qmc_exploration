#ifndef HARNESS_STATS_CHI2_H_
#define HARNESS_STATS_CHI2_H_

#include <span>

namespace qmc::stats {

struct Chi2Result {
  double chi2 = 0.0;
  int dof = 0;
  bool passed = false;
};

[[nodiscard]] Chi2Result ChiSquare(std::span<const double> observed,
                                   std::span<const double> expected,
                                   double max_chi2);

}  // namespace qmc::stats

#endif
