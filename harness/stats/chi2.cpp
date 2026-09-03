#include "harness/stats/chi2.h"

#include <algorithm>

namespace qmc::stats {

Chi2Result ChiSquare(std::span<const double> observed,
                     std::span<const double> expected, double max_chi2) {
  Chi2Result result;
  const int n = static_cast<int>(observed.size());
  result.dof = n > 0 ? n - 1 : 0;
  double sum = 0.0;
  const int count = std::min(n, static_cast<int>(expected.size()));
  for (int i = 0; i < count; ++i) {
    const double e = expected[static_cast<size_t>(i)];
    if (e > 0.0) {
      const double delta = observed[static_cast<size_t>(i)] - e;
      sum += delta * delta / e;
    }
  }
  result.chi2 = sum;
  result.passed = sum < max_chi2;
  return result;
}

}  // namespace qmc::stats
