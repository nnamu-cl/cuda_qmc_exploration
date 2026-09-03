#include "harness/stats/ks.h"

#include <algorithm>
#include <cmath>

namespace qmc::stats {

KsResult KolmogorovSmirnovD(std::span<const double> samples_sorted,
                            const CdfFn& cdf, double max_d) {
  KsResult result;
  const int n = static_cast<int>(samples_sorted.size());
  if (n == 0) {
    result.D = 1.0;
    result.passed = false;
    return result;
  }
  double d = 0.0;
  for (int i = 0; i < n; ++i) {
    const double f = cdf(samples_sorted[static_cast<size_t>(i)]);
    const double lo = static_cast<double>(i) / n;
    const double hi = static_cast<double>(i + 1) / n;
    d = std::max(d, std::max(std::abs(hi - f), std::abs(f - lo)));
  }
  result.D = d;
  result.passed = d < max_d;
  return result;
}

}  // namespace qmc::stats
