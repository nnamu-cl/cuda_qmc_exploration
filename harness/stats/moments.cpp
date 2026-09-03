#include "harness/stats/moments.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

namespace qmc::stats {

MomentResult MomentVsExact(std::span<const double> values, double exact,
                           double n_sigma) {
  MomentResult result{.exact = exact};
  const auto n = static_cast<int>(values.size());
  if (n < 2) {
    result.passed = false;
    return result;
  }
  const double mean = std::ranges::fold_left(values, 0.0, std::plus{}) / n;
  const double var =
      std::ranges::fold_left(values, 0.0, [mean](double acc, double v) {
        const double d = v - mean;
        return acc + d * d;
      }) / (n - 1);
  result.sample_mean = mean;
  result.standard_error = std::sqrt(var / n);
  result.passed = std::abs(mean - exact) < n_sigma * result.standard_error;
  return result;
}

}  // namespace qmc::stats
