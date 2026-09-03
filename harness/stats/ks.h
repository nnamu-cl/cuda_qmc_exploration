#ifndef HARNESS_STATS_KS_H_
#define HARNESS_STATS_KS_H_

#include <functional>
#include <span>

namespace qmc::stats {

struct KsResult {
  double D = 0.0;
  bool passed = false;
};

using CdfFn = std::function<double(double)>;

[[nodiscard]] KsResult KolmogorovSmirnovD(
    std::span<const double> samples_sorted, const CdfFn& cdf, double max_d);

}  // namespace qmc::stats

#endif
