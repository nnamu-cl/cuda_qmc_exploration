#include "parts/inverse-table/src/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace qmc::inverse {

std::vector<float> BuildRadialQuantile(const qmc::cpu::HydrogenSampler& sampler,
                                       int k) {
  std::vector<float> table(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    const double u =
        (k <= 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(k - 1);
    table[static_cast<size_t>(i)] = static_cast<float>(sampler.InvertRadial(u));
  }
  return table;
}

std::vector<float> BuildThetaQuantile(const qmc::cpu::HydrogenSampler& sampler,
                                      int k) {
  std::vector<float> table(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    const double u =
        (k <= 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(k - 1);
    table[static_cast<size_t>(i)] = static_cast<float>(sampler.InvertTheta(u));
  }
  return table;
}

std::vector<std::uint8_t> BuildJumpFlags(std::span<const double> nodes,
                                         std::span<const double> cdf,
                                         std::span<const float> table) {
  std::vector<std::uint8_t> flags(
      table.size() > 1 ? table.size() - 1 : 0, 0);
  if (nodes.size() < 3 || table.size() < 2) {
    return flags;
  }
  const int n = static_cast<int>(nodes.size());
  std::vector<double> pdf(static_cast<size_t>(n), 0.0);
  double max_pdf = 0.0;
  for (int i = 1; i < n; ++i) {
    const double dr =
        nodes[static_cast<size_t>(i)] - nodes[static_cast<size_t>(i - 1)];
    pdf[static_cast<size_t>(i)] =
        (dr > 0.0)
            ? (cdf[static_cast<size_t>(i)] - cdf[static_cast<size_t>(i - 1)]) / dr
            : 0.0;
    max_pdf = std::max(max_pdf, pdf[static_cast<size_t>(i)]);
  }
  const double cut = 1.0e-4 * std::max(max_pdf, 1.0e-30);
  std::vector<char> low(static_cast<size_t>(n), 0);
  for (int i = 1; i < n; ++i) {
    low[static_cast<size_t>(i)] = pdf[static_cast<size_t>(i)] < cut;
  }
  std::vector<std::pair<double, double>> holes;
  int i = 1;
  while (i < n) {
    if (!low[static_cast<size_t>(i)]) {
      ++i;
      continue;
    }
    int j = i;
    while (j < n && low[static_cast<size_t>(j)]) {
      ++j;
    }
    if (i > 1 && j < n) {
      holes.emplace_back(nodes[static_cast<size_t>(i - 1)],
                         nodes[static_cast<size_t>(j - 1)]);
    }
    i = j;
  }
  for (size_t t = 0; t + 1 < table.size(); ++t) {
    const double a = table[t];
    const double b = table[t + 1];
    for (const auto& hole : holes) {
      if (a < hole.second && b > hole.first) {
        flags[t] = 1;
      }
    }
  }
  return flags;
}

}  // namespace qmc::inverse
