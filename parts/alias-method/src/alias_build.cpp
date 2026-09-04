#include "parts/alias-method/src/sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "cpu-reference/special.h"

namespace qmc::alias {
namespace {

std::vector<AliasBin> BinsFromMasses(std::span<const double> masses,
                                     std::span<const double> nodes,
                                     std::span<const double> pdf) {
  const int n = static_cast<int>(masses.size());
  std::vector<double> lo(static_cast<size_t>(n));
  std::vector<double> width(static_cast<size_t>(n));
  std::vector<double> y0(static_cast<size_t>(n));
  std::vector<double> y1(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double left = nodes[static_cast<size_t>(i)];
    const double right = nodes[static_cast<size_t>(i + 1)];
    const double w = right - left;
    const double raw0 = pdf[static_cast<size_t>(i)];
    const double raw1 = pdf[static_cast<size_t>(i + 1)];
    const double trap = 0.5 * (raw0 + raw1) * w;
    const double scale =
        (trap > 0.0) ? (masses[static_cast<size_t>(i)] / trap) : 0.0;
    lo[static_cast<size_t>(i)] = left;
    width[static_cast<size_t>(i)] = w;
    y0[static_cast<size_t>(i)] = scale * raw0;
    y1[static_cast<size_t>(i)] = scale * raw1;
  }
  return BuildVoseBins(masses, lo, width, y0, y1);
}

}  // namespace

std::vector<AliasBin> BuildVoseBins(std::span<const double> masses,
                                    std::span<const double> lo,
                                    std::span<const double> width,
                                    std::span<const double> y0,
                                    std::span<const double> y1) {
  const int n = static_cast<int>(masses.size());
  std::vector<AliasBin> bins(static_cast<size_t>(n));
  if (n <= 0) {
    return bins;
  }
  std::vector<double> scaled(static_cast<size_t>(n));
  std::vector<int> small;
  std::vector<int> large;
  small.reserve(static_cast<size_t>(n));
  large.reserve(static_cast<size_t>(n));
  const double n_real = static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    scaled[static_cast<size_t>(i)] = masses[static_cast<size_t>(i)] * n_real;
    if (scaled[static_cast<size_t>(i)] < 1.0) {
      small.push_back(i);
    } else {
      large.push_back(i);
    }
    bins[static_cast<size_t>(i)].lo = static_cast<float>(lo[static_cast<size_t>(i)]);
    bins[static_cast<size_t>(i)].width =
        static_cast<float>(width[static_cast<size_t>(i)]);
    bins[static_cast<size_t>(i)].y0 = static_cast<float>(y0[static_cast<size_t>(i)]);
    bins[static_cast<size_t>(i)].y1 = static_cast<float>(y1[static_cast<size_t>(i)]);
    bins[static_cast<size_t>(i)].alias = i;
    bins[static_cast<size_t>(i)].prob = 1.0f;
  }
  while (!small.empty() && !large.empty()) {
    const int l = small.back();
    small.pop_back();
    const int g = large.back();
    large.pop_back();
    bins[static_cast<size_t>(l)].prob =
        static_cast<float>(scaled[static_cast<size_t>(l)]);
    bins[static_cast<size_t>(l)].alias = g;
    scaled[static_cast<size_t>(g)] =
        (scaled[static_cast<size_t>(g)] + scaled[static_cast<size_t>(l)]) - 1.0;
    if (scaled[static_cast<size_t>(g)] < 1.0) {
      small.push_back(g);
    } else {
      large.push_back(g);
    }
  }
  while (!large.empty()) {
    const int g = large.back();
    large.pop_back();
    bins[static_cast<size_t>(g)].prob = 1.0f;
    bins[static_cast<size_t>(g)].alias = g;
  }
  while (!small.empty()) {
    const int l = small.back();
    small.pop_back();
    bins[static_cast<size_t>(l)].prob = 1.0f;
    bins[static_cast<size_t>(l)].alias = l;
  }
  return bins;
}

std::vector<double> ReconstructMasses(std::span<const AliasBin> bins) {
  const int n = static_cast<int>(bins.size());
  std::vector<double> masses(static_cast<size_t>(n), 0.0);
  if (n <= 0) {
    return masses;
  }
  const double inv_n = 1.0 / static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    const double keep = static_cast<double>(bins[static_cast<size_t>(i)].prob);
    masses[static_cast<size_t>(i)] += keep * inv_n;
    const int other = bins[static_cast<size_t>(i)].alias;
    if (other >= 0 && other < n) {
      masses[static_cast<size_t>(other)] += (1.0 - keep) * inv_n;
    }
  }
  return masses;
}

std::vector<AliasBin> BuildRadialAlias(const qmc::cpu::HydrogenSampler& sampler) {
  const auto nodes = sampler.RadialNodes();
  const auto cdf = sampler.RadialCdf();
  const int n_nodes = static_cast<int>(nodes.size());
  const int n_bins = std::max(n_nodes - 1, 0);
  std::vector<double> masses(static_cast<size_t>(n_bins));
  std::vector<double> pdf(static_cast<size_t>(n_nodes));
  const qmc::cpu::QuantumNumbers qn = sampler.Numbers();
  for (int i = 0; i < n_nodes; ++i) {
    const double r = nodes[static_cast<size_t>(i)];
    const double radial = qmc::cpu::RadialRnl(qn.n, qn.l, r);
    pdf[static_cast<size_t>(i)] = r * r * radial * radial;
  }
  for (int i = 0; i < n_bins; ++i) {
    masses[static_cast<size_t>(i)] =
        cdf[static_cast<size_t>(i + 1)] - cdf[static_cast<size_t>(i)];
  }
  return BinsFromMasses(masses, nodes, pdf);
}

std::vector<AliasBin> BuildThetaAlias(const qmc::cpu::HydrogenSampler& sampler) {
  const auto nodes = sampler.ThetaNodes();
  const auto cdf = sampler.ThetaCdf();
  const int n_nodes = static_cast<int>(nodes.size());
  const int n_bins = std::max(n_nodes - 1, 0);
  std::vector<double> masses(static_cast<size_t>(n_bins));
  std::vector<double> pdf(static_cast<size_t>(n_nodes));
  const qmc::cpu::QuantumNumbers qn = sampler.Numbers();
  const int m_abs = qn.m < 0 ? -qn.m : qn.m;
  for (int i = 0; i < n_nodes; ++i) {
    const double theta = nodes[static_cast<size_t>(i)];
    const double plm =
        qmc::cpu::AssociatedLegendrePositiveM(qn.l, m_abs, std::cos(theta));
    pdf[static_cast<size_t>(i)] = std::sin(theta) * plm * plm;
  }
  for (int i = 0; i < n_bins; ++i) {
    masses[static_cast<size_t>(i)] =
        cdf[static_cast<size_t>(i + 1)] - cdf[static_cast<size_t>(i)];
  }
  return BinsFromMasses(masses, nodes, pdf);
}

}  // namespace qmc::alias
