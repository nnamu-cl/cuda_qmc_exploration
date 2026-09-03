#include "cpu-reference/hydrogen.h"

#include <algorithm>
#include <cmath>
#include <numbers>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "harness/rng/philox.h"

namespace qmc::cpu {
namespace {

double InvertCdf(std::span<const double> nodes, std::span<const double> cdf,
                 double u) {
  if (u <= cdf.front()) {
    return nodes.front();
  }
  if (u >= cdf.back()) {
    return nodes.back();
  }
  const auto it = std::ranges::lower_bound(cdf, u);
  const auto i = static_cast<size_t>(it - cdf.begin());
  if (i == 0) {
    return nodes.front();
  }
  const double c0 = cdf[i - 1];
  const double c1 = cdf[i];
  const double t = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.0;
  return nodes[i - 1] + t * (nodes[i] - nodes[i - 1]);
}

double EvalCdf(std::span<const double> nodes, std::span<const double> cdf,
               double x) {
  if (x <= nodes.front()) {
    return cdf.front();
  }
  if (x >= nodes.back()) {
    return cdf.back();
  }
  const auto it = std::ranges::lower_bound(nodes, x);
  const auto i = static_cast<size_t>(it - nodes.begin());
  if (i == 0) {
    return cdf.front();
  }
  const double width = nodes[i] - nodes[i - 1];
  const double t = (width > 0.0) ? (x - nodes[i - 1]) / width : 0.0;
  return cdf[i - 1] + t * (cdf[i] - cdf[i - 1]);
}

void NormalizeCdf(std::vector<double>& cdf) {
  const double z = cdf.back();
  for (double& value : cdf) {
    value /= z;
  }
  cdf.back() = 1.0;
}

void BuildRadial(int n, int l, int n_bins, std::vector<double>& nodes,
                 std::vector<double>& cdf) {
  const double r_max = 10.0 * n * n * kBohrRadius;
  nodes.resize(static_cast<size_t>(n_bins));
  std::vector<double> pdf(static_cast<size_t>(n_bins));
  const double dr = r_max / (n_bins - 1);
  for (int i = 0; i < n_bins; ++i) {
    const double r = static_cast<double>(i) * dr;
    nodes[static_cast<size_t>(i)] = r;
    const double radial = RadialRnl(n, l, r);
    pdf[static_cast<size_t>(i)] = r * r * radial * radial;
  }
  cdf.assign(static_cast<size_t>(n_bins), 0.0);
  for (int i = 1; i < n_bins; ++i) {
    cdf[static_cast<size_t>(i)] =
        cdf[static_cast<size_t>(i - 1)] +
        0.5 * (pdf[static_cast<size_t>(i - 1)] + pdf[static_cast<size_t>(i)]) *
            dr;
  }
  NormalizeCdf(cdf);
}

void BuildTheta(int l, int m, int n_bins, std::vector<double>& nodes,
                std::vector<double>& cdf) {
  const int m_abs = m < 0 ? -m : m;
  nodes.resize(static_cast<size_t>(n_bins));
  std::vector<double> pdf(static_cast<size_t>(n_bins));
  const double dtheta = std::numbers::pi / (n_bins - 1);
  for (int i = 0; i < n_bins; ++i) {
    const double theta = static_cast<double>(i) * dtheta;
    nodes[static_cast<size_t>(i)] = theta;
    const double plm = AssociatedLegendrePositiveM(l, m_abs, std::cos(theta));
    pdf[static_cast<size_t>(i)] = std::sin(theta) * plm * plm;
  }
  cdf.assign(static_cast<size_t>(n_bins), 0.0);
  for (int i = 1; i < n_bins; ++i) {
    cdf[static_cast<size_t>(i)] =
        cdf[static_cast<size_t>(i - 1)] +
        0.5 * (pdf[static_cast<size_t>(i - 1)] + pdf[static_cast<size_t>(i)]) *
            dtheta;
  }
  NormalizeCdf(cdf);
}

}  // namespace

HydrogenSampler::HydrogenSampler(QuantumNumbers qn) : qn_(qn) {
  BuildRadial(qn.n, qn.l, kRadialBins, r_nodes_, r_cdf_);
  BuildTheta(qn.l, qn.m, kThetaBins, theta_nodes_, theta_cdf_);
  BuildRadial(qn.n, qn.l, kRadialBins * kFineCdfScale, r_nodes_fine_,
              r_cdf_fine_);
  BuildTheta(qn.l, qn.m, kThetaBins * kFineCdfScale, theta_nodes_fine_,
             theta_cdf_fine_);
}

double HydrogenSampler::InvertRadial(double u) const {
  return InvertCdf(r_nodes_, r_cdf_, u);
}

double HydrogenSampler::InvertTheta(double u) const {
  return InvertCdf(theta_nodes_, theta_cdf_, u);
}

HydrogenSample HydrogenSampler::Sample(std::uint64_t seed,
                                       std::uint32_t index) const {
  const qmc::rng::Philox4x32Key key = qmc::rng::SeedToKey(seed);
  const qmc::rng::Philox4x32Ctr ctr = qmc::rng::Philox4x32TenRounds(
      qmc::rng::MakeCounter(index, 0), key);
  HydrogenSample sample;
  sample.r = InvertCdf(r_nodes_, r_cdf_, qmc::rng::Uint32ToUnitDouble(ctr.v[0]));
  sample.theta = InvertCdf(theta_nodes_, theta_cdf_,
                           qmc::rng::Uint32ToUnitDouble(ctr.v[1]));
  sample.phi = 2.0 * std::numbers::pi *
               qmc::rng::Uint32ToUnitDouble(ctr.v[2]);
  const double sin_theta = std::sin(sample.theta);
  const double cos_theta = std::cos(sample.theta);
  const double sin_phi = std::sin(sample.phi);
  const double cos_phi = std::cos(sample.phi);
  sample.x = sample.r * sin_theta * cos_phi;
  sample.y = sample.r * sin_theta * sin_phi;
  sample.z = sample.r * cos_theta;
  sample.density =
      WavefunctionDensity(qn_.n, qn_.l, qn_.m, sample.r, sample.theta);
  return sample;
}

void HydrogenSampler::SampleMany(std::uint64_t seed,
                                 std::span<HydrogenSample> out,
                                 bool parallel) const {
  const int n = static_cast<int>(out.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
  for (int i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = Sample(seed, static_cast<std::uint32_t>(i));
  }
#else
  (void)parallel;
  for (int i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = Sample(seed, static_cast<std::uint32_t>(i));
  }
#endif
}

double HydrogenSampler::RadialCdf(double r) const {
  return EvalCdf(r_nodes_fine_, r_cdf_fine_, r);
}

double HydrogenSampler::ThetaCdf(double theta) const {
  return EvalCdf(theta_nodes_fine_, theta_cdf_fine_, theta);
}

}  // namespace qmc::cpu
