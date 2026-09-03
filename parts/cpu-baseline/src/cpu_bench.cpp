#include "parts/cpu-baseline/src/cpu_bench.h"

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fmt/format.h>

#include "cpu-reference/hydrogen.h"
#include "harness/env.h"
#include "harness/results.h"
#include "harness/timing.h"

namespace qmc::cpu {

int RunCpuBench(const CpuBenchOptions& opt) {
#ifdef _OPENMP
  if (opt.threads > 0) {
    omp_set_num_threads(opt.threads);
  }
  const bool parallel = opt.threads != 1;
  const int threads_used = parallel ? omp_get_max_threads() : 1;
#else
  const bool parallel = false;
  const int threads_used = 1;
#endif
  const HydrogenSampler sampler({opt.principal, opt.l, opt.m});
  std::vector<HydrogenSample> out(static_cast<size_t>(opt.n));
  sampler.SampleMany(0xC0FFEEULL, out, parallel);

  const auto timing = qmc::harness::TimeHost({}, [&] {
    sampler.SampleMany(0xC0FFEEULL, out, parallel);
  });

  qmc::harness::BenchmarkRecord rec;
  rec.part = "cpu-baseline";
  rec.kernel = parallel ? "cpu_openmp" : "cpu_single";
  rec.workload = "hydrogen_samples";
  rec.count = opt.n;
  rec.quantum_numbers = {opt.principal, opt.l, opt.m};
  rec.reps_ms = timing.reps_ms;
  rec.median_ms = timing.median_ms;
  rec.bytes_per_unit = 32.0;
  rec.units_per_s = static_cast<double>(opt.n) / (timing.median_ms * 1e-3);
  rec.achieved_gbps =
      (static_cast<double>(opt.n) * rec.bytes_per_unit) /
      (timing.median_ms * 1e-3) / 1e9;
  rec.env = qmc::harness::CaptureEnv();

  fmt::print(
      "{} threads={} n={} median_ms={:.3f} Msamples/s={:.3f} "
      "GB/s={:.4f} locked={}\n",
      rec.kernel, threads_used, opt.n, rec.median_ms, rec.units_per_s / 1e6,
      rec.achieved_gbps, rec.env.clocks_locked ? 1 : 0);

  if (!opt.out_path.empty()) {
    if (const auto written =
            qmc::harness::WriteJson(opt.out_path, rec, !opt.scratch);
        !written) {
      fmt::print(stderr, "{}\n", written.error());
      return 1;
    }
  }
  return 0;
}

}  // namespace qmc::cpu
