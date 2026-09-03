#include <algorithm>

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "harness/microbench.h"
#include "harness/stats_smoke.h"
#include "harness/verify_gate.h"
#include "parts/cpu-baseline/src/cpu_bench.h"
#include "parts/cpu-baseline/verify/check_reference.h"
#include "parts/inverse-table/src/sampler.h"
#include "parts/inverse-table/verify/check_inverse.h"
#include "parts/naive-cuda/src/sampler.h"
#include "parts/naive-cuda/verify/check_naive.h"

int main(int argc, char** argv) {
  CLI::App app{"Hydrogen orbital sampling worklog"};
  app.require_subcommand(1);

  auto* smoke = app.add_subcommand("stats-smoke", "KS / chi-square / moment smoke");
  auto* verify_fast =
      app.add_subcommand("verify-fast", "goldens, bug hunt, 1e5-sample smoke");
  auto* verify_full =
      app.add_subcommand("verify-full", "1e7-sample matrix; writes the results gate");

  qmc::harness::MicrobenchOptions bench;
  auto* micro = app.add_subcommand("microbench", "Referee kernels: copy, FMA, cuRAND");
  micro->add_option("--part", bench.part, "results JSON part key")
      ->capture_default_str();
  micro->add_option("--kind", bench.kind,
                    "bw_copy | fma32 | fma64 | curand_philox | curand_xorwow")
      ->capture_default_str();
  micro->add_option("--n", bench.n, "elements")->capture_default_str();
  micro->add_option("--out", bench.out_path, "write JSON here");
  bool official = false;
  micro->add_flag("--official", official, "refuse JSON unless verify-full passed");

  qmc::cpu::CpuBenchOptions cpu;
  auto* cpu_bench = app.add_subcommand("cpu-bench", "FP64 CPU sampler throughput");
  cpu_bench->add_option("--n", cpu.n, "samples")->capture_default_str();
  cpu_bench->add_option("--threads", cpu.threads,
                        "1 = serial; 0 = every core; >1 pin OpenMP")
      ->capture_default_str();
  cpu_bench->add_option("--principal", cpu.principal, "n quantum number")
      ->capture_default_str();
  cpu_bench->add_option("--l", cpu.l, "l quantum number")->capture_default_str();
  cpu_bench->add_option("--m", cpu.m, "m quantum number")->capture_default_str();
  cpu_bench->add_option("--out", cpu.out_path, "write JSON here");
  bool cpu_official = false;
  cpu_bench->add_flag("--official", cpu_official,
                      "refuse JSON unless verify-full passed");

  qmc::naive::NaiveBenchOptions naive;
  std::string naive_kernel = "naive_fp64";
  auto* naive_bench =
      app.add_subcommand("naive-bench", "Naive CUDA sampler throughput");
  naive_bench->add_option("--n", naive.n, "samples")->capture_default_str();
  naive_bench->add_option("--kernel", naive_kernel,
                          "naive_fp64 | naive_fp32 | naive_fp32_fast")
      ->capture_default_str();
  naive_bench->add_option("--principal", naive.principal, "n quantum number")
      ->capture_default_str();
  naive_bench->add_option("--l", naive.l, "l quantum number")
      ->capture_default_str();
  naive_bench->add_option("--m", naive.m, "m quantum number")
      ->capture_default_str();
  naive_bench->add_option("--out", naive.out_path, "write JSON here");
  bool naive_official = false;
  naive_bench->add_flag("--official", naive_official,
                        "refuse JSON unless verify-full passed");

  std::string error_out;
  auto* naive_error = app.add_subcommand(
      "naive-error", "FP32 vs FP64 special-function error table");
  naive_error->add_option("--out", error_out, "write JSON here");

  qmc::inverse::BenchOptions inverse;
  std::string inverse_kernel = "shared_cdf";
  auto* inverse_bench =
      app.add_subcommand("inverse-bench", "Shared-CDF / inverse-table throughput");
  inverse_bench->add_option("--n", inverse.n, "samples")->capture_default_str();
  inverse_bench->add_option("--kernel", inverse_kernel,
                            "shared_cdf | inverse_table | inverse_nodes")
      ->capture_default_str();
  inverse_bench->add_option("--principal", inverse.principal, "n quantum number")
      ->capture_default_str();
  inverse_bench->add_option("--l", inverse.l, "l quantum number")
      ->capture_default_str();
  inverse_bench->add_option("--m", inverse.m, "m quantum number")
      ->capture_default_str();
  inverse_bench->add_option("--threads", inverse.launch.threads, "block size")
      ->capture_default_str();
  inverse_bench->add_flag("--cache-nodes", inverse.launch.cache_nodes,
                         "also stage node arrays in shared memory");
  inverse_bench->add_option("--table-k", inverse.launch.table_k_radial,
                           "quantile table size for r (theta is k/2)")
      ->capture_default_str();
  inverse_bench->add_option("--out", inverse.out_path, "write JSON here");
  bool inverse_official = false;
  inverse_bench->add_flag("--official", inverse_official,
                         "refuse JSON unless verify-full passed");

  std::string trap_out;
  int trap_n = 1000000;
  auto* inverse_trap = app.add_subcommand(
      "inverse-trap", "Quantile-K sweep χ² on the (3,1) radial node");
  inverse_trap->add_option("--n", trap_n, "samples")->capture_default_str();
  inverse_trap->add_option("--out", trap_out, "write JSON here");

  std::string occ_out;
  int occ_n = 8000000;
  auto* inverse_occ = app.add_subcommand(
      "inverse-occupancy", "Shared-memory occupancy vs block size");
  inverse_occ->add_option("--n", occ_n, "samples")->capture_default_str();
  inverse_occ->add_option("--out", occ_out, "write JSON here");

  CLI11_PARSE(app, argc, argv);
  bench.scratch = !official;
  cpu.scratch = !cpu_official;
  naive.scratch = !naive_official;
  inverse.scratch = !inverse_official;
  inverse.launch.table_k_theta = std::max(inverse.launch.table_k_radial / 2, 2);
  if (inverse_kernel == "inverse_table" || inverse_kernel == "table") {
    inverse.kernel = qmc::inverse::KernelKind::InverseTable;
  } else if (inverse_kernel == "inverse_nodes" || inverse_kernel == "nodes") {
    inverse.kernel = qmc::inverse::KernelKind::InverseNodes;
  } else {
    inverse.kernel = qmc::inverse::KernelKind::SharedCdf;
  }
  if (naive_kernel == "naive_fp32" || naive_kernel == "fp32") {
    naive.kernel = qmc::naive::KernelKind::Fp32;
  } else if (naive_kernel == "naive_fp32_fast" || naive_kernel == "fp32_fast") {
    naive.kernel = qmc::naive::KernelKind::Fp32Fast;
  } else {
    naive.kernel = qmc::naive::KernelKind::Fp64;
  }

  if (smoke->parsed()) {
    return qmc::harness::RunStatsSmoke();
  }
  if (verify_fast->parsed()) {
    const int cpu_rc = qmc::cpu::verify::RunVerify(false);
    const int gpu_rc = qmc::naive::verify::RunVerify(false);
    const int inv_rc = qmc::inverse::verify::RunVerify(false);
    return cpu_rc != 0 || gpu_rc != 0 || inv_rc != 0 ? 1 : 0;
  }
  if (verify_full->parsed()) {
    const int cpu_rc = qmc::cpu::verify::RunVerify(true);
    const int gpu_rc = qmc::naive::verify::RunVerify(true);
    const int inv_rc = qmc::inverse::verify::RunVerify(true);
    if (cpu_rc != 0 || gpu_rc != 0 || inv_rc != 0) {
      qmc::harness::ClearVerifyFull();
      return 1;
    }
    if (!qmc::harness::RecordVerifyFullOk()) {
      fmt::print(stderr, "could not write verify-full gate\n");
      return 1;
    }
    return 0;
  }
  if (micro->parsed()) {
    return qmc::harness::RunMicrobench(bench);
  }
  if (cpu_bench->parsed()) {
    return qmc::cpu::RunCpuBench(cpu);
  }
  if (naive_bench->parsed()) {
    return qmc::naive::RunNaiveBench(naive);
  }
  if (naive_error->parsed()) {
    return qmc::naive::RunFp32ErrorAnalysis(error_out);
  }
  if (inverse_bench->parsed()) {
    return qmc::inverse::RunInverseBench(inverse);
  }
  if (inverse_trap->parsed()) {
    return qmc::inverse::RunNodeTrap(trap_out, trap_n, true);
  }
  if (inverse_occ->parsed()) {
    return qmc::inverse::RunOccupancySweep(occ_out, occ_n, true);
  }
  fmt::print(stderr, "no subcommand\n");
  return 2;
}
