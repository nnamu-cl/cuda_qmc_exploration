#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "harness/microbench.h"
#include "harness/stats_smoke.h"
#include "harness/verify_gate.h"
#include "parts/cpu-baseline/src/cpu_bench.h"
#include "parts/cpu-baseline/verify/check_reference.h"
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

  CLI11_PARSE(app, argc, argv);
  bench.scratch = !official;
  cpu.scratch = !cpu_official;
  naive.scratch = !naive_official;
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
    return cpu_rc != 0 || gpu_rc != 0 ? 1 : 0;
  }
  if (verify_full->parsed()) {
    const int cpu_rc = qmc::cpu::verify::RunVerify(true);
    const int gpu_rc = qmc::naive::verify::RunVerify(true);
    if (cpu_rc != 0 || gpu_rc != 0) {
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
  fmt::print(stderr, "no subcommand\n");
  return 2;
}
