#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "harness/microbench.h"
#include "harness/stats_smoke.h"
#include "parts/cpu-baseline/src/cpu_bench.h"
#include "parts/cpu-baseline/verify/check_reference.h"

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

  CLI11_PARSE(app, argc, argv);
  bench.scratch = !official;
  cpu.scratch = !cpu_official;

  if (smoke->parsed()) {
    return qmc::harness::RunStatsSmoke();
  }
  if (verify_fast->parsed()) {
    return qmc::cpu::verify::RunVerify(false);
  }
  if (verify_full->parsed()) {
    return qmc::cpu::verify::RunVerify(true);
  }
  if (micro->parsed()) {
    return qmc::harness::RunMicrobench(bench);
  }
  if (cpu_bench->parsed()) {
    return qmc::cpu::RunCpuBench(cpu);
  }
  fmt::print(stderr, "no subcommand\n");
  return 2;
}
