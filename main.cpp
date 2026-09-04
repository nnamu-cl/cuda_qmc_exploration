#include <algorithm>

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "harness/microbench.h"
#include "harness/stats_smoke.h"
#include "harness/verify_gate.h"
#include "parts/cpu-baseline/src/cpu_bench.h"
#include "parts/cpu-baseline/verify/check_reference.h"
#include "parts/alias-method/src/sampler.h"
#include "parts/alias-method/verify/check_alias.h"
#include "parts/inverse-table/src/sampler.h"
#include "parts/inverse-table/verify/check_inverse.h"
#include "parts/endgame/src/sampler.h"
#include "parts/endgame/verify/check_endgame.h"
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

  qmc::alias::BenchOptions alias;
  std::string alias_kernel = "alias_linear";
  auto* alias_bench =
      app.add_subcommand("alias-bench", "Walker/Vose alias sampler throughput");
  alias_bench->add_option("--n", alias.n, "samples")->capture_default_str();
  alias_bench->add_option("--kernel", alias_kernel,
                          "alias_uniform | alias_linear | alias_float4 | alias_split")
      ->capture_default_str();
  alias_bench->add_option("--principal", alias.principal, "n quantum number")
      ->capture_default_str();
  alias_bench->add_option("--l", alias.l, "l quantum number")->capture_default_str();
  alias_bench->add_option("--m", alias.m, "m quantum number")->capture_default_str();
  alias_bench->add_option("--threads", alias.launch.threads, "block size")
      ->capture_default_str();
  alias_bench->add_option("--out", alias.out_path, "write JSON here");
  bool alias_official = false;
  alias_bench->add_flag("--official", alias_official,
                        "refuse JSON unless verify-full passed");

  std::string interior_out;
  int interior_n = 1000000;
  auto* alias_interior = app.add_subcommand(
      "alias-interior", "Uniform vs linear-within-bin radial χ²");
  alias_interior->add_option("--n", interior_n, "samples")->capture_default_str();
  alias_interior->add_option("--out", interior_out, "write JSON here");

  std::string hole_out;
  int hole_n = 10000000;
  auto* alias_hole = app.add_subcommand(
      "alias-hole", "(3,1) node-window χ² vs inverse-table");
  alias_hole->add_option("--n", hole_n, "samples")->capture_default_str();
  alias_hole->add_option("--out", hole_out, "write JSON here");

  qmc::endgame::BenchOptions endgame;
  std::string endgame_kernel = "philox";
  auto* endgame_bench =
      app.add_subcommand("endgame-bench", "Philox alias sampler throughput");
  endgame_bench->add_option("--n", endgame.n, "samples")->capture_default_str();
  endgame_bench->add_option("--kernel", endgame_kernel,
                            "philox | philox_ilp | thrust_alias")
      ->capture_default_str();
  endgame_bench->add_option("--principal", endgame.principal, "n quantum number")
      ->capture_default_str();
  endgame_bench->add_option("--l", endgame.l, "l quantum number")
      ->capture_default_str();
  endgame_bench->add_option("--m", endgame.m, "m quantum number")
      ->capture_default_str();
  endgame_bench->add_option("--threads", endgame.launch.threads, "block size")
      ->capture_default_str();
  endgame_bench
      ->add_option("--samples-per-thread", endgame.launch.samples_per_thread,
                   "independent samples in flight (K8)")
      ->capture_default_str();
  endgame_bench->add_flag("--persist-tables", endgame.launch.persist_tables,
                          "L2 persistence hint on the radial alias table");
  endgame_bench->add_option("--out", endgame.out_path, "write JSON here");
  bool endgame_official = false;
  endgame_bench->add_flag("--official", endgame_official,
                          "refuse JSON unless verify-full passed");

  std::string sweep_out;
  int sweep_n = 8000000;
  auto* endgame_sweep = app.add_subcommand(
      "endgame-sweep", "S x block-size launch table for Philox ILP");
  endgame_sweep->add_option("--n", sweep_n, "samples")->capture_default_str();
  endgame_sweep->add_option("--out", sweep_out, "write JSON here");

  std::string bits_out;
  int bits_n = 4096;
  auto* endgame_bits = app.add_subcommand(
      "endgame-bits", "Philox host/device/cuRAND bit check");
  endgame_bits->add_option("--n", bits_n, "counters")->capture_default_str();
  endgame_bits->add_option("--out", bits_out, "write JSON here");

  std::string dump_prefix;
  int dump_n = 1000000;
  int dump_principal = 3;
  int dump_l = 1;
  int dump_m = -1;
  auto* endgame_dump = app.add_subcommand(
      "endgame-dump", "JSON metadata + float4 sample binary");
  endgame_dump->add_option("--n", dump_n, "samples")->capture_default_str();
  endgame_dump->add_option("--prefix", dump_prefix, "output prefix")->required();
  endgame_dump->add_option("--principal", dump_principal, "n quantum number")
      ->capture_default_str();
  endgame_dump->add_option("--l", dump_l, "l quantum number")
      ->capture_default_str();
  endgame_dump->add_option("--m", dump_m, "m quantum number")
      ->capture_default_str();

  CLI11_PARSE(app, argc, argv);
  bench.scratch = !official;
  cpu.scratch = !cpu_official;
  naive.scratch = !naive_official;
  inverse.scratch = !inverse_official;
  alias.scratch = !alias_official;
  endgame.scratch = !endgame_official;
  if (endgame_kernel == "philox_ilp" || endgame_kernel == "ilp") {
    endgame.kernel = qmc::endgame::KernelKind::PhiloxIlp;
  } else if (endgame_kernel == "thrust_alias" || endgame_kernel == "thrust") {
    endgame.kernel = qmc::endgame::KernelKind::ThrustAlias;
  } else {
    endgame.kernel = qmc::endgame::KernelKind::Philox;
  }
  if (alias_kernel == "alias_uniform" || alias_kernel == "uniform") {
    alias.kernel = qmc::alias::KernelKind::Uniform;
  } else if (alias_kernel == "alias_float4" || alias_kernel == "float4") {
    alias.kernel = qmc::alias::KernelKind::Float4;
  } else if (alias_kernel == "alias_split" || alias_kernel == "split") {
    alias.kernel = qmc::alias::KernelKind::Split;
  } else {
    alias.kernel = qmc::alias::KernelKind::Linear;
  }
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
    const int alias_rc = qmc::alias::verify::RunVerify(false);
    const int end_rc = qmc::endgame::verify::RunVerify(false);
    return cpu_rc != 0 || gpu_rc != 0 || inv_rc != 0 || alias_rc != 0 ||
                   end_rc != 0
               ? 1
               : 0;
  }
  if (verify_full->parsed()) {
    const int cpu_rc = qmc::cpu::verify::RunVerify(true);
    const int gpu_rc = qmc::naive::verify::RunVerify(true);
    const int inv_rc = qmc::inverse::verify::RunVerify(true);
    const int alias_rc = qmc::alias::verify::RunVerify(true);
    const int end_rc = qmc::endgame::verify::RunVerify(true);
    if (cpu_rc != 0 || gpu_rc != 0 || inv_rc != 0 || alias_rc != 0 ||
        end_rc != 0) {
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
  if (alias_bench->parsed()) {
    return qmc::alias::RunAliasBench(alias);
  }
  if (alias_interior->parsed()) {
    return qmc::alias::RunInteriorCompare(interior_out, interior_n, true);
  }
  if (alias_hole->parsed()) {
    return qmc::alias::RunNodeHole(hole_out, hole_n, true);
  }
  if (endgame_bench->parsed()) {
    return qmc::endgame::RunEndgameBench(endgame);
  }
  if (endgame_sweep->parsed()) {
    return qmc::endgame::RunLaunchSweep(sweep_out, sweep_n, true);
  }
  if (endgame_bits->parsed()) {
    return qmc::endgame::RunBitCheck(bits_out, bits_n, true);
  }
  if (endgame_dump->parsed()) {
    return qmc::endgame::RunSampleDump(
        dump_prefix, dump_n, {dump_principal, dump_l, dump_m}, 0xC0FFEEULL);
  }
  fmt::print(stderr, "no subcommand\n");
  return 2;
}
