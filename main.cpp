#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "harness/microbench.h"
#include "harness/stats_smoke.h"

int main(int argc, char** argv) {
  CLI::App app{"Hydrogen orbital sampling worklog"};
  app.require_subcommand(1);

  auto* smoke = app.add_subcommand("stats-smoke", "KS / chi-square / moment smoke");

  qmc::harness::MicrobenchOptions bench;
  auto* micro = app.add_subcommand("microbench", "Referee kernels: copy, FMA, cuRAND");
  micro->add_option("--kind", bench.kind,
                    "bw_copy | fma32 | fma64 | curand_philox | curand_xorwow")
      ->capture_default_str();
  micro->add_option("--n", bench.n, "elements")->capture_default_str();
  micro->add_option("--out", bench.out_path, "write JSON here");
  bool official = false;
  micro->add_flag("--official", official, "refuse JSON unless verify-full passed");

  CLI11_PARSE(app, argc, argv);
  bench.scratch = !official;

  if (smoke->parsed()) {
    return qmc::harness::RunStatsSmoke();
  }
  if (micro->parsed()) {
    return qmc::harness::RunMicrobench(bench);
  }
  fmt::print(stderr, "no subcommand\n");
  return 2;
}
