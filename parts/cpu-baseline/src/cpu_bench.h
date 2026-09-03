#ifndef PARTS_CPU_BASELINE_SRC_CPU_BENCH_H_
#define PARTS_CPU_BASELINE_SRC_CPU_BENCH_H_

#include <string>

namespace qmc::cpu {

struct CpuBenchOptions {
  int n = 8000000;
  int threads = 0;
  int principal = 1;
  int l = 0;
  int m = 0;
  std::string out_path;
  bool scratch = true;
};

int RunCpuBench(const CpuBenchOptions& opt);

}  // namespace qmc::cpu

#endif
