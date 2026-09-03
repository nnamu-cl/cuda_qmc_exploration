#ifndef HARNESS_MICROBENCH_H_
#define HARNESS_MICROBENCH_H_

#include <string>

namespace qmc::harness {

struct MicrobenchOptions {
  std::string part = "cpu-baseline";
  std::string kind = "bw_copy";
  int n = 1 << 24;
  std::string out_path;
  bool scratch = true;
};

int RunMicrobench(const MicrobenchOptions& opt);

}  // namespace qmc::harness

#endif
