#ifndef HARNESS_ENV_H_
#define HARNESS_ENV_H_

#include <string>

namespace qmc::harness {

struct Env {
  std::string gpu;
  std::string driver;
  std::string cuda;
  std::string git;
  int clocks_sm_mhz = 0;
  int clocks_mem_mhz = 0;
  int temperature_c = 0;
  double power_w = 0.0;
  bool clocks_locked = false;
};

Env CaptureEnv();

}  // namespace qmc::harness

#endif
