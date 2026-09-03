#include "harness/verify_gate.h"

#include <cstdlib>
#include <fstream>
#include <unistd.h>

namespace qmc::harness {

std::string VerifyFullPath() {
  const char* from_env = std::getenv("QMC_VERIFY_GATE");
  if (from_env != nullptr && from_env[0] != '\0') {
    return from_env;
  }
#ifdef QMC_DEFAULT_VERIFY_GATE
  return QMC_DEFAULT_VERIFY_GATE;
#else
  return ".verify_full_ok";
#endif
}

bool VerifyFullPassed() {
  return access(VerifyFullPath().c_str(), F_OK) == 0;
}

bool RecordVerifyFullOk() {
  std::ofstream out(VerifyFullPath());
  out << "ok\n";
  return static_cast<bool>(out);
}

}  // namespace qmc::harness
