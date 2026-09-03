#ifndef HARNESS_VERIFY_GATE_H_
#define HARNESS_VERIFY_GATE_H_

#include <string>

namespace qmc::harness {

[[nodiscard]] std::string VerifyFullPath();
[[nodiscard]] bool VerifyFullPassed();
[[nodiscard]] bool RecordVerifyFullOk();
void ClearVerifyFull();

}  // namespace qmc::harness

#endif
