#ifndef CPU_REFERENCE_ORIGINAL_BUGGY_H_
#define CPU_REFERENCE_ORIGINAL_BUGGY_H_

#include <random>

namespace qmc::cpu::buggy {

double SampleRadius(int n, int l, std::mt19937& gen);
double SampleTheta(int l, int m, std::mt19937& gen);
float SamplePhi();
void ResetTables();

}  // namespace qmc::cpu::buggy

#endif
