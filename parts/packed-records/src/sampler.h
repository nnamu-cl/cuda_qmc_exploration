#ifndef PARTS_PACKED_RECORDS_SRC_SAMPLER_H_
#define PARTS_PACKED_RECORDS_SRC_SAMPLER_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "cpu-reference/hydrogen.h"
#include "parts/alias-method/src/sampler.h"

namespace qmc::packed {

enum class KernelKind { Packed, PackedLinear, PackedShared, PackedIlp };

// The alias draw itself. Vose 1991 needs exactly these two numbers; nothing
// else belongs in the record the fair-die probe touches. alignas(8) on an
// 8-byte POD is what licenses nvcc to emit a single LDG.E.64.
struct alignas(8) AliasDraw {
  float prob = 0.0f;
  std::uint32_t alias = 0;
};
static_assert(sizeof(AliasDraw) == 8, "AliasDraw must be one 64-bit load");
static_assert(alignof(AliasDraw) == 8, "AliasDraw must be 8-byte aligned");

// Within-bin geometry, uniform rule: one aligned 64-bit load of the winner.
struct alignas(8) BinUniform {
  float lo = 0.0f;
  float width = 0.0f;
};
static_assert(sizeof(BinUniform) == 8, "BinUniform must be one 64-bit load");

// Within-bin geometry, linear rule: one aligned 128-bit load of the winner.
struct alignas(16) BinLinear {
  float lo = 0.0f;
  float width = 0.0f;
  float y0 = 0.0f;
  float y1 = 0.0f;
};
static_assert(sizeof(BinLinear) == 16, "BinLinear must be one 128-bit load");

// A K5/K7 AliasBin vector, transposed into the three arrays above. The values
// are copied, never recomputed: the Vose tables are not forked here.
struct PackedTable {
  std::vector<AliasDraw> draw;
  std::vector<BinUniform> uniform;
  std::vector<BinLinear> linear;
};

[[nodiscard]] PackedTable SplitAliasBins(
    std::span<const qmc::alias::AliasBin> bins);

struct GpuDraw {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> density;
  std::vector<double> r;
  std::vector<double> theta;
  std::vector<double> phi;
};

struct KernelTiming {
  double setup_ms = 0.0;
  std::vector<double> reps_ms;
  double median_ms = 0.0;
  double bytes_per_sample = 0.0;
  int occupancy_blocks = 0;
  int grid_blocks = 0;
  std::size_t shared_bytes = 0;
};

struct LaunchConfig {
  int threads = 256;
  int samples_per_thread = 1;
  int grid_blocks = 0;
};

struct BenchOptions {
  int n = 8000000;
  int principal = 1;
  int l = 0;
  int m = 0;
  KernelKind kernel = KernelKind::Packed;
  LaunchConfig launch;
  std::string out_path;
  bool scratch = true;
};

// Destination pointers. Either `packed` (16 B float4 store) or the SoA floats
// are used; `r`/`theta`/`phi` are the verification-only spherical dump.
struct Outputs {
  qmc::alias::PackedXyzw* packed = nullptr;
  float* x = nullptr;
  float* y = nullptr;
  float* z = nullptr;
  float* density = nullptr;
  float* r = nullptr;
  float* theta = nullptr;
  float* phi = nullptr;
};

struct Params {
  int n = 0;
  int principal = 1;
  int l = 0;
  int m_abs = 0;
  float radial_norm = 0.0f;
  float y_norm = 0.0f;
  std::uint64_t seed = 0;
};

void LaunchPacked(const AliasDraw* radial_draw, const BinUniform* radial_bin,
                  int n_radial, const AliasDraw* theta_draw,
                  const BinUniform* theta_bin, int n_theta, const Outputs& out,
                  const Params& params, const LaunchConfig& cfg);

void LaunchPackedLinear(const AliasDraw* radial_draw,
                        const BinLinear* radial_bin, int n_radial,
                        const AliasDraw* theta_draw, const BinLinear* theta_bin,
                        int n_theta, const Outputs& out, const Params& params,
                        const LaunchConfig& cfg);

void LaunchPackedShared(const AliasDraw* radial_draw,
                        const BinUniform* radial_bin, int n_radial,
                        const AliasDraw* theta_draw, const BinUniform* theta_bin,
                        int n_theta, const Outputs& out, const Params& params,
                        const LaunchConfig& cfg);

void LaunchPackedIlp(const AliasDraw* radial_draw, const BinUniform* radial_bin,
                     int n_radial, const AliasDraw* theta_draw,
                     const BinUniform* theta_bin, int n_theta,
                     const Outputs& out, const Params& params,
                     const LaunchConfig& cfg);

[[nodiscard]] const char* KernelName(KernelKind kind);
[[nodiscard]] std::size_t SharedBytesFor(int n_radial, int n_theta);
[[nodiscard]] int SharedOccupancyBlocks(int threads, std::size_t shared_bytes);
[[nodiscard]] int IlpOccupancyBlocks(int samples_per_thread, int threads);
[[nodiscard]] int GridBlocksFor(KernelKind kind, const LaunchConfig& cfg,
                                std::size_t shared_bytes);

void DrawGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn, std::uint64_t seed,
             int n, GpuDraw* out, const LaunchConfig& cfg = {});
[[nodiscard]] KernelTiming TimeGpu(KernelKind kind, qmc::cpu::QuantumNumbers qn,
                                   std::uint64_t seed, int n,
                                   const LaunchConfig& cfg = {});

// Timed runs reuse one process-lifetime output buffer by default; repeated
// cudaMalloc/cudaFree of ~128 MB measurably slows later cells in the same
// process. Turn it off only to reproduce that effect.
void SetScratchReuse(bool on);

int RunPackedBench(const BenchOptions& opt);
int RunOrbitalSweep(const std::string& out_path, int n, bool scratch);
int RunLayoutSweep(const std::string& out_path, int n, bool scratch);
int RunAllocChurn(const std::string& out_path, int n, bool scratch);

}  // namespace qmc::packed

#endif
