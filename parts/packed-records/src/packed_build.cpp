#include "parts/packed-records/src/sampler.h"

#include <cstdint>

namespace qmc::packed {

PackedTable SplitAliasBins(std::span<const qmc::alias::AliasBin> bins) {
  PackedTable table;
  const std::size_t n = bins.size();
  table.draw.resize(n);
  table.uniform.resize(n);
  table.linear.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const qmc::alias::AliasBin& src = bins[i];
    table.draw[i].prob = src.prob;
    table.draw[i].alias = static_cast<std::uint32_t>(src.alias);
    table.uniform[i].lo = src.lo;
    table.uniform[i].width = src.width;
    table.linear[i].lo = src.lo;
    table.linear[i].width = src.width;
    table.linear[i].y0 = src.y0;
    table.linear[i].y1 = src.y1;
  }
  return table;
}

const char* KernelName(KernelKind kind) {
  switch (kind) {
    case KernelKind::Packed:
      return "packed";
    case KernelKind::PackedLinear:
      return "packed_linear";
    case KernelKind::PackedShared:
      return "packed_shared";
    case KernelKind::PackedIlp:
      return "packed_ilp";
  }
  return "unknown";
}

std::size_t SharedBytesFor(int n_radial, int n_theta) {
  return (static_cast<std::size_t>(n_radial) +
          static_cast<std::size_t>(n_theta)) *
         sizeof(AliasDraw);
}

}  // namespace qmc::packed
