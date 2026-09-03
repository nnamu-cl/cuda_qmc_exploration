#ifndef HARNESS_RNG_PHILOX_H_
#define HARNESS_RNG_PHILOX_H_

#include <cstdint>

#if defined(__CUDACC__)
#define QMC_HOST_DEVICE __host__ __device__
#else
#define QMC_HOST_DEVICE
#endif

namespace qmc::rng {

inline constexpr std::uint32_t kPhiloxM0 = 0xD2511F53u;
inline constexpr std::uint32_t kPhiloxM1 = 0xCD9E8D57u;
inline constexpr std::uint32_t kPhiloxW0 = 0x9E3779B9u;
inline constexpr std::uint32_t kPhiloxW1 = 0xBB67AE85u;

struct Philox4x32Ctr {
  std::uint32_t v[4];
};

struct Philox4x32Key {
  std::uint32_t v[2];
};

QMC_HOST_DEVICE inline std::uint32_t MulHiLo32(std::uint32_t a, std::uint32_t b,
                                               std::uint32_t* lo) {
  const std::uint64_t product =
      static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
  *lo = static_cast<std::uint32_t>(product);
  return static_cast<std::uint32_t>(product >> 32);
}

QMC_HOST_DEVICE inline Philox4x32Ctr Philox4x32Round(Philox4x32Ctr ctr,
                                                     Philox4x32Key key) {
  std::uint32_t lo0 = 0;
  std::uint32_t lo1 = 0;
  const std::uint32_t hi0 = MulHiLo32(kPhiloxM0, ctr.v[0], &lo0);
  const std::uint32_t hi1 = MulHiLo32(kPhiloxM1, ctr.v[2], &lo1);
  Philox4x32Ctr out;
  out.v[0] = hi1 ^ ctr.v[1] ^ key.v[0];
  out.v[1] = lo1;
  out.v[2] = hi0 ^ ctr.v[3] ^ key.v[1];
  out.v[3] = lo0;
  return out;
}

QMC_HOST_DEVICE inline Philox4x32Key BumpKey(Philox4x32Key key) {
  key.v[0] += kPhiloxW0;
  key.v[1] += kPhiloxW1;
  return key;
}

QMC_HOST_DEVICE inline Philox4x32Ctr Philox4x32TenRounds(Philox4x32Ctr ctr,
                                                         Philox4x32Key key) {
  for (int round = 0; round < 10; ++round) {
    ctr = Philox4x32Round(ctr, key);
    key = BumpKey(key);
  }
  return ctr;
}

QMC_HOST_DEVICE inline float Uint32ToUnitFloat(std::uint32_t x) {
  return (static_cast<int>(x >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

QMC_HOST_DEVICE inline Philox4x32Key SeedToKey(std::uint64_t seed) {
  Philox4x32Key key;
  key.v[0] = static_cast<std::uint32_t>(seed);
  key.v[1] = static_cast<std::uint32_t>(seed >> 32);
  return key;
}

QMC_HOST_DEVICE inline Philox4x32Ctr MakeCounter(std::uint32_t sample_index,
                                                 std::uint32_t draw_slot) {
  Philox4x32Ctr ctr;
  ctr.v[0] = sample_index;
  ctr.v[1] = draw_slot;
  ctr.v[2] = 0;
  ctr.v[3] = 0;
  return ctr;
}

}  // namespace qmc::rng

#endif
