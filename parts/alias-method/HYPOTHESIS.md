# Hypothesis — Walker/Vose alias sampler

## Claim
Kernel 5 (Walker/Vose alias on the binned radial and θ masses, linear-within-bin) is a **correctness** win, not a throughput win. It will sit near **2.6 Gsamples/s**, still limited by XORWOW 48 B state + 16 B output — the same 112 B DRAM roof K4 already occupies. Alias is already O(1); K4 was already O(1) and already at 86% DRAM. Kernel 6 (float4 store; fused vs split) will not lift that roof. Split pays an extra ~12 B/sample round trip and should land near **2.4 Gsamples/s**. Do not start Philox.

## Arithmetic
Referees are **measured** Release JSON and **ncu stall names**, same RTX 2070 Super Max-Q, SM 1500 MHz. ncu replay sat at ~1.17 GHz — stalls and ratios only, never mixed with JSON Gsamples/s.

| Referee | File | Number |
|---|---|---|
| inverse table (K4) | `parts/inverse-table/results/inverse_table.json` | **2.72 Gsamples/s** |
| inverse nodes | `parts/inverse-table/results/inverse_nodes.json` | 2.66 Gsamples/s |
| naive FP32 (K2) | `parts/naive-cuda/results/naive_fp32.json` | 1.78 Gsamples/s |
| `bw_copy` | `parts/cpu-baseline/results/bw_copy.json` | **305 GB/s** |
| `fma32` | cpu-baseline results | 3.75 TFLOPS |
| cuRAND XORWOW | `parts/cpu-baseline/results/curand_xorwow.json` | 67.1 Gsamples/s |

ncu on K4 (`parts/inverse-table/results/nsight/reading.md`), not vibes:

| Fact | Number |
|---|---|
| DRAM / mem GB/s | **85.7%** / **300** |
| FP32 peak used | **5%** |
| Top stall | **LG throttle 73%** (XORWOW 48 B) |
| Uncoalesced extra sectors | 77% |
| Branch efficiency | 100% |
| (3,1) node-window χ² | **11572** at 1e7, K=4096 uniform |

XORWOW setup stays a separate kernel (~163 ms / 8e6). Not this part.

Work per sample after tables exist, FP32, tutorial XORWOW (48 B load + 48 B store):

- Walker generation (Schwarz / Walker 1977): fair die + biased coin per axis, then one uniform inside the bin, plus φ. Honest count is **7 uniforms** (2+1 radial, 2+1 θ, 1 φ), not the plan’s “likely 6.” Extra uniforms are extra XORWOW **rounds on already-resident state**, not extra DRAM. Raw XORWOW is 67.1 Gsamples/s; seven sequential uniforms are still tens of Gsamples/s of RNG if that were the only work.
- Two alias records (interleaved `float prob + int alias`, 8 B) plus trapezoid geometry (`lo, width, y0, y1`, 16 B) per axis. Radial K=4095, θ K=2047 → ~147 KB of tables. That is **larger than Turing L1 (64 KB)** and **tiny versus L2 (4 MB)**. Charge them as L2 hits, not against `bw_copy`. K4’s 24 KB quantile tables were the L1 case; this is not a reason to predict a DRAM step-up.
- Linear-within-bin: one `sqrtf` per axis (inverse of a trapezoid CDF). Uniform-within-bin skips the sqrts and leaves a histogram plateau. Keep linear after χ²-compare.
- Cartesian map + `|ψ|²` weight (same Laguerre / Legendre / `expf` / `sincosf` as K4).

### Kernel 5 — alias draw

- DRAM that actually leaves the SM: XORWOW 96 B + float4-worth of SoA stores 16 B = **112 B**. Same sketch as K4. 305 GB/s / 112 B = **2.72 Gsamples/s**.
- Table traffic is data-dependent and uncoalesced (different bins per lane), same *kind* of scatter as K4’s two lerps, plus more bytes per lane (24 B × 2 axes vs ~16 B of quantile loads). L2 should absorb it. If it does not, DRAM SOL goes up and we slow down — that would be a miss, not “alias is O(1).”
- Compute: two `sqrtf` extra versus K4, plus four extra XORWOW rounds, plus two biased compares. Turing 40 SM × 4 SFU × 1.5 GHz ≈ 240×10⁹ SFU-op/s. Five SFU-ish ops/sample → ~48 Gsamples/s if the pipe were the roof. FP32 FMA roof at 80 ops: 3.75e12/80 ≈ 47 Gsamples/s. **Math is not the roof** unless ncu says the pipes are busy. K4 used 5% of FP32 peak.
- Therefore the binding constraint should be **the same XORWOW 48 B LG throttle**, with a small tax for extra uncoalesced table loads and two sqrts. Alias does not beat 2.72 by being O(1).

### Kernel 6 — layout and fusion

- SoA four scalar stores vs one `float4 {x,y,z,w}`: both 16 B/sample, perfectly coalesced either way if adjacent threads write adjacent samples. Predicted difference is **noise** (band ±3%). Report the measurement; do not repeat “AoS is slow.”
- Split pipeline (sample kernel writes 12 B spherical, weight kernel reads 12 B and writes 16 B xyzw): extra **12 B/sample** round trip on top of 112 B. 305 GB/s / 124 B ≈ **2.46 Gsamples/s** if the extra traffic is DRAM. Fusion’s lesson is that number, not a sermon.
- Fused alias+weight+float4 still loads 48 B XORWOW. Layout cannot delete that.

### The node hole (correctness, the actual point)

K4 lerps a tabulated quantile across a flat CDF and manufactures samples inside a radial node. Alias masses are CDF differences: a zero-mass bin has p=0, Vose gives it `prob=0` and an alias, and the biased coin never keeps it. **The node bug is structurally impossible.** (3,1) node-window χ² at 1e7 must **pass**, not print 11572. Uniform-within-bin still plateaus inside live bins; linear-within-bin is the density we actually built (trapezoid, same as the CPU CDF). Moments vs exact to 5σ; vs FP64 CPU reference means to <0.1%. Bitwise A/B vs K2/K4 is impossible — different algorithm.

## Predicted number
- K5 alias, linear interior, SoA stores: **2.6 Gsamples/s** (band 2.2–2.8). About 0.95× K4. Correctness, not speed.
- K5 alias, uniform interior: same rate (one fewer sqrt). Worse χ². Keep linear.
- K6 float4 fused: **2.6 Gsamples/s** (band 2.2–2.8). Same roof.
- K6 split sample+weight: **2.4 Gsamples/s** (band 2.0–2.6). The 12 B round trip.

## What would falsify this
- K5 `dram__throughput` ≪ 80% of peak **and** ≪ 2.2 Gsamples/s → extra sqrts / extra uniforms became the roof; the XORWOW story is wrong.
- K5 `smsp__pipe_fma` or SFU pipe >40% of peak → weight math took over (K4’s 5% line).
- K5 faster than 3.0 Gsamples/s on the same 112 B XORWOW+out → K4 was not on the DRAM roof, or I mis-counted bytes.
- (3,1) node-window χ² still failing at 1e7 with alias-linear → Vose did not zero the hole (builder bug, not “interpolation”).
- Split **not** slower than fused by more than noise → the 12 B round trip is L2, not DRAM, and fusion is a negative result worth stating.
