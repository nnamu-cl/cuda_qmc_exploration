# Hypothesis — naive CUDA hydrogen sampler

## Claim
Kernel 1 (faithful FP64 port, one thread per sample, cuRAND XORWOW device state, host-built tables in global memory) will reach about 120 million samples/s on this RTX 2070 Super Max-Q at 1500 MHz SM, limited by software double-precision libm plus 23-step divergent CDF walks — not by the 305 GB/s copy roof. Kernel 2 (same algorithm, sampling and weight in FP32, tables stored as float, construction still host FP64) will reach about 450 million samples/s, still search-latency bound.

## Arithmetic
Referees are measured Release JSON in `parts/cpu-baseline/results/`, same clock policy, not the spec sheet. The plan's "~0.2 TFLOPS FP64" is a spoiler; this card measured **88.2 GFLOPS**.

| Referee | File | Number |
|---|---|---|
| CPU single-thread | `cpu_single.json` | 2.12 Msamples/s |
| CPU OpenMP 16 threads | `cpu_openmp.json` | 15.1 Msamples/s |
| `bw_copy` (16,777,216 float4) | `bw_copy.json` | **305 GB/s** → 16 B roof **19.1 Gsamples/s** |
| `fma32` | `fma32.json` | **3.75 TFLOPS** |
| `fma64` | `fma64.json` | **88.2 GFLOPS** |
| cuRAND XORWOW | `curand_xorwow.json` | **67.1 Gsamples/s** |

Work per sample after tables exist (corrected inverse-CDF, not the visualizer):

- Load/store XORWOW state: 48 B + 48 B (tutorial default; `curand_init` is a separate setup kernel).
- Two binary searches: 4096 radial + 2048 θ bins → 12 + 11 = 23 dependent CDF loads. Interpolation adds two node loads per axis.
- Three uniforms, φ = 2πu, Cartesian map, weight `|ψ|²` (Laguerre / Legendre recurrences, one `exp`, ρ^l).

### Kernel 1 — FP64

- Bytes moved per sample: 96 (state) + 23 × 8 (CDF) + 4 × 8 (nodes) + 32 (double4 out) = **344 B**.
- Bandwidth bound: 305 GB/s / 344 B = **0.89 Gsamples/s**.
- Arithmetic FLOPs besides libm are small (~20). Double `sin`/`cos`/`exp` are software sequences on this architecture. Counting ~500 FP64 ops/sample for two `sincos` + `exp` + the polynomials: 88.2e9 / 500 = **176 Msamples/s**.
- Latency/divergence: 23 uncoalesced, data-dependent loads. Warps in a binary search take different midpoints. Occupancy will be modest (XORWOW state in registers/local plus FP64).
- Therefore the binding constraint should be **FP64 libm throughput and divergent table latency**, not DRAM. The bandwidth number is ~5× above the compute/latency guess.

### Kernel 2 — FP32

- Table build stays host FP64; device tables are float.
- Bytes: 96 (state is still 48 B) + 23 × 4 + 16 (nodes) + 16 (float4) = **220 B**.
- Bandwidth bound: 305 GB/s / 220 B = **1.39 Gsamples/s**.
- FP32 FMA roof at 500 ops: 3.75e12 / 500 = 7.5 Gsamples/s — irrelevant. Hardware `sinf`/`expf` plus the same 23-step walk.
- Therefore K2 should still be **search-latency bound**, a few times K1, far below both the copy roof and cuRAND.

### FP32 as numerics, not a flag

Monte Carlo error after N = 10⁹ i.i.d. samples is ~3×10⁻⁵. FP32 roundoff on these recurrences (n ≤ 6) should sit near 10⁻⁶–10⁻⁷, below that floor. `--use_fast_math` stays off. `sincosf` / `__expf` are a third, separately timed kernel.

## Predicted number
- K1 FP64: **120 Msamples/s** (band 50–200). About 8× OpenMP, 60× one CPU thread, 0.18% of the 16 B copy roof.
- K2 FP32: **450 Msamples/s** (band 200–800). About 4× K1.
- K2 + intrinsics: **540 Msamples/s** if IEEE-32 libm was leftover; a 1.2× bump, not a new regime.

## What would falsify this
- Nsight: `smsp__average_warps_issue_stalled_long_scoreboard` low **and** FP64 pipe busy → I overstated the search; it is a pure FP64-throughput kernel.
- K1 above 400 Msamples/s → tables hit L1, DP libm cheaper than 500 ops.
- K2 less than 2× K1 → FP64 was not the villain (then the walk, or XORWOW traffic, dominates both).
- K2 near 1.4 Gsamples/s → I was wrong about latency; it is bandwidth after all.
