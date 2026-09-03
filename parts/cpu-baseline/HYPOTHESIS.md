# Hypothesis — CPU hydrogen sampler

## Claim
A corrected FP64 inverse-CDF sampler (tabulated R and θ, uniform φ, interpolated bins) will run at about 4 million samples/s on one i7-10875H thread, limited by dependent binary searches plus libm, and about 22 million samples/s with OpenMP over the 8 physical cores.

## Arithmetic
CPU: Intel Core i7-10875H, 8 cores / 16 threads, 2.3 GHz base, 5.1 GHz single-core turbo. One busy core is ~4.8 GHz; all-core turbo is closer to ~4.0 GHz. No GPU referee numbers yet — those wait for Release `bw_copy` / `fma_loop` / `curand_raw` under the clock policy.

Work per sample after the tables are built:

- One Philox4x32-10 call (three uniforms from one counter).
- Two binary searches: 4096 radial bins + 2048 θ bins → 12 + 11 = 23 dependent loads. Both tables fit in L1 (32 KiB + 16 KiB of doubles).
- Two linear interpolations inside the hit bin (piecewise-linear inverse CDF).
- Cartesian conversion: `sin`, `cos` (or `sincos`) on θ and φ.
- Weight `w = |ψ|²`: associated Laguerre (`k = n−l−1 ≤ 5` here), associated Legendre, `exp`, `pow`.

Cycle guess: ~40 (Philox) + ~23 × 5 L1 (search) + ~150–300 (libm) + ~50 (polynomials) ≈ 400–700 cycles/sample.

- 4.8e9 cycles/s ÷ 600 cycles/sample ≈ 8 Msamples/s if libm is cheap.
- 4.8e9 ÷ 1200 ≈ 4 Msamples/s if `exp`/`pow`/`sin` dominate.

The second number is the one I believe. Binding constraint: serial libm + the 23-step search, not DRAM (tables are L1-resident; the write is 32 B/sample ≈ 128 MB/s at 4 Msamples/s).

OpenMP: 8 cores × (4.0/4.8 turbo drop) × 4 Msamples/s ≈ 27 Msamples/s, then knock ~20% for OpenMP and write contention → 22 Msamples/s. Hyper-threads should add little.

## Predicted number
- Single thread: **4.0 Msamples/s** (factor-of-two band: 2–8).
- OpenMP, 8 cores / 16 threads: **22 Msamples/s** (band: 12–40).

GPU Gsamples/s claims are out of scope until the Release microbenches exist.

## What would falsify this
- `perf stat` cycles/sample far from 400–1200 (wrong bottleneck).
- OpenMP scaling under 4× on 8 cores (allocator / false sharing, not arithmetic).
- Single-thread over 10 Msamples/s (weight eval is cheaper than the libm guess).
