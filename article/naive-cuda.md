# Part 2 — The obvious CUDA port, and the FP64 tax that was not the whole bill

I ported the corrected CPU sampler, not the visualizer. One thread per sample. Host-built CDF tables in global memory. Piecewise-linear inverse CDF (Devroye ch. II). cuRAND XORWOW device state, tutorial style: `curand_init(seed, thread_id, 0)`, 48 B load/store per draw. Weight `|ψ|²` from the same Laguerre / Legendre recurrences. `--use_fast_math` off.

I predicted from measured referees, not the spec sheet: `bw_copy` **305 GB/s**, `fma64` **88.2 GFLOPS** (the plan's ~0.2 TFLOPS is a spoiler; this card measured 88), OpenMP **15.1 Msamples/s**. I counted 344 B/sample and ~500 FP64 ops, and wrote down **120 Msamples/s**, limited by software double libm plus a 23-step divergent walk.

## What I measured

Release, SM 1500 MHz, 8×10⁶ samples of 1s, median of 20:

| Kernel | Throughput | vs OpenMP |
|---|---|---|
| CPU single-thread | 2.12 Msamples/s | — |
| CPU OpenMP (16 threads) | 15.1 Msamples/s | 1× |
| K1 FP64 | **560 Msamples/s** | 37× |
| K2 FP32 | **1.78 Gsamples/s** | 118× |
| K2 + `__expf`/`__sincosf` | 1.76 Gsamples/s | 117× |
| cuRAND XORWOW uniforms | 67.1 Gsamples/s | — |
| 16 B copy roof | 19.1 Gsamples/s | — |

I predicted 120; I got 560. Factor 4.7. The 400 Msamples/s falsification line in the hypothesis is crossed: the tables are 96 KiB and live in L2 (4 MiB on this GPU), so charging 23 × 8 B against the 305 GB/s copy kernel was the wrong roof. Double `sincos`/`exp` also do not cost 500 FLOPs. 88.2e9 / 560e6 ≈ 158 ops/sample if the FP64 pipe were saturated, and it likely is not.

K2 / K1 = 3.17×, not 32×. Turing GeForce really is 1/32 FP64 in the FMA microbench. This kernel is not that microbench. The remaining DRAM is XORWOW state (96 B) plus a float4 (16 B). 305 GB/s / 112 B ≈ 2.7 Gsamples/s; K2 at 1.78 is 65% of that sketch. `__expf` did not move the needle. The leftover is the walk and the state, not IEEE-32 libm.

XORWOW setup for 8×10⁶ threads is **182 ms**. The FP64 sample kernel is 14.3 ms. I exclude setup from sample/s, as the harness does for table H2D. The tax is still there. That is Part 5's problem (Philox, Salmon et al. SC'11), not a reason to ignore the tutorial default now.

## FP32 as an error analysis

Table construction stays host FP64. Device tables are `float`. Factorials become one host-computed constant; no `tgamma` on device. ρ^l is iterated multiply, not `exp(l log ρ)` (2.5×10⁻¹⁶ vs 3.5×10⁻¹⁵ max rel. error on the sweep).

On a dense (n,l,r) grid, mean rel. error of FP32 R_nl vs FP64 is **5.0×10⁻⁷**; P_l^m **3.0×10⁻⁷**. The grid max (6×10⁻⁴ at 6s tail points, 4×10⁻⁴ at P_4^0) is zeros and underflows. On 10⁷ device samples with |w| > 10⁻⁸, max rel. error vs the FP64 weight is **7.7×10⁻⁵**. Monte Carlo 1/√N at N = 10⁹ is ~3×10⁻⁵. The mean is under that floor; the sampled max sits beside it. That is the argument for FP32, not the 3× wall-clock.

CUDA's ULP tables (Programming Guide, mathematical functions): `expf` / `sincosf` are ~2 ULP. `__expf` / `__sincosf` are the fast intrinsics. I measured them separately. They did not pay.

## Correctness

Device goldens vs the mpmath/scipy file: FP64 < 10⁻¹²; FP32 < 10⁻⁵. GPU invert of the host tables matches the CPU invert (4096 probes). KS / χ² / ⟨r⟩, ⟨r²⟩, ⟨1/r⟩ on 10⁷ samples × six orbitals, including (3,1,−1) and (5,0,0), pass the same gates as Part 1. Bitwise self-determinism holds. XORWOW ≠ Philox, so there is no bitwise A/B against the CPU sampler — the invert A/B is the algorithm check.

Nsight Compute is not installed here. The miss is already large enough to read without a stall table: I overcharged DRAM and libm. Next part should remove the binary search, not tune `__expf`.
