# Hypothesis — stateless Philox alias sampler

## Claim
Kernel 7 (same Walker/Vose bins as K5, XORWOW replaced by **stateless Philox4x32-10**) will reach **6.0 Gsamples/s**, limited by the two uncoalesced 24 B alias-record loads that already knocked K5 off the DRAM roof — **not** by “counters being free.” Setup time goes to **0**. Kernel 8 (grid-stride, S independent samples per thread) will reach **8.5 Gsamples/s** by hiding L2/Philox latency, still about **2×** short of the 16 B `bw_copy` roof. Alias stays the sampling algorithm.

## Arithmetic
Referees are **measured** Release JSON and **ncu stall names**, same RTX 2070 Super Max-Q, SM 1500 MHz. ncu replay sat at ~1.15 GHz — stalls and ratios only, never mixed with JSON Gsamples/s.

| Referee | File | Number |
|---|---|---|
| alias linear (K5) | `parts/alias-method/results/alias_linear.json` | **1.76 Gsamples/s** |
| alias uniform | `parts/alias-method/results/alias_uniform.json` | 1.95 Gsamples/s |
| inverse table (K4) | `parts/inverse-table/results/inverse_table.json` | **2.72 Gsamples/s** |
| naive FP32 (K2) | `parts/naive-cuda/results/naive_fp32.json` | 1.78 Gsamples/s |
| CPU OpenMP | `parts/cpu-baseline/results/cpu_openmp.json` | 15.1 Msamples/s |
| `bw_copy` | `parts/cpu-baseline/results/bw_copy.json` | **305 GB/s** |
| `fma32` | `parts/cpu-baseline/results/fma32.json` | **3.75 TFLOPS** |
| cuRAND Philox host API | `parts/cpu-baseline/results/curand_philox.json` | **66.0 Gsamples/s**, 264 GB/s |
| cuRAND XORWOW host API | `parts/cpu-baseline/results/curand_xorwow.json` | 67.1 Gsamples/s |

ncu / nsys on K5 (`parts/alias-method/results/nsight/reading.md`), not vibes:

| Fact | Number |
|---|---|
| K5 JSON | **1.76 Gsamples/s** |
| K5 DRAM / ncu mem GB/s | **54%** / **191** |
| K5 L2 / L1 hit | **91.9%** / **61.5%** |
| K5 top stall | LG throttle **83%** (61 of 73.8 cycles) |
| K5 uncoalesced extra sectors | **80%** |
| K5 FP32 peak | **5%** |
| K5 inst at 1e6 | **11.0e6** (K4 was 7.5e6) |
| nsys setup vs sample at 1e6 | **83%** / **17%** |
| XORWOW setup | **163 ms / 8e6** |

16 B/sample DRAM roof: 305 GB/s / 16 B = **19.06 Gsamples/s**.
cuRAND raw writes 4 B/uniform: 66.0 Gsamples/s is 264/305 = **86%** of `bw_copy`. Generating bits into DRAM is already nearly a copy.

### The 48 B going away

XORWOW state is **48 B/thread**. K5 loads it and stores it back every sample: **96 B** of LG traffic, plus **16 B** of float4-worth of stores = **112 B**. At 1.76 Gsamples/s that is 96×1.76 = **169 GB/s** of state plus 16×1.76 = **28 GB/s** of output = **197 GB/s**, next to ncu’s 191. That is the LG throttle. Philox used **statelessly** (Salmon, Moraes, Dror & Shaw, SC’11) has counter = `(sample_index, draw_slot)` and key = seed. **Zero** `curand_init`. **Zero** 48 B load/store.

It is not free arithmetic. One Philox4x32-10 is ten rounds of 32-bit mul-hi/lo + xor + Weyl key bump. Alias still needs **7 uniforms** (fair die + biased coin + interior × two axes, plus φ). One call yields 4×32-bit, so **2 calls/sample**, 1 wasted lane. Host-API Philox writes 4 B at 66.0 Gsamples/s; two calls consume 8 uniforms, so a *store-the-bits* roof on the RNG alone is 66.0/8 = **8.25 G physics-samples/s**. We do **not** store the uniforms. That 8.25 is a Philox-throughput ceiling if the 4 B write were the only extra, not a promise that in-kernel rounds are cheaper than a copy.

### What remains after the state dies

- Coalesced output: **16 B/sample** (keep packed xyzw; K6 already measured SoA vs float4 as noise).
- Two data-dependent `AliasBin` records, 24 B each = **48 B** uncoalesced. Radial K=4095, θ K=2047 → ~147 KB, larger than Turing L1 (64 KB), tiny vs L2 (4 MB). K5 already measured L2 hit **91.9%**, L1 hit **61.5%**. Charge the misses: 0.081×48 ≈ **3.9 B** DRAM, plus 16 B out ≈ **20 B**. If this bound: 305/20 = **15.3 Gsamples/s**.
- Same two `sqrtf` + `sincosf` + Laguerre/Legendre weight as K5. K5 used **5%** of FP32 peak. Math is not the roof unless ncu moves that number.
- Philox integer: 2×10 rounds. Turing IMAD is not scarce. Do not call this free, do not call it the roof without a pipe metric.

K5 was **not** on the 112 B DRAM roof (DRAM SOL 54%, vs K4’s 86%). Deleting 96 B of XORWOW removes the traffic that *was* filling LG, and leaves the traffic that already made alias slower than inverse-table: scattered 24 B records. The 15.3 Gsamples/s DRAM sketch is an upper bound **if** L2 scatter stops hiding. It will not: K5’s extra sectors were 80% with XORWOW **and** the two records. Dropping 48 B sequential-ish state should drop extra sectors, not delete the data-dependent pair.

### Kernel 7 — one sample per thread, block 256

- Binding constraint: **uncoalesced L2 alias loads + two Philox calls of latency**, not DRAM, not SFU.
- Central number sits between K5 (1.76, XORWOW-dominated) and the 8.25 Philox-consume sketch and well below 15.3/19.06.
- Setup: **0 ms**. nsys sample bar becomes the whole kernel timeline.

### Kernel 8 — grid-stride (Harris) and ILP (Volkov)

- Grid-stride loop, S samples per thread with **independent** counters (sample index, not thread id). Sweep S ∈ {1,2,4,8} × blocks {128,256,512}. Occupancy will drop as S grows (more registers for in-flight Philox). Volkov’s point is that independent chains hide the remaining L2/Philox latency *at lower occupancy*.
- Do not amortize one Philox output across two sample indices — that couples streams and breaks the “counter = sample index” CI check.
- Last-squeeze candidates, each measured, kept only if they pay: `__ldg` / `const __restrict__` on tables; `--use_fast_math` on this kernel only with Part 2’s FP32 error table re-run; L2 persistence window on 147 KB (`cudaAccessPropertyPersisting`). Persistence on KB tables should be **noise**.

### Idiomatic competitor (not an opponent)

Thrust + cuRAND host API: generate 7 uniform buffers (28 B/sample extra DRAM), `thrust::for_each` over the same Vose bins, write 16 B. Bandwidth sketch 305/(28+16) = **6.9 Gsamples/s** if it were one streaming kernel. It is several launches. Predict **1.2 Gsamples/s**. It must pass the same suite or it is not a competitor.

## Predicted number
- K7 philox, alias-linear, packed xyzw, 256 threads, S=1: **6.0 Gsamples/s** (band 4.0–9.0). About 3.4× K5. Setup 0.
- K8 winning cell of the S×block sweep: **8.5 Gsamples/s** (band 6.0–12.0). About 2.2× short of 19.06. Enters the plan’s “1.2–2× of the 16 B roof” band only if the top of the error bar is real.
- Thrust + cuRAND pipeline: **1.2 Gsamples/s** (band 0.5–2.5).
- Fast-math / L2 persist: **noise** (±5%) unless ncu says SFU or L2 changed.

## What would falsify this
- K7 still ~1.76 Gsamples/s **and** LG throttle still ~80% → the 48 B state was not the tax; I mis-counted the bytes going away.
- K7 `dram__throughput` >80% of peak at ≥12 Gsamples/s → tables *were* free once XORWOW died, and the 6.0 central number is low.
- K7 `smsp__pipe_fma` or integer/SFU pipe >40% of peak → Philox rounds or the two sqrts became the roof (K5’s 5% line).
- K7 still shows a 163 ms setup kernel in nsys → we left `curand_init` in.
- Different block sizes or K7 vs K8 disagree bitwise at the same seed and sample index → the counter is a thread id.
- (3,1) node-window χ² fails → we forked the alias tables; do not. Reuse K5 Vose bins.
- Thrust pipeline fails KS/χ²/moments → it is not a competitor; fix the functor, do not quote its rate.
