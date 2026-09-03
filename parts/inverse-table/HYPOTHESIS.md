# Hypothesis — shared-memory CDF and inverse-table sampler

## Claim
Kernel 3 (same FP32 inverse-CDF walk as naive K2, CDF tables copied into per-block shared memory) will reach about **1.9 Gsamples/s**, a small bump over measured K2, still limited by XORWOW 48 B state traffic (LG throttle) and by bank-conflicted shared loads — not by “fixing warp divergence.” Kernel 4 (tabulated quantile, two adjacent lerps, no search) will reach about **2.7 Gsamples/s**, limited by the same XORWOW load/store plus 16 B output, approaching the 112 B DRAM sketch. Weight-function `expf`/`sincosf` become *eligible* as a bottleneck for the first time and still should not win.

## Arithmetic
Referees are **measured** Release JSON and **ncu stall names**, same RTX 2070 Super Max-Q, SM 1500 MHz. ncu replay sat at ~1.17 GHz — stalls and ratios only, never mixed with JSON Gsamples/s. The 344 B DRAM model and the “divergent binary search” story are both dead (naive RETRO / `parts/naive-cuda/results/nsight/reading.md`).

| Referee | File | Number |
|---|---|---|
| naive FP32 (K2) | `parts/naive-cuda/results/naive_fp32.json` | **1.78 Gsamples/s** |
| naive FP64 (K1) | `parts/naive-cuda/results/naive_fp64.json` | 560 Msamples/s |
| `bw_copy` | `parts/cpu-baseline/results/bw_copy.json` | **305 GB/s** |
| `fma32` | cpu-baseline results | 3.75 TFLOPS |
| cuRAND XORWOW | `parts/cpu-baseline/results/curand_xorwow.json` | 67.1 Gsamples/s |

ncu on K2 (`sample_fp32.txt`), not vibes:

| Fact | Number |
|---|---|
| FP32 peak used | **3%** |
| DRAM / compute SOL | 52% / 31% |
| Top stall | **LG throttle 54%**, then scoreboard 31% |
| Uncoalesced extra sectors | **78%** (4.9 B of 32) |
| L2 hit | **84%** |
| Branch efficiency | **100%** (walk is predicated) |
| Occupancy | 95% (49 regs) |
| ncu’s uncoalesced-fix hint | **56–75%** speedup |

XORWOW setup stays a separate kernel (182 ms / 8e6). Not this part.

Work per sample after tables exist, FP32, tutorial XORWOW (48 B load + 48 B store):

- Three uniforms, φ = 2πu, Cartesian map, weight `|ψ|²` (Laguerre / Legendre, one `expf`, two `sincosf`).
- K3: the same 23-step `lower_bound` on the CDF, but the CDF lives in shared (4096 + 2048 floats = 24 KB). Node arrays stay in global unless an occupancy experiment copies them too (48 KB).
- K4: `idx = u*(K−1)`, lerp `table[idx]` and `table[idx+1]` on r and on θ. Four scalar loads, no search.

### Kernel 3 — shared CDF

- Global bytes that actually leave the SM: XORWOW 96 B + float4 out 16 B + four node loads ~16 B ≈ **128 B**. The 23 CDF loads are shared, not DRAM. Charging them against `bw_copy` was the naive-part mistake; I will not repeat it.
- Bandwidth bound if those 128 B were DRAM: 305 GB/s / 128 B = **2.38 Gsamples/s**. K2 already showed tables are not DRAM (L2 hit 84%). Shared memory does not “pull a 344 B kernel out of DRAM.”
- Latency: the walk is still ~23 dependent steps. Shared hit is ~20–30 cycles; K2’s scoreboard was 31% and L1TEX-bound. Turing L1/shared are the same physical SRAM. The honest question is **bank conflicts on data-dependent 32-bit indices**, not branch divergence (ncu: 100% branch efficiency).
- Occupancy: 24 KB dynamic shared, 256 threads, 64 KB Turing shared → 2 blocks/SM → 16 warps vs K2’s 95%. Shared can *hurt* occupancy while helping the walk.
- XORWOW LG throttle (54% of K2 issue gaps) does not move. K3 cannot beat a ceiling set by 48 B state even if the walk becomes free.
- Therefore the binding constraint should be **XORWOW LG traffic, with bank-conflicted shared loads as the leftover walk cost**. Predicted gain over 1.78 Gsamples/s is small.

### Kernel 4 — inverse table (Devroye ch. II / III, tabulated quantile)

- Per sample: 2 lerps (≤ 16 B, adjacent so one 8 B pair per axis; still uncoalesced across the warp because u differs), 96 B XORWOW, 16 B out. Tables (K=4096 r + 2048 θ floats ≈ 24 KB) fit in L1/L2 for the whole grid.
- DRAM sketch, XORWOW+out only: 305 GB/s / 112 B = **2.72 Gsamples/s**. That is the number K2 was already 65% of.
- ncu’s 56–75% uncoalesced-fix on 1.78 Gsamples/s → **2.78–3.12 Gsamples/s**. Same neighborhood.
- Compute: ~3 SFU ops (`expf` + 2×`sincosf`) plus a short recurrence. Turing 40 SM × 4 SFU × 1.5 GHz ≈ 240×10⁹ SFU-op/s → ~80 Gsamples/s if each op were one cycle (they are not; still tens of Gsamples/s). FP32 FMA roof at 50 ops: 3.75e12/50 = 75 Gsamples/s. **Math is not the roof** unless the profiler says the pipes are busy.
- Therefore the binding constraint should be **XORWOW 48 B state + the last uncoalesced lerp**, not SFU. This is the first kernel where a busy FP32/SFU pipe would surprise me.

### The node trap (correctness, not speed)

Radial densities with n−l−1 ≥ 1 have nodes. The tabulated CDF is nearly flat there, so the quantile jumps. Lerping a uniform-u table across the jump manufactures samples inside the hole. χ² on fine radial bins around the node of (3,1) should **fail** for uniform K4 and shrink ~1/K without vanishing. Node-aware snap (do not interpolate a jump) is the fiddly mitigation. The clean algorithm is Part 4. Do not start it here.

## Predicted number
- K3 shared CDF: **1.9 Gsamples/s** (band 1.5–2.4). About 1.1× K2. Occupancy drop or bank conflicts can cancel the latency win entirely.
- K4 inverse table: **2.7 Gsamples/s** (band 2.2–3.3). About 1.5× K2, sitting on the 112 B XORWOW+out sketch. Not 10×. Not cuRAND.
- K4 node-aware: same rate as uniform K4 (one extra compare), different χ².

## What would falsify this
- K3 `l1tex__data_bank_conflicts` ≈ 0 **and** >2.5 Gsamples/s → shared was a free latency win; I overstated banks and occupancy.
- K3 slower than 1.5 Gsamples/s with occupancy <40% → shared-memory-per-block was the real knob, not the walk.
- K4 `smsp__pipe_fma` or SFU pipe >40% of peak → weight math became the roof and the XORWOW story is wrong.
- K4 still long-scoreboard-dominated on four table loads → tables did not sit in L1; the 23-load diagnosis was incomplete.
- K4 χ² on (3,1,*) passing at K=4096 with uniform lerp → the node trap is weaker than the piecewise-linear CDF led me to expect.
