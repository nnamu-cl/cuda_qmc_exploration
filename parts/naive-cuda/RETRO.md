# Retro — naive CUDA hydrogen sampler

Predicted: 120 Msamples/s FP64, 450 Msamples/s FP32, 540 Msamples/s FP32+intrinsics
Measured: **560 Msamples/s** FP64, **1.78 Gsamples/s** FP32, **1.76 Gsamples/s** FP32+`__expf`/`__sincosf`
Ratio: 4.7×, 3.9×, 3.3× (all faster than the model; intrinsics: 0.99× IEEE-32)

Release, RTX 2070 Super Max-Q, SM graphics lock 1500 MHz, 8×10⁶ samples of (1,0,0), median of 20 after 3 warmups. XORWOW setup excluded from sample/s (182 ms to init 8×10⁶ states). JSON in `results/`.

## Where the model was right
- Not the 305 GB/s copy roof. K1 is 0.56 Gsamples/s vs 19.1 Gsamples/s at 16 B (2.9%).
- FP32 is a few times FP64 (3.17×), not 32×. Turing's 1:32 FP64 rate is real in `fma64` (88.2 GFLOPS) and does not fully show up here.
- Setup is expensive: 182 ms vs 14.3 ms K1 kernel (~13×). Tutorial XORWOW `curand_init(seed, tid, 0)` is content for Part 5, not a rounding error.
- Far from cuRAND (67.1 Gsamples/s). The naive port is a 37× OpenMP win and still 2.6% of the RNG roof.

## Where the model was wrong
I priced ~500 FP64 ops/sample and 344 B of DRAM. I got 560 Msamples/s, which is above the 400 Msamples/s falsification line.

- **Tables are not DRAM.** 4096+2048 doubles is 96 KiB. Turing L2 on this card is 4 MiB. The 23-step walk is L2, not the copy kernel. Counting 23 × 8 B against 305 GB/s was the wrong roof.
- **DP libm is cheaper than 500 FLOPs.** 88.2e9 / 560e6 ≈ 158 FP64-equivalent ops/sample if the pipe were full — and it probably is not. Two `sincos` + `exp` plus a tiny polynomial do not fill 500 ops.
- **K2 beat the 220 B DRAM bound** (1.78 Gsamples/s vs 1.39). Same story: state + output are the DRAM terms (~112 B → 2.7 Gsamples/s roof); 1.78 is 65% of that. Search-latency-in-DRAM was the wrong bottleneck.
- **Intrinsics did nothing.** `__expf`/`__sincosf` 1.76 vs IEEE-32 1.78 Gsamples/s. The leftover was not `expf`.

Nsight reports: `results/nsight/reading.md` (quote sheet). ncu SM was 1.17 GHz — stalls only, not Gsamples/s.

- K1 SOL: compute **84.8%**, DRAM 19%. **FP64 pipe 84.8%** (53% of FP64 peak). Long scoreboard **65.5%** of issue gaps (42.9 cy L1TEX). L2 hit **84%**. Branch efficiency **100%** — the walk is predicated, not warp-divergent. Uncoalesced **68%** extra sectors (8 B of 32).
- K2 SOL: compute 31%, DRAM 52%, **3% of FP32 peak**. Latency: LG throttle 54%, then scoreboard 31%. Uncoalesced **78%**. That is why `__expf` was a no-op.
- Setup: 22 ms / 1e6, DRAM **3%**, LSU/mem-pipes **90%**. nsys: setup 60% vs sample 40% of GPU kernel time at this mix.

The 344 B DRAM model is dead. Next: inverse-table (uncoalesced walk), then Philox (setup + 48 B state). Not fast-math.

## FP32 numerics
Host sweep (`experiments/fp32_vs_fp64.json`): mean rel. error R_nl **5.0×10⁻⁷**, P_l^m **3.0×10⁻⁷**. Max on the grid is 6.4×10⁻⁴ at (n=6,l=0) tail points and 4.5×10⁻⁴ at P_4^0 — those are zeros/underflows, not the sampled bulk. On 10⁷ device samples with |w|>10⁻⁸, max rel. error vs FP64 weight is **7.7×10⁻⁵**. 1/√N at N=10⁹ is ~3×10⁻⁵. Mean error is below that floor; the sampled max sits next to it. Iterated ρ^l (2.5×10⁻¹⁶) beats `exp(l log ρ)` (3.5×10⁻¹⁵). Gate is mixed 10⁻⁸ abs + 10⁻⁵ rel so nodes do not fire the alarm.

## What this changes going into the next part
Drop the 344 B DRAM model. ncu: K1 is the FP64 pipe + uncoalesced L1TEX, not DRAM; the search does not diverge. K2 is LG-queue latency from those loads and XORWOW state. Inverse-table kills the 23 loads. Philox kills setup (90% LSU skipahead, 3% DRAM) and the 48 B traffic. `__expf` is closed.
