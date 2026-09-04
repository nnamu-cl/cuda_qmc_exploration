# Reading — what Nsight actually said

Quote these files. Official throughput: `../alias_linear.json` (1.76 Gsamples/s), `../alias_uniform.json` (1.95), `../alias_float4.json` (1.76), `../alias_split.json` (1.51), SM 1500 MHz. ncu replay ~1.15 GHz — stalls and ratios only.

## The picture in one table

| | K5 linear SoA | K6 float4 | K6 split sample | K6 split weight | K4 inverse (contrast) |
|---|---|---|---|---|---|
| Compute (SM) | 10.1% | 9.7% | 6.3% | 31.1% | 10.8% |
| DRAM | **54.4%** | 53.3% | 49.1% | **85.1%** | **85.7%** |
| ncu memory GB/s | 191 | — | — | — | **300** |
| ncu’s call | L2 + uncoalesced | L2 + uncoalesced | L2 | **DRAM** | **DRAM** |
| Top stall | **LG throttle 83%** | LG 81% | LG 73% | scoreboard 73% | LG 73% |
| Occupancy | 94.9% (theory 100%) | 94.3% | 93.4% | 92.7% | 89% |
| Uncoalesced extra sectors | **80%** | 80% | 81% | — | 77% |
| L1 / L2 hit | 61.5% / 91.9% | — | — | — | (K4 tables were L1) |
| Branch efficiency | **100%** | 100% | 100% | 100% | 100% |
| FP32 peak used | **5%** | 5% | 2% | 18% | **5%** |
| Inst at 1e6 samples | **11.0e6** | — | — | — | 7.5e6 |

ncu SM 1.15 GHz. JSON: linear **1.76 Gsamples/s**, uniform 1.95, float4 1.76, split 1.51. K4 was **2.72**.

## K5 — alias linear

SOL: *“Memory is more heavily utilized than Compute … identify the L2 bottleneck.”* DRAM **54%**, not K4’s 86%. The 112 B XORWOW+out sketch assumed tables were free. They are in L2 (hit 91.9%) but **not** in L1 (hit 61.5%) — 147 KB of `AliasBin` vs Turing 64 KB L1, as the hypothesis allowed. Uncoalesced extra sectors **80%**: XORWOW 48 B plus two data-dependent 24 B bin records.

Top stall: LG queue full, 61 of 73.8 cycles between issues (**83%**). Same name as K4 (73%), more of it. FP32 peak **5%** — the two `sqrtf`s did not take the pipe (falsify line was >40%). Occupancy 95%, branch 100%. The extra cost is **more uncoalesced global ops**, not math.

11.0e6 instructions vs K4’s 7.5e6 at the same 1e6 samples. 7 uniforms + two sqrts + two alias compares vs K4’s 3 uniforms + 2 lerps. 2.72 × (7.5/11.0) ≈ 1.85, next to measured 1.76.

## K6 — float4 vs SoA

float4 DRAM 53%, LG 81%, FP32 5%, occupancy 94%. JSON 1.76 vs linear 1.76. **Layout is noise.** ncu does not care that we stored `alignas(16) PackedXyzw` instead of four scalars; both are 16 B/sample next to 96 B of XORWOW.

## K6 — split

nsys (`nsys/alias_split.txt`): SetupXorwow 80%, SampleSplitCoords 17% (med 512 µs), WeightSplit 3% (med 91 µs). Fused linear sample bar is 472 µs (`nsys/alias_linear.txt`). Split pays the weight kernel on top of a *longer* sample kernel (alias without fusion did not get cheaper). Weight kernel is the one ncu calls DRAM-bound (85%, FP32 18%, scoreboard 73%) — a 12 B/16 B round trip on a tiny kernel. Fusion’s lesson is ~91 µs of 603 µs (~15%), not a new bandwidth regime.

## nsys timeline (1e6 samples)

`nsys/alias_linear.txt`: setup **83%** vs sample **17%**. K4 was 86% / 14%. Alias made the sample bar *longer*. Setup did not move. Philox (Part 5) is still that bar.

## What this changes

Alias is the correctness fix (node-window χ² 37 vs K4’s 11572). It is not a throughput win and not on the 112 B DRAM roof. Extra uncoalesced table records + extra XORWOW rounds dropped DRAM SOL from 86% to 54% and JSON from 2.72 to 1.76. Do not start a bigger table. Do not claim float4. Next delete the 48 B state.
