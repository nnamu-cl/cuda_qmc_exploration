# Reading — what Nsight actually said

Quote these files. Official throughput: `../shared_cdf.json` (1.37 Gsamples/s), `../inverse_table.json` (2.72 Gsamples/s), SM 1500 MHz. ncu replay ~1.17 GHz — stalls and ratios only.

## The picture in one table

| | K3 `SampleSharedCdfKernel` | K4 `SampleInverseKernel<false>` | K2 FP32 (naive, for contrast) |
|---|---|---|---|
| Compute (SM) | 21.6% | 10.8% | 30.9% |
| DRAM | 36.8% | **85.7%** | 52.3% |
| ncu memory GB/s | 130 | **300** | — |
| ncu’s call | latency + occupancy | **DRAM** | latency (LG + scoreboard) |
| Top stall | L1TEX scoreboard 39% of issue gap | **LG throttle 73%** of issue gap | LG 54%, scoreboard 31% |
| Occupancy | **47.7%** (theory 50%, shared-limited) | 89% (theory 100%) | 95% |
| Shared / block | 24.6 KB → 2 blocks/SM | 0 | 0 |
| Bank conflicts | **7.1-way**, 86% of shared-load wavefronts | n/a | n/a |
| Uncoalesced extra sectors | 65% global + 78% shared wavefronts | 77% global | 78% |
| Branch efficiency | **77%** | **100%** | 100% |
| FP32 peak used | **2%** | **5%** | 3% |
| Block limit | shared mem (2) | warps (4) / regs (5) | — |

ncu SM 1.17 GHz. JSON: K3 **1.37 Gsamples/s**, K4 **2.72 Gsamples/s**, K2 **1.78 Gsamples/s**.

## K3 — shared CDF

SOL: *“low compute and memory … typically latency.”* Occupancy section: *“theoretical occupancy (50.0%) is limited by the required amount of shared memory.”* That is the hypothesis falsifier (slower than 1.5 Gsamples/s with occupancy <40% — we landed at 47.7% / 1.37).

Bank conflicts are real, not vibes: *“7.1-way bank conflict … 86.01% of the overall … wavefronts for shared loads.”* Est. speedup 52% if those went away — still would not beat K2’s 1.78 if occupancy stays at 2 blocks.

Branch efficiency **77%** vs K2’s **100%**. Moving the walk into shared did not keep the predicated `lower_bound`; some of the search now actually diverges. That was not in the hypothesis.

XORWOW global traffic remains (65% extra sectors). Shared did not touch the 48 B state.

512 threads (occupancy sweep) reaches 1.63 Gsamples/s — still under K2. Caching nodes too (48 KB, 1 block/SM) is worse at 256 threads (0.96).

## K4 — inverse table

SOL: *“utilizing greater than 80% … Start by analyzing DRAM.”* Memory throughput **300 GB/s** vs measured `bw_copy` **305 GB/s**. DRAM 86%. This is the 112 B XORWOW+out roof, not SFU.

FP32 peak **5%**. Falsify line was >40%. Weight `expf`/`sincosf` did not become the bottleneck.

Top stall: LG queue full, 47.3 of 65.2 cycles between issues (**73%**). Same XORWOW 48 B load/store as K2, now with nothing else to hide behind. Uncoalesced 77% is that state plus the two scattered lerps.

Branch 100%, occupancy 89%. The 23-step walk is gone (7.5e6 instructions vs K3’s 36e6 at the same 1e6 samples).

## nsys timeline (1e6 samples)

`nsys/shared_cdf.txt` / `inverse_table.txt`:

- Shared: SetupXorwow ×3 **78%** of GPU kernel time vs sample ×23 **22%**
- Inverse: setup **86%** vs sample **14%**

K4 made the sample bar shorter; setup did not move. Philox (Part 5) is that bar.

## What this changes

Do not chase more shared memory. K3 lost to occupancy + banks + a walk that was already L2. K4 hit the copy roof on XORWOW+out. Next algorithm that still uses tutorial XORWOW cannot beat ~2.7 Gsamples/s on this card. Alias-method is the correctness fix for the node trap, not a new bandwidth regime until Part 5 deletes the 48 B state.
