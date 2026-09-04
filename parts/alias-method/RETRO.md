# Retro — Walker/Vose alias sampler

Predicted: 2.6 Gsamples/s K5 linear, 2.6 float4, 2.4 split
Measured: **1.76 Gsamples/s** linear, **1.95** uniform, **1.76** float4, **1.51** split
Ratio: 0.68×, —, 0.68×, 0.63×

Release, RTX 2070 Super Max-Q, SM 1500 MHz, 8×10⁶ samples of (1,0,0), median of 20 after 3 warmups. XORWOW setup excluded (163 ms / 8e6). JSON in `results/`.

## Where the model was right
- **Correctness, not speed.** (3,1) node-window χ² **37.1** at 1e7 (limit ~83). Inverse-table was **11572**. Vose reconstruction: 598 zero-mass bins, reconstructed hole mass 0. The node bug is structurally gone.
- **Math is not the roof.** ncu FP32 peak **5%**, same as K4. Falsify line was >40%. Two `sqrtf`s did not take the pipe.
- **float4 vs SoA is noise.** 1.758 vs 1.760 Gsamples/s. 16 B/sample either way, sitting next to 96 B of XORWOW.
- **Split is slower.** 1.51 vs 1.76. nsys: weight kernel 91 µs on top of a 512 µs sample bar (fused sample 472 µs). Direction matches; the 12 B round trip is real and small.
- **Setup did not move.** nsys 83% SetupXorwow vs 17% sample. Philox is still Part 5.
- **Branch 100%, occupancy 95%.** No K3-style occupancy tax.

## Where the model was wrong
K5 predicted 2.6 (band 2.2–2.8). Measured **1.76**, below the 2.2 falsification line.

- **Tables were not free.** Hypothesis said 147 KB sits in L2, not DRAM, so charge 112 B XORWOW+out. ncu: L2 hit **91.9%**, L1 hit **61.5%**, DRAM SOL **54%** (K4 was 86%). ncu’s call is *“identify the L2 bottleneck”* plus uncoalesced 80% extra sectors. Two data-dependent 24 B `AliasBin` records per sample are extra uncoalesced global ops on top of XORWOW. The 112 B sketch ignored them because they miss L1.
- **Extra uniforms are extra LG traffic, not free ALU.** Honest count was 7 vs K4’s 3. I called that “rounds on already-resident state.” ncu disagrees in effect: 11.0e6 instructions vs K4’s 7.5e6 at 1e6 samples; LG throttle **83%** of a *longer* issue gap (61 of 73.8 cycles). 2.72 × 7.5/11.0 ≈ 1.85, next to 1.76.
- Uniform interior (no sqrts) is **1.95**, between linear and K4 — the sqrts are a ~10% tax, not the story. The story is the extra scattered loads.
- Split’s 2.4 prediction used 124 B on a 2.6 fused roof that did not exist. Measured 1.51 / 1.76 ≈ 0.86×, same neighborhood as 2.4/2.6.

K5’s 0.68× miss is the section. float4’s 1.00× vs linear is the one-paragraph correct prediction.

## What this changes going into the next part
Alias closed the node hole. It moved us *off* K4’s DRAM roof by adding uncoalesced table records and extra XORWOW rounds. Layout/fusion are measured negative-or-small. The only lever left on this card is deleting the 48 B XORWOW state (Philox, Part 5). Do not start a bigger alias table. Do not reopen shared memory.
