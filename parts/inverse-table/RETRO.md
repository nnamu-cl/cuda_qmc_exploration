# Retro — shared-memory CDF and inverse-table sampler

Predicted: 1.9 Gsamples/s K3, 2.7 Gsamples/s K4
Measured: **1.37 Gsamples/s** shared CDF, **2.72 Gsamples/s** inverse table, **2.66 Gsamples/s** node-aware
Ratio: 0.72×, 1.01×, 0.99×

Release, RTX 2070 Super Max-Q, SM 1500 MHz, 8×10⁶ samples of (1,0,0), median of 20 after 3 warmups. XORWOW setup excluded (163 ms / 8e6). JSON in `results/`.

## Where the model was right
- **K4 sits on the 112 B XORWOW+out sketch.** 305 GB/s / 112 B = 2.72 Gsamples/s; measured 2.72. ncu: DRAM **85.7%**, 300 GB/s, FP32 peak **5%**. Math did not take over.
- Shared memory was not a DRAM story. K3 DRAM 37%. Tables were already L2 (K2 hit 84%).
- Bank conflicts exist: ncu 7.1-way, 86% of shared-load wavefronts.
- Occupancy vs 24 KB shared is the knob the occupancy sweep measured (`experiments/occupancy.json`): 128/256/512 threads → 0.73 / 1.37 / 1.63 Gsamples/s, always 2 blocks/SM.
- The node trap fires. Uniform K=4096, (3,1), 10⁷ samples: node-window χ² **11572**. Snap helps (6531) and does not close it. K-sweep: node χ² shrinks ~1/K (53054 at 256 → 85 at 65536) and never cleanly vanishes.

## Where the model was wrong
K3 predicted 1.9 (band 1.5–2.4). Measured **1.37**, below the 1.5 falsification line.

- **Shared occupancy tax > walk-latency win.** Theoretical occupancy 50%, achieved 47.7%, limited by shared memory (ncu). K2 was 95%. The 23-step walk was already L2; paying 2 blocks/SM to put it in SRAM made the kernel *slower than naive FP32* (1.78).
- **Branch efficiency dropped to 77%** (K2 was 100%). The shared walk is not the same predicated `lower_bound`.
- 512 threads recovers to 1.63 Gsamples/s — still under K2. Caching nodes (48 KB, 1 block) is worse at the default 256 threads (0.96).
- Node-aware snap at huge K can *raise* node-window χ² (166 vs 85 at K=65536). Fiddly, as advertised.

K4’s 1.01× hit is the one-paragraph correct prediction. K3 is the section.

## What this changes going into the next part
K3 is a negative result: memory-hierarchy tricks on the *same algorithm* lost to occupancy. K4 killed the walk and hit DRAM on XORWOW+out; SFU is still 5%. The node trap is the correctness cliffhanger — alias method (Part 4), not a bigger table. Philox (Part 5) is the only way past 2.7 Gsamples/s on this card. Do not start either until this part is reported.
