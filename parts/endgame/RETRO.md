# Retro — stateless Philox alias sampler

Predicted: 6.0 Gsamples/s K7, 8.5 K8, 1.2 Thrust
Measured: **4.60 Gsamples/s** philox, **4.59** philox_ilp (128, S=1), **2.21** thrust_alias
Ratio: 0.77×, 0.54×, 1.84×

Release, RTX 2070 Super Max-Q, SM 1500 MHz, 8×10⁶ samples of (1,0,0), median of 20 after 3 warmups. Setup **0 ms**. JSON in `results/`.

## Where the model was right
- **The 48 B went away.** nsys: 100% `SamplePhiloxKernel`. K5 was 83% `SetupXorwow`. 163 ms / 8e6 is gone.
- **DRAM SOL collapsed.** K5 54% / 191 GB/s → K7 **16%** / 57 GB/s. The 96 B XORWOW term was real DRAM. ncu L2 hit **99.9%** (K5 91.9%).
- **Math is still not the roof.** FP32 peak **10%**, falsify line was >40%. Two Philox calls added integer (13.3e6 inst vs K5’s 11.0e6) and did not take the FMA pipe.
- **Tables still scatter.** Uncoalesced extra sectors **83%**, top stall still LG throttle **71%**. The hypothesis named this leftover. It was not free.
- **Launch configs agree bitwise.** 128 vs 256 vs K8 S=2/4, same seed, same sample index. Counter is not a thread id.
- **(3,1) node-window χ² 29.7** at 1e7. Alias tables were reused, not forked.
- **Thrust passes the suite** at 1e7. It is a competitor. 2.21 Gsamples/s sits in the 0.5–2.5 band.
- **Persist window unsupported** (`invalid argument`). Predicted noise; got a no-op.

## Where the model was wrong
K7 predicted 6.0 (band 4.0–9.0). Measured **4.60**, inside the band, 0.77× the central number.

- **LG throttle did not empty.** I counted 96 B of XORWOW leaving and 48 B of tables remaining, then guessed 6.0. ncu still spends 20.4 of 28.6 issue cycles in the LG queue. The two 24 B records are the whole remaining memory story, and they are enough to keep the kernel latency-bound (ncu: both compute and DRAM <60%).
- **K8 did not hide that latency.** Predicted 8.5. Sweep winner is S=1, 128 threads, **4.58**, identical to K7. S=2,4,8 all slower. Occupancy dropped (94% → 69% on the profiled S=1 grid-stride cell). Volkov needs independent work that the extra registers can cover; here every sample still does the same two uncoalesced loads.
- **16 B roof is 4.1× away, not 1.2–2×.** 305/16 = 19.06. 4.60 is not in the same league as cuRAND’s 264 GB/s. The Drop 1 headline is not earned.
- Thrust was **faster** than 1.2 (2.21). The 7-uniform generate is one Philox host-API kernel, not seven, and cub `for_each` is 63% of its GPU time. Still 2.1× behind K7, not “several ×.”

K8’s 0.54× miss is the section. K7’s 0.77× is one paragraph: right bottleneck, optimistic constant.

## What this changes going into the next part
Do not open Drop 2 from a speed claim. Direct sampling of a factorized hydrogen density is done: alias is correct, Philox deleted setup, the leftover is uncoalesced 24 B records, and a second electron kills every trick here. Stop.
