# Reading — endgame (Philox, ILP, launch)

### Salmon, Moraes, Dror & Shaw (2011) — Parallel Random Numbers: As Easy as 1, 2, 3
- Link / DOI: SC’11, https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
- What it originated: counter-based PRNGs (Random123). Philox4x32-10 is a 10-round bijection of a 4×32-bit counter under a 2×32-bit key. No shared state; skip-ahead is incrementing the counter. GPU-cheap integer mul.
- What I actually need from it: use it **statelessly**. Counter = `(sample_index, draw_slot)`, key = seed. One call = 4×32-bit. Constants already in `harness/rng/philox.h` (`kPhiloxM0/M1`, Weyl `kPhiloxW0/W1`). Do not invent a second Philox. Bit-check the 4×32 output against this header; cuRAND’s counter packing may differ — that is an experiment, not a rewrite.
- Read before: this part (before HYPOTHESIS.md)

### Volkov (2010) — Better Performance at Lower Occupancy
- Link: GTC 2010, https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf
- What it originated: occupancy is a latency-hiding tool, not a goal. Independent instruction streams (ILP) hide math and memory latency at *lower* occupancy.
- What I actually need from it: Kernel 8’s S samples/thread are independent Philox counters, not a bigger block. The sweep’s winning cell is justified by occupancy + issue-stall metrics, not “more threads.”
- Read before: this part (before writing K8)

### Harris — CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops
- Link: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-with-grid-stride-loops/
- What it originated: `i += gridDim.x * blockDim.x` so a thread covers many elements and the grid need not match N. Adjacent threads still hit adjacent addresses.
- What I actually need from it: K8 launch. Do not use thread-id as the Philox counter — that would make the stride a correctness bug.
- Read before: implementing K8

### OpenRAND (Passos et al., arXiv:2310.19925) — reproducibility framing
- Link: https://arxiv.org/abs/2310.19925
- What it originated: counter-based RNGs as a professionalism marker for parallel Monte Carlo (same seed + same sample index → same bits across launch configs).
- What I actually need from it: one CI assertion, K7 vs K8 vs {128,256,512} threads, bitwise on the output buffer.
- Read before: writing the verify gate

### Williams, Waterman & Patterson (2009) — Roofline
- Link / DOI: CACM 52(4)
- What it originated: operational intensity vs *measured* bandwidth and *measured* FLOP roofs.
- What I actually need from it: 305 GB/s / 16 B = 19.06 Gsamples/s is the physics-output roof. cuRAND Philox at 66.0 Gsamples/s is 4 B/uniform. Report both Gsamples/s and GB/s.
- Read before: writing HYPOTHESIS.md

### Alias-method Nsight reading
- Link: `parts/alias-method/results/nsight/reading.md`
- What it originated: K5 DRAM 54%, LG throttle 83%, 48 B XORWOW, 80% extra sectors, nsys setup 83%, (3,1) χ² 37.
- What I actually need from it: every number in this part’s hypothesis. Philox deletes the 48 B term. Do not reopen Vose.
- Read before: this part
