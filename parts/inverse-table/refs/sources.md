# Reading — inverse table

### Devroye (1986) — Non-Uniform Random Variate Generation, ch. II (inversion), skim ch. III (tables)
- Link: https://luc.devroye.org/rnbookindex.html (author PDF)
- What it originated: inverse-transform sampling; tabulated quantile F⁻¹(u) as the O(1) version of inversion
- What I actually need from it: K4 is `idx = u*(K−1)`, lerp adjacent F⁻¹ values. Interpolation error across a jump in F⁻¹ is the node trap. Credit line: “tabulated quantile function (Devroye 1986, ch. II–III)”
- Read before: this part

### NVIDIA CUDA C++ Programming Guide — shared memory, occupancy, bank conflicts
- Link: CUDA programming guide, memory-hierarchy / occupancy sections
- What it originated: 32 banks, 32-bit words; data-dependent indices conflict; Turing 64 KB combined L1/shared; occupancy vs shared-per-block
- What I actually need from it: K3 prediction uses bank conflicts + occupancy, not “divergence.” Measure `l1tex__data_bank_conflicts` (or the Turing equivalent ncu names). Occupancy from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, not a hand spreadsheet
- Read before: this part

### NVIDIA Nsight Compute — warp stall reasons / source counters
- Link: Nsight Compute docs, “Warp Scheduler States”
- What it originated: LG throttle vs long scoreboard vs math-pipe busy
- What I actually need from it: K2 leftover is LG throttle 54% (XORWOW) + scoreboard 31% (walk). K3 can only touch the second. K4 should drop the 23-load scoreboard
- Read before: this part (quote `parts/naive-cuda/results/nsight/reading.md`)

### Williams, Waterman & Patterson (2009) — Roofline
- Link / DOI: CACM 52(4)
- What it originated: operational intensity vs measured bandwidth and measured FLOP roofs
- What I actually need from it: XORWOW+out 112 B vs measured 305 GB/s = 2.72 Gsamples/s. Do not revive the 344 B DRAM model
- Read before: writing HYPOTHESIS.md

### CUDA occupancy API
- Link: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (CUDA runtime)
- What it originated: occupancy from the compiler’s register/shared footprint, not from guessing
- What I actually need from it: the occupancy-vs-shared table in `experiments/`
- Read before: K3 occupancy sweep
