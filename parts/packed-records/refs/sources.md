# Reading — packed-records (record layout, vector loads, sector utilisation)

### Vose (1991) — A Linear Algorithm for Generating Random Numbers with a Given Distribution
- Link / DOI: IEEE TSE 17(9), 972–975, https://doi.org/10.1109/32.92917
- What it originated: the O(n) small/large worklist construction of Walker's alias tables. The
  sampling-time state of an alias table is exactly **two** numbers per bin — a cutoff `prob` and
  an `alias` index. Everything else a caller wants to know about bin *j* (where it starts, how
  wide it is, what the density does inside it) is *not* alias-method state; it is geometry the
  caller chose to staple onto the same struct.
- What I actually need from it: the separation. `{prob, alias}` is 8 bytes and is what the draw
  reads. The interior geometry is a different array, read once, for a different index. K5/K7
  fused them into one 24-byte `AliasBin` and paid for it on every probe.
- Read before: this part (before HYPOTHESIS.md)

### Walker (1977) — An Efficient Method for Generating Discrete Random Variables with General Distributions
- Link / DOI: ACM TOMS 3(3), 253–256, https://doi.org/10.1145/355744.355749
- What it originated: the alias table itself. Attributed in `parts/alias-method/refs/sources.md`;
  repeated here only because this part changes the table's *layout*, not its *contents*. The Vose
  bins are byte-for-byte the ones K5 and K7 used — see `verify/` for the reconstruction check.
- Read before: this part

### NVIDIA CUDA C++ Programming Guide — Device Memory Accesses / coalescing
- Link: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses
- What it originated: the rule that a warp's global access is serviced in 32-byte sectors, and
  that a naturally aligned 8- or 16-byte access is serviced in *one* instruction while an
  unaligned or under-aligned struct is split into scalar accesses.
- What I actually need from it: `AliasBin` is 24 bytes with no `alignas`, so nvcc must assume
  4-byte alignment and cannot vector-load it. `alignas(8)` and `alignas(16)` on a POD whose size
  matches are the documented way to get `LD.64` / `LD.128`.
- Read before: writing the packed structs

### NVIDIA CUDA Best Practices Guide — Coalesced Access to Global Memory
- Link: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory
- What it originated: the "bytes utilised per sector" framing that Nsight Compute reports back as
  *"On average, only N of the 32 bytes transmitted per sector are utilized by each thread."*
- What I actually need from it: K7's ncu says **4.2 of 32 bytes**. That is the fingerprint of a
  scalar 32-bit gather, and it is the number this part is trying to move.
- Read before: HYPOTHESIS.md

### Luitjens — CUDA Pro Tip: Increase Performance with Vectorized Memory Access
- Link: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- What it originated: replacing N scalar loads with one wide load reduces *instruction* count and
  the number of L1 wavefronts, even when the byte count is unchanged.
- What I actually need from it: the unit of the prediction. For a fully divergent gather the L1TEX
  cost is (number of load instructions) × (32 lanes), not (bytes). Merging four `LDG.E` into one
  `LDG.E.64` is a 4× cut in that pipe even though the same bytes move.
- Read before: HYPOTHESIS.md arithmetic

### Volkov (2010) — Better Performance at Lower Occupancy
- Link: GTC 2010, https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf
- What I actually need from it: K8 already measured that S>1 does not pay while the stall is
  "LG queue full". If this part empties that queue, the ILP question is *reopened*, not settled.
  Retest S ∈ {2,4} on the winner only, and only if ncu says the LG throttle actually moved.
- Read before: the ILP experiment (not before the main kernels)

### Salmon, Moraes, Dror & Shaw (2011) — Parallel Random Numbers: As Easy as 1, 2, 3
- Link: SC'11, https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
- What I actually need from it: nothing new. The counter scheme is frozen — counter is
  `(sample_index, draw_slot)`, key is the seed, and this part must not touch it. It is repeated
  here because the bitwise launch-config gate depends on it and this part adds new launch shapes
  (1024-thread blocks for the shared variant).
- Read before: writing the verify gate

### Endgame Nsight reading
- Link: `parts/endgame/results/nsight/reading.md`, `parts/endgame/results/nsight/ncu/philox.txt`
- What it originated: K7 at 4.60 Gsamples/s with L1/TEX throughput **96.14%**, DRAM **16.04%**,
  Compute (SM) **25.90%**, LG throttle **71.3%** of 28.6 cycles, uncoalesced extra sectors 83%,
  4.2 of 32 bytes per sector utilised, occupancy 94%.
- What I actually need from it: every referee number in this part's hypothesis.
- Read before: this part

### Part 4 implementation plan (private)
- Link: `private/implementation_plan/06_part4_alias_method.md`
- What it originated: *"`float prob + int alias` 8 B record"* and *"Alias draw: 2×8 B table
  records"*. The plan said eight bytes. The implementation shipped twenty-four.
- What I actually need from it: this part is not a new idea. It is the plan's original layout,
  arrived at two parts late, with an ncu report explaining what the drift cost.
- Read before: this part
