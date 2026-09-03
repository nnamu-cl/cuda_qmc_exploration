# Reading — alias method

### Keith Schwarz — Darts, Dice, and Coins: Sampling from a Discrete Distribution (2011)
- Link: https://www.keithschwarz.com/darts-dice-coins/
- What it originated: the pedagogical derivation of Walker’s alias method (fair die + biased coin; the two-rectangle “darts” picture) and a correct O(n) Vose build (small/large stacks, `scaled[g] = (scaled[g] + scaled[l]) - 1`)
- What I actually need from it: do not invent an alias builder. Generation is: draw column `j = ⌊n U₁⌋`, keep `j` if `U₂ < Prob[j]`, else take `Alias[j]`. The leftover-Small `Prob=1` pass is numerical, not a third algorithm
- Read before: this part (before HYPOTHESIS.md)

### Walker (1974) — New fast method for generating discrete random numbers with arbitrary frequency distributions
- Link / DOI: Electronics Letters 10(8) 127–128
- What it originated: the alias table itself
- What I actually need from it: credit line “alias method (Walker 1974; 1977)”
- Read before: writing only

### Walker (1977) — An Efficient Method for Generating Discrete Random Variables with General Distributions
- Link / DOI: ACM TOMS 3(3) 253–256
- What it originated: the practical generation procedure used here (two uniforms)
- What I actually need from it: generation, not the original O(n²) build
- Read before: writing only

### Vose (1991) — A Linear Algorithm for Generating Random Numbers with a Given Distribution
- Link / DOI: IEEE Trans. Software Eng. 17(9) 972–975
- What it originated: robust O(K) table construction
- What I actually need from it: Schwarz’s version of the O(K) build (Vose’s paper has a known off-by-one presentation; do not transcribe the paper blindly)
- Read before: implementing `alias_build.cpp`

### Williams, Waterman & Patterson (2009) — Roofline
- Link / DOI: CACM 52(4)
- What it originated: operational intensity vs measured bandwidth and measured FLOP roofs
- What I actually need from it: K4 already sits on 305 GB/s / 112 B = 2.72 Gsamples/s. Alias does not change the XORWOW 48 B term
- Read before: writing HYPOTHESIS.md

### Inverse-table Nsight reading
- Link: `parts/inverse-table/results/nsight/reading.md`
- What it originated: K4 DRAM 86%, LG throttle 73%, FP32 5%, (3,1) node-window χ² 11572
- What I actually need from it: every number in this part’s hypothesis; the node trap is the thing alias must kill
- Read before: this part
