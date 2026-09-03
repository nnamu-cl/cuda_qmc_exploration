# Reading — naive CUDA

### Williams, Waterman & Patterson (2009) — Roofline: An Insightful Visual Performance Model
- Link / DOI: CACM 52(4)
- What it originated: operational intensity vs measured bandwidth and measured FLOP roofs
- What I actually need from it: K1/K2 predictions use `bw_copy` 305 GB/s and `fma64` 88.2 GFLOPS, never the 448 GB/s / 0.2 TFLOPS spec-sheet pair
- Read before: this part

### CUDA C++ Programming Guide §5.5 — Mathematical Functions (ULP tables)
- Link: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html
- What it originated: IEEE vs intrinsic accuracy; `__expf` / `__sincosf` / `sinf` ULP bounds
- What I actually need from it: K2 is IEEE-32 (`expf`, `sincosf`), not `--use_fast_math`. The fast variant quotes these ULP numbers next to a measured extra error
- Read before: this part

### CUDA Math API — single-precision functions and intrinsics
- Link: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html
- What it originated: `sincosf` vs `__sincosf`, `expf` vs `__expf`
- What I actually need from it: call the documented functions; do not roll a fast-exp
- Read before: this part

### cuRAND device API — XORWOW `curand_init` / `curand_uniform_double`
- Link: https://docs.nvidia.com/cuda/curand/device-api-overview.html
- What it originated: tutorial default (one `curandStateXORWOW` per thread, sequence = thread id); 48 B state; setup is slower than generation; save/restore state in global memory
- What I actually need from it: K1/K2 use this default on purpose. Setup kernel time is reported, not folded into sample/s. Philox comes in Part 5
- Read before: this part

### NVIDIA Turing whitepaper — GeForce FP64 rate
- Link: NVIDIA Turing architecture whitepaper
- What it originated: consumer Turing FP64 is 1/32 of FP32 (two FP64 units/SM vs 64 FP32 cores)
- What I actually need from it: the segmentation story in the article; the *number* is the measured 88.2 GFLOPS, not 1/32 × spec FP32
- Read before: writing only

### Nsight Compute — warp stall reasons
- Link: Nsight Compute docs, "Warp Scheduler States"
- What it originated: `stalled_long_scoreboard` vs math-pipe busy
- What I actually need from it: the metric that falsifies "divergent CDF walks" vs "FP64 pipe"
- Read before: after the first kernel timing

### Devroye (1986) ch. II — inverse-transform sampling
- Link: https://luc.devroye.org/rnbookindex.html
- What it originated: piecewise-linear CDF inversion (already in the CPU reference)
- What I actually need from it: port the interpolated invert, never the bin-edge visualizer
- Read before: this part
