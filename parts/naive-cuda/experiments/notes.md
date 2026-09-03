# Experiments — naive CUDA

## FP32 vs FP64 special functions

```bash
./cmake-build-release/cuda_qmc_exploration naive-error \
  --out parts/naive-cuda/experiments/fp32_vs_fp64.json
```

Host eval of the same recurrences (`device_math.cuh`) in `float` vs `double` on a 200-point r grid per (n,l) up to 6, and P_l^m on x ∈ [−1,1]. Also ρ^l: iterated multiply vs `exp(l log ρ)`.

JSON keys: `radial_max_rel_error`, `radial_mean_rel_error`, `legendre_*`, `rho_l_iterated_max_rel_error`, `rho_l_explog_max_rel_error`, plus per-orbital rows.

## Official benches

Only after Release `verify-full` (gate file is per build dir).

```bash
export QMC_LOCK_SM_MHZ=1500
scripts/bench.sh ./cmake-build-release/cuda_qmc_exploration naive-bench \
  --kernel naive_fp64 --n 8000000 \
  --out parts/naive-cuda/results/naive_fp64.json --official
# naive_fp32, naive_fp32_fast the same
```

Kernel-only time (CUDA events). XORWOW `curand_init` is printed as `setup_ms` and is not in `gsamples_per_s`.
