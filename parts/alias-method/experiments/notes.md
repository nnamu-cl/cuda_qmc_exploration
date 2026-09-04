# Experiments — alias method

## Uniform vs linear interior

```bash
./cmake-build-release/cuda_qmc_exploration alias-interior \
  --n 1000000 --out parts/alias-method/experiments/uniform_vs_linear.json
```

(1,0,0), 256 radial χ² bins. Measured at 1e6: both **247.7**. Keep linear (matches the trapezoid CDF), not because χ² split them.

## Node hole vs inverse-table

```bash
./cmake-build-release/cuda_qmc_exploration alias-hole \
  --n 10000000 --out parts/alias-method/experiments/node_hole.json
```

(3,1,−1), node window r ∈ [4.5, 7.5], 1e7 samples. Measured: inverse_table **11571.7**, alias_uniform **37.0**, alias_linear **37.1**.

## Official benches

Only after Release `verify-full`.

```bash
export QMC_LOCK_SM_MHZ=1500
scripts/bench.sh ./cmake-build-release/cuda_qmc_exploration alias-bench \
  --kernel alias_linear --n 8000000 \
  --out parts/alias-method/results/alias_linear.json --official
# alias_uniform, alias_float4, alias_split the same
```
