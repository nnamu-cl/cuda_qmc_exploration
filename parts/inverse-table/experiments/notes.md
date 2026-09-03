# Experiments — inverse table

## Occupancy vs shared memory

```bash
./cmake-build-release/cuda_qmc_exploration inverse-occupancy \
  --n 8000000 --out parts/inverse-table/experiments/occupancy.json
```

Threads 128 / 256 / 512, CDF-only (24 KB, 2 blocks/SM) vs CDF+nodes (48 KB, 1 block). Occupancy from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

Measured (8e6, 1500 MHz): 0.73 / 1.37 / 1.63 Gsamples/s CDF-only; 0.40 / 0.96 / 1.58 with nodes. Best K3 (512, CDF-only) still under naive FP32 1.78.

## Node trap vs table size

```bash
./cmake-build-release/cuda_qmc_exploration inverse-trap \
  --n 1000000 --out parts/inverse-table/experiments/node_trap.json
```

(3,1,−1), K ∈ {256, 1024, 4096, 16384, 65536}, uniform lerp vs node-aware snap. χ² on 256 radial bins and on r ∈ [4.5, 7.5].

## Official benches

Only after Release `verify-full`.

```bash
export QMC_LOCK_SM_MHZ=1500
scripts/bench.sh ./cmake-build-release/cuda_qmc_exploration inverse-bench \
  --kernel shared_cdf --n 8000000 \
  --out parts/inverse-table/results/shared_cdf.json --official
scripts/bench.sh ./cmake-build-release/cuda_qmc_exploration inverse-bench \
  --kernel inverse_table --n 8000000 \
  --out parts/inverse-table/results/inverse_table.json --official
scripts/bench.sh ./cmake-build-release/cuda_qmc_exploration inverse-bench \
  --kernel inverse_nodes --n 8000000 \
  --out parts/inverse-table/results/inverse_nodes.json --official
```
