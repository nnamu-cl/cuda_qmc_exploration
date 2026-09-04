# Experiments — endgame

## Philox bits

Host Random123 rounds in `harness/rng/philox.h` vs the device dump kernel. Must match. cuRAND host API uses a different counter packing — record the mismatch, do not rewrite Philox to chase it.

```bash
./cmake-build-release/cuda_qmc_exploration endgame-bits \
  --n 4096 --out parts/endgame/experiments/philox_bit_check.json
```

## Launch sweep (Harris grid-stride × Volkov S)

S ∈ {1,2,4,8} × blocks {128,256,512}, plus one persist-window cell at 256×4.

```bash
./cmake-build-release/cuda_qmc_exploration endgame-sweep \
  --n 8000000 --out parts/endgame/experiments/launch_sweep.json
```

Winning cell was **128 threads, S=1** at 4.58 Gsamples/s — identical to K7. Every S>1 is slower. Persist returned `invalid argument`.

## Official benches

Only after Release `verify-full`. SM is often already 1500 — do not sudo unless `nvidia-smi` disagrees.

```bash
export QMC_LOCK_SM_MHZ=1500
scripts/bench.sh ./cmake-build-release/cuda_qmc_exploration endgame-bench \
  --kernel philox --n 8000000 \
  --out parts/endgame/results/philox.json --official
scripts/bench.sh ./cmake-build-release/cuda_qmc_exploration endgame-bench \
  --kernel philox_ilp --samples-per-thread 4 --n 8000000 \
  --out parts/endgame/results/philox_ilp.json --official
scripts/bench.sh ./cmake-build-release/cuda_qmc_exploration endgame-bench \
  --kernel thrust_alias --n 8000000 \
  --out parts/endgame/results/thrust_alias.json --official
```

## Sample dump (density renders)

```bash
./cmake-build-release/cuda_qmc_exploration endgame-dump \
  --n 1000000 --principal 3 --l 1 --m -1 \
  --prefix parts/endgame/experiments/density_3p
python scripts/plot/render_density.py \
  --meta parts/endgame/experiments/density_3p.json \
  --out parts/endgame/experiments/density_3p.png
```

`--use_fast_math` stays off globally. Persistence is the one optional squeeze in the sweep.
