#!/usr/bin/env bash
# Nsight Compute captures for the packed-records part. Needs root for perf counters:
#   sudo bash scripts/ncu_packed.sh
# Three captures: the winner in its fast regime (8M), the same kernel in the
# footprint-cliff regime (64M), and the global-path packed kernel (8M).
set -euo pipefail
cd "$(dirname "$0")/.."

NCU=${NCU:-/usr/bin/ncu}
BIN=./cmake-build-release/cuda_qmc_exploration
OUT=parts/packed-records/results/nsight/ncu
mkdir -p "$OUT"

capture() { # name, extra bench args...
  local name=$1; shift
  "$NCU" --set full --launch-skip 3 --launch-count 1 -f -o "$OUT/$name" \
    "$BIN" packed-bench "$@"
  "$NCU" --import "$OUT/$name.ncu-rep" --page details > "$OUT/$name.txt"
}

capture packed_shared_8m  --kernel packed_shared --threads 512 --n 8000000
capture packed_shared_64m --kernel packed_shared --threads 512 --n 64000000
capture packed_8m         --kernel packed        --threads 128 --n 8000000

chown -R "$(stat -c %U:%G .git)" "$OUT"
echo "done: $OUT/{packed_shared_8m,packed_shared_64m,packed_8m}.{ncu-rep,txt}"
