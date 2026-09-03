#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

allow_unlocked=0
if [[ "${1:-}" == "--allow-unlocked" ]]; then
  allow_unlocked=1
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/bench.sh [--allow-unlocked] <command>..." >&2
  exit 2
fi

lock_mhz="${QMC_LOCK_SM_MHZ:-1500}"
sm_mhz="$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | awk '{print int($1+0)}')"
export QMC_LOCK_SM_MHZ="$lock_mhz"

if [[ "$sm_mhz" -ne "$lock_mhz" ]]; then
  echo "SM clock is ${sm_mhz} MHz; expected ${lock_mhz} (QMC_LOCK_SM_MHZ)."
  echo "This Max-Q: graphics lock works, memory lock does not."
  echo "  sudo nvidia-smi -pm 1"
  echo "  sudo nvidia-smi -lgc ${lock_mhz},${lock_mhz}"
  echo "Do not use -lmc (unsupported here). Applications Clocks reads Not Active even when -lgc stuck."
  echo "Re-run with --allow-unlocked only for harness smoke, never for article numbers."
  if [[ "$allow_unlocked" -ne 1 ]]; then
    exit 2
  fi
fi

clock_log="${QMC_CLOCK_LOG:-}"
smi_pid=""
if [[ -n "$clock_log" ]]; then
  mkdir -p "$(dirname "$clock_log")"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -lms 100 >"$clock_log" &
  smi_pid=$!
  trap 'if [[ -n "$smi_pid" ]]; then kill "$smi_pid" 2>/dev/null || true; fi' EXIT
fi

nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
"$@"
