#!/usr/bin/env bash
set -euo pipefail

# Unlock NVIDIA GPU performance counters for Nsight Compute, then capture
# the naive-cuda reports. Polkit pops a password dialog (pkexec).
#
#   scripts/unlock-nsight.sh              permanent drop-in + ncu reports
#   scripts/unlock-nsight.sh --permanent  only write modprobe + mkinitcpio
#   scripts/unlock-nsight.sh --profile    only ncu as root (this session)
#
# Permanent takes effect after reboot. --profile works immediately.

root="$(cd "$(dirname "$0")/.." && pwd)"
self="$(readlink -f "$0")"
conf="/etc/modprobe.d/nvidia-profiling.conf"
out_ncu="$root/parts/naive-cuda/results/nsight/ncu"
bin="$root/cmake-build-release/cuda_qmc_exploration"
mode="both"

if [[ ${1:-} == "--permanent" ]]; then
  mode="permanent"
  shift
elif [[ ${1:-} == "--profile" ]]; then
  mode="profile"
  shift
elif [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  sed -n '2,12p' "$self"
  exit 0
fi

if [[ ${EUID} -ne 0 ]]; then
  if ! command -v pkexec >/dev/null; then
    echo "pkexec not found; install polkit" >&2
    exit 1
  fi
  exec pkexec "$self" --mode "$mode"
fi

# pkexec re-enters here as root. --mode is the internal flag.
if [[ ${1:-} == "--mode" ]]; then
  mode="$2"
fi

owner="${PKEXEC_UID:-${SUDO_UID:-}}"

permanent() {
  echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" >"$conf"
  chmod 644 "$conf"
  echo "wrote $conf"
  # This machine puts nvidia in the initramfs and boots Limine. `mkinitcpio -P`
  # has no presets here; limine-mkinitcpio is the real rebuild.
  if [[ -x /usr/bin/limine-mkinitcpio ]]; then
    /usr/bin/limine-mkinitcpio
  elif [[ -x /usr/bin/dracut ]]; then
    /usr/bin/dracut --force
  else
    echo "no limine-mkinitcpio/dracut; drop-in applies on next module load" >&2
  fi
  echo "permanent unlock staged. reboot before a user-session ncu will work."
}

profile_one() {
  local name="$1"
  local kernel="$2"
  local bench="$3"
  local skip="$4"
  echo "ncu $name ($kernel)"
  QMC_LOCK_SM_MHZ=1500 /usr/bin/ncu --set full \
    --kernel-name "regex:${kernel}" \
    --launch-skip "$skip" --launch-count 1 \
    -f -o "$out_ncu/$name" \
    "$bin" naive-bench --kernel "$bench" --n 1000000
  /usr/bin/ncu --import "$out_ncu/$name.ncu-rep" >"$out_ncu/$name.txt"
}

profile() {
  mkdir -p "$out_ncu"
  if [[ ! -x $bin ]]; then
    echo "missing $bin — build Release with -DQMC_LINEINFO=ON first" >&2
    exit 1
  fi
  export QMC_LOCK_SM_MHZ=1500
  profile_one sample_fp64 SampleFp64Kernel naive_fp64 3
  profile_one sample_fp32 SampleFp32Kernel naive_fp32 3
  profile_one setup_xorwow SetupXorwowKernel naive_fp64 0
  if [[ -n $owner ]]; then
    chown -R "$owner:$owner" "$root/parts/naive-cuda/results/nsight"
  fi
  echo "ncu reports in $out_ncu"
}

case "$mode" in
  permanent) permanent ;;
  profile) profile ;;
  both)
    permanent
    profile
    ;;
  *)
    echo "unknown mode $mode" >&2
    exit 2
    ;;
esac
