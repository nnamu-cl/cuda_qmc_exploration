# Nsight reports — naive CUDA

Article source for Part 2 profiler beats. Raw dumps live here; the argument is in `reading.md`. Do not quote stalls from memory — quote these files.

## Layout

```
nsight/
  README.md                 this index
  reading.md                patterns, limits, why inverse-table is next (article notes)
  ncu/                      Nsight Compute (kernel)
    sample_fp64.ncu-rep     SampleFp64Kernel, 1e6 samples, --set full, skip 3 warmups
    sample_fp32.ncu-rep     SampleFp32Kernel, same
    setup_xorwow.ncu-rep    SetupXorwowKernel (the 182 ms tax)
    *.txt                   text export of the same reports
  nsys/                     Nsight Systems (CPU + GPU timeline)
    naive_fp64.nsys-rep
    naive_fp32.nsys-rep
    *.txt                   nsys stats
```

## Open later

```bash
ncu-ui parts/naive-cuda/results/nsight/ncu/sample_fp64.ncu-rep
nsys-ui parts/naive-cuda/results/nsight/nsys/naive_fp64.nsys-rep
```

Text without the GUI:

```bash
ncu --import parts/naive-cuda/results/nsight/ncu/sample_fp64.ncu-rep
nsys stats parts/naive-cuda/results/nsight/nsys/naive_fp64.nsys-rep
```

## Unlock GPU counters (password popup)

`ncu --set full` needs admin counters. From the repo root, this pops a polkit dialog:

```bash
scripts/unlock-nsight.sh
```

Writes `/etc/modprobe.d/nvidia-profiling.conf` (reboot before user-session `ncu` works) **and** captures the three `.ncu-rep` files as root now. `--permanent` / `--profile` for one half only.

## How they were taken

- Binary: `cmake-build-release` with `-DQMC_LINEINFO=ON` (not the JSON-timing binary story; clocks still 1500 MHz)
- Command: `naive-bench --n 1000000` (plan: 1–10 M is enough; ncu serializes)
- `ncu --set full --kernel-name <exact> --launch-skip 3 --launch-count 1`
- Never Debug. Never `--use_fast_math`.

Timing JSON in `../naive_fp64.json` etc. stays the official throughput. These reports explain the gap, they do not replace the JSON.
