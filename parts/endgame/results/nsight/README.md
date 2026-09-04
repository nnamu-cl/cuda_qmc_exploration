# Nsight reports — endgame

Article source for Part 5 profiler beats. Quote these files, not memory.

## Layout

```
nsight/
  README.md
  reading.md
  ncu/
    philox.ncu-rep / .txt           SamplePhiloxKernel, 1e6, --set full, skip 3
    philox_ilp.ncu-rep / .txt       SamplePhiloxIlpKernel<1>, 128 threads, same
    thrust_alias.ncu-rep / .txt     first two kernels after skip 3 (cuRAND seed + generate)
  nsys/
    philox.nsys-rep / .txt
    philox_ilp.nsys-rep / .txt
    thrust_alias.nsys-rep / .txt
```

`.sqlite` is local-only. Do not stage it.

## Open later

```bash
ncu-ui parts/endgame/results/nsight/ncu/philox.ncu-rep
nsys-ui parts/endgame/results/nsight/nsys/philox.nsys-rep
```

## How they were taken

- Binary: `cmake-build-release` `-DQMC_LINEINFO=ON`
- `endgame-bench --n 1000000`
- `ncu --set full --kernel-name regex:SamplePhiloxKernel --launch-skip 3 --launch-count 1` via `pkexec`
- ncu SM ~1.15 GHz on replay. Official Gsamples/s stay in `../philox.json` (1500 MHz).
