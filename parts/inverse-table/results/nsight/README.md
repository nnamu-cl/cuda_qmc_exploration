# Nsight reports — inverse table

Article source for Part 3 profiler beats. Quote these files, not memory.

## Layout

```
nsight/
  README.md
  reading.md
  ncu/
    shared_cdf.ncu-rep / .txt     SampleSharedCdfKernel, 1e6, --set full, skip 3
    inverse_table.ncu-rep / .txt  SampleInverseKernel<false>, same
  nsys/
    shared_cdf.nsys-rep / .txt
    inverse_table.nsys-rep / .txt
```

## Open later

```bash
ncu-ui parts/inverse-table/results/nsight/ncu/shared_cdf.ncu-rep
ncu-ui parts/inverse-table/results/nsight/ncu/inverse_table.ncu-rep
nsys-ui parts/inverse-table/results/nsight/nsys/inverse_table.nsys-rep
```

## How they were taken

- Binary: `cmake-build-release` `-DQMC_LINEINFO=ON`
- `inverse-bench --n 1000000`
- `ncu --set full --kernel-name regex:<Kernel> --launch-skip 3 --launch-count 1`
- ncu SM ~1.17 GHz on replay. Official Gsamples/s stay in `../shared_cdf.json` / `../inverse_table.json` (1500 MHz).
