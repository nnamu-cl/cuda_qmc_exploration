# Nsight reports — alias method

Article source for Part 4 profiler beats. Quote these files, not memory.

## Layout

```
nsight/
  README.md
  reading.md
  ncu/
    alias_linear.ncu-rep / .txt          SampleAliasKernel<true,false>, 1e6, --set full, skip 3
    alias_float4.ncu-rep / .txt          SampleAliasKernel<true,true>, same
    alias_split_sample.ncu-rep / .txt    SampleSplitCoordsKernel, same
    alias_split_weight.ncu-rep / .txt    WeightSplitKernel, same
  nsys/
    alias_linear.nsys-rep / .txt
    alias_float4.nsys-rep / .txt
    alias_split.nsys-rep / .txt
```

## Open later

```bash
ncu-ui parts/alias-method/results/nsight/ncu/alias_linear.ncu-rep
nsys-ui parts/alias-method/results/nsight/nsys/alias_linear.nsys-rep
```

## How they were taken

- Binary: `cmake-build-release` `-DQMC_LINEINFO=ON`
- `alias-bench --n 1000000`
- `ncu --set full --kernel-name regex:<Kernel> --launch-skip 3 --launch-count 1` via `pkexec`
- ncu SM ~1.15 GHz on replay. Official Gsamples/s stay in `../alias_*.json` (1500 MHz).
