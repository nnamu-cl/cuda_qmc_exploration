# Reading — what Nsight actually said

Quote these files, not memory. Official throughput stays in `../naive_fp64.json` (1500 MHz). ncu ran at **SM 1.17 GHz** (replay drops the graphics lock) — use ncu for *ratios and stall names*, not for Gsamples/s.

Open later: `ncu-ui ncu/sample_fp64.ncu-rep`, `nsys-ui nsys/naive_fp64.nsys-rep`.

## The picture in one table

| | K1 FP64 `sample_fp64` | K2 FP32 `sample_fp32` | XORWOW setup |
|---|---|---|---|
| Compute (SM) | **84.8%** | 30.9% | **90.1%** |
| DRAM | 19.4% | **52.3%** | 3.0% |
| ncu's call | FP64 pipe + L1TEX scoreboard | latency (LG full + scoreboard) | LSU / skipahead |
| Top stall | **long scoreboard 65.5%** (42.9 cy L1TEX) | **LG throttle 53.6%** then scoreboard 30.8% | LG throttle 73.7% |
| L1 / L2 hit | 79% / 84% | 87% / 84% | 98% / 94% |
| Uncoalesced extra sectors | **68%** (8.0 B of 32) | **78%** (4.9 B of 32) | 19% |
| Branch efficiency | **100%** (divergent branches 0.01) | 100% | 97.8% (predicated 16.6/32) |
| Occupancy | 71% (66 regs) | 95% (49 regs) | 95% (63 regs) |
| FP64 peak used | **53%** | 0% | 0% |
| FP32 peak used | 0% | **3%** | 0% |
| Local spills | 0 | 0 | 0 |

ncu SOL on K1: *“utilizing greater than 80% … start with Compute Workload Analysis.”* Then: *“FP64 is the highest-utilized pipeline (84.8%) … over-utilized and likely a performance bottleneck.”* Roofline: *32:1 FP32:FP64; this workload 53% of FP64 peak, 0% of FP32. If FP64 bound, consider 32-bit.* That is the K2 chapter, from the tool, not from us.

K2 SOL: *“low compute and memory relative to peak … typically latency. Look at Warp State Statistics.”* Not “you are DRAM-bound at 305 GB/s.” DRAM 52% with 3% of FP32 peak.

## What the hypothesis got right
- K1 is not the copy roof (DRAM 19%).
- K2 is a few times K1, not 32× (ncu: 53% of the FP64 peak on K1; K2 never sees that pipe).
- XORWOW setup is a different kernel and a real tax. nsys GPU kernel time at 1e6: setup **60%** (3 launches) vs sample **40%** (23 launches). ncu setup duration 22 ms / 1e6 states ≈ 176 ms / 8e6, matches the 182 ms JSON.

## What the hypothesis got wrong (article must say this)
1. **Tables are not DRAM.** L2 hit **84%** on both sample kernels. Charging 23 × 8 B against `bw_copy` was the wrong roof. ncu memory throughput K1 is 68 GB/s, not 192.
2. **The binary search does not warp-diverge.** Branch efficiency 100%, 0.01 divergent branches, 32 active threads/warp. The compiler predicates the walk. The leftover is **uncoalesced L1TEX** (8 B used per 32 B sector) plus scoreboard waits, not `stalled` on taken branches.
3. **K1 is FP64-pipe bound first, scoreboard second.** I had them in the other order. 84.8% FP64 pipe is the SOL headline; long scoreboard is *why issue slots are empty* (91% no-eligible). Both can be true.
4. **K2 is not math-bound.** 3% of FP32 peak. ALU (integer) is the busiest compute pipe at 31%. `__expf` could not help; ncu already said the leftover is LG queue + uncoalesced loads. That matches the 1.76 vs 1.78 Gsamples/s tie.
5. **Setup is not DRAM.** 3% DRAM, 90% mem-pipes/LSU, 1.3e9 instructions for 1e6 inits. Skipahead arithmetic. Philox (Part 5) deletes this kernel, it does not “make memcpy faster.”

## Next-step argument (inverse-table, then Philox)

Do **not** chase `__expf`, occupancy, or a 344 B DRAM model.

- **Part 3 (inverse table):** kill the 23 uncoalesced predicated loads. ncu estimates **56–75%** speedup from fixing uncoalesced global access on K1/K2. Direct `u → x` lookup is that fix. Shared memory for a small table is ncu’s other hint; a dense inverse CDF is the science version of “move hot data closer.”
- **Part 5 (Philox):** kill `SetupXorwowKernel` (22 ms vs 2 ms sample at 1e6) and the 48 B state load/store that feeds K2’s LG throttle. Not this part.

K2 already did ncu’s FP64→FP32 suggestion (53% of FP64 peak → 3% of FP32 peak, 3.17× wall clock). The remaining K2 SOL is latency from the walk + XORWOW traffic. That is why inverse-table is next and why the fast-math kernel was a blank.

## nsys timeline (CPU + GPU)

`nsys/naive_fp64.nsys-rep` — `cuda_gpu_kern_sum`:

- `SetupXorwowKernel` × 3: 57.8 ms total (avg 19.3 ms)
- `SampleFp64Kernel` × 23: 38.5 ms total (avg 1.68 ms)
- H2D tables: 8.5 µs, 0.098 MB (noise)

CPU side is `poll` / `cudaEventSynchronize` / `cudaMalloc` (222 ms one-time). Article: setup is the first bar on the timeline, sample is the short repeating bar. Do not fold setup into Gsamples/s; do not pretend it is free.

## Limits of these reports
- ncu SM 1.17 GHz vs locked 1.50 GHz on the JSON. Never mix.
- `--set full` replays; occupancy/stalls are the portable facts.
- No `sample_fp32` nsys yet (optional; K1 timeline already shows the setup tax).
- Source-counter line hits need `ncu-ui` (the `.ncu-rep` files). The `.txt` is the quote sheet.

## File map
| File | Use in article |
|---|---|
| `ncu/sample_fp64.txt` §SOL, §Compute, §Warp State, §Source Counters | K1 villain = FP64 pipe 84.8% + uncoalesced 68% |
| `ncu/sample_fp32.txt` §SOL, §Warp State, §Source Counters | K2 villain = latency / LG full, 3% FP32 peak, 78% uncoalesced |
| `ncu/setup_xorwow.txt` §SOL, §Compute | skipahead, 3% DRAM, 90% LSU |
| `nsys/naive_fp64.nsys-rep` GPU kernel sum | setup 60% vs sample 40% at this launch mix |
