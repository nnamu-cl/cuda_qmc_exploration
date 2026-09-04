# Reading — what Nsight actually said

Quote these files. Official throughput: `../philox.json` (**4.60 Gsamples/s**), `../philox_ilp.json` (4.59), `../thrust_alias.json` (2.21), SM 1500 MHz. ncu replay ~1.15 GHz — stalls and ratios only.

## The picture in one table

| | K7 philox | K8 ilp S=1/128 | K5 linear (contrast) |
|---|---|---|---|
| Compute (SM) | **25.9%** | 23.2% | 10.1% |
| DRAM | **16.0%** | 16.6% | **54.4%** |
| ncu memory GB/s | 57 | 58 | **191** |
| ncu’s call | latency (LG) | latency (LG) | L2 + uncoalesced |
| Top stall | **LG throttle 71%** | LG 68% | LG **83%** |
| Occupancy | 94.0% (theory 100%) | **68.9%** | 94.9% |
| Uncoalesced extra sectors | **83%** | 83% | 80% |
| L1 / L2 hit | 64.8% / **99.9%** | 66.4% / 99.7% | 61.5% / 91.9% |
| Branch efficiency | **100%** | 100% | 100% |
| FP32 peak used | **10%** | 10% | **5%** |
| Inst at 1e6 samples | **13.3e6** | 12.4e6 | 11.0e6 |

ncu SM 1.15 GHz. JSON: philox **4.60 Gsamples/s**, philox_ilp 4.59, thrust 2.21. K5 was **1.76**. K4 was **2.72**.

## K7 — stateless Philox

SOL: *“low compute throughput and memory bandwidth … typically indicate latency issues.”* DRAM **16%**, not K5’s 54%. The 96 B XORWOW load/store is gone. L2 hit **99.9%**: the 147 KB alias tables now sit in L2 without XORWOW state traffic next to them. Uncoalesced extra sectors still **83%** — two data-dependent 24 B `AliasBin` records per sample. That is the remaining LG queue.

Top stall: LG queue full, 20.4 of 28.6 cycles between issues (**71%**). Same *name* as K5 (83%), less of a longer-ago gap, and **no 48 B state**. FP32 peak **10%** (K5 was 5%). Occupancy 94%, branch 100%. Setup kernel: **none**.

13.3e6 instructions vs K5’s 11.0e6 at 1e6 samples. Two Philox4x32-10 (ten rounds each) cost more integer than XORWOW updates. They do not cost 96 B of DRAM. JSON 4.60 / 1.76 ≈ **2.6×**.

## K8 — grid-stride / ILP

S=1, 128 threads (sweep winner). DRAM 17%, LG 68%, FP32 10%, occupancy **69%**. JSON 4.59 vs K7 4.60. **ILP did not pay.** Sweep: every S>1 is slower (128×8 = 3.57 Gsamples/s). Volkov’s independent chains need a latency the extra registers can hide; here the stall is still “LG queue full” from the same two scattered loads, and occupancy dropped.

L2 persist window: `cudaStreamSetAttribute` returned invalid argument on this Max-Q. Noise, then unsupported.

## nsys timeline (1e6 samples)

`nsys/philox.txt`: **100%** `SamplePhiloxKernel`, 23 launches (3 warmup + 20). No `SetupXorwow`. K5 was 83% setup / 17% sample. The 163 ms bar is gone.

`nsys/thrust_alias.txt`: cub `for_each` **63%**, `gen_sequenced_Philox` **32%**, `generate_seed_pseudo` **5%**. The idiomatic pipeline still has a seed kernel; it is 5%, not 83%. Extra 28 B of uniforms is the 32% generate bar.

## What this changes

Philox deleted the 48 B state and the setup kernel. It did not delete the uncoalesced alias records, so we are not on the 16 B DRAM roof (19.06 Gsamples/s). 4.60 is 4.1× off that roof. The Drop 1 sentence that sampling costs “about as much as the bits” is **not** earned on this card: cuRAND Philox raw is 66.0 Gsamples/s / 264 GB/s; we emit 16 B of physics at 73.5 GB/s. Stop. Do not open Drop 2.
