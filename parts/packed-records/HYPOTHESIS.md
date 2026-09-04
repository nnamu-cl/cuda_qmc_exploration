# Hypothesis — packed alias records

## Claim
Kernel 9 (same Vose bins as K5/K7, same stateless Philox, **24-byte `AliasBin` split into an
8-byte `{prob, alias}` draw array and an 8-byte `{lo, width}` interior array, uniform within-bin**)
will reach **10.5 Gsamples/s**, a **2.3×** on K7's 4.60. The binding constraint moves but does not
leave: L1TEX stays the roof, at about a third of the cycles it spends today. Staging the draw
arrays in shared memory will **not** pay. The 16 B DRAM roof (19.06 Gsamples/s) stays out of reach.

## Arithmetic

Referees are **measured** Release JSON and the **existing K7 ncu report**, same RTX 2070 Super
Max-Q, SM 1500 MHz. ncu replay sat at ~1.15 GHz — stalls, ratios and SASS only, never mixed with
JSON Gsamples/s.

| Referee | File | Number |
|---|---|---|
| philox (K7) | `parts/endgame/results/philox.json` | **4.60 Gsamples/s** |
| philox_ilp (K8, S=1/128) | `parts/endgame/results/philox_ilp.json` | 4.59 Gsamples/s |
| thrust_alias | `parts/endgame/results/thrust_alias.json` | 2.21 Gsamples/s |
| alias linear (K5) | `parts/alias-method/results/alias_linear.json` | 1.76 Gsamples/s |
| alias uniform | `parts/alias-method/results/alias_uniform.json` | **1.95 Gsamples/s** |
| uniform vs linear χ² | `parts/alias-method/experiments/uniform_vs_linear.json` | 247.67 vs 247.71 |
| `bw_copy` | `parts/cpu-baseline/results/bw_copy.json` | **305 GB/s** |
| cuRAND Philox host API | `parts/cpu-baseline/results/curand_philox.json` | 66.0 Gsamples/s, **264 GB/s** |

K7 ncu (`parts/endgame/results/nsight/ncu/philox.txt`), not vibes:

| Fact | Number |
|---|---|
| L1/TEX Cache Throughput | **96.14%** |
| Compute (SM) Throughput / Issue Slots Busy | **25.90%** |
| DRAM Throughput | 16.04% |
| Mem Pipes Busy | 6.82% |
| Top stall | LG throttle, 20.4 of **28.55** cycles (**71.3%**) |
| Bytes utilised per 32 B sector | **4.2** |
| L1 / L2 hit | 64.83% / 99.90% |
| Occupancy | 94.0% |
| Executed instructions at 1e6 samples | **13,281,292** |

### What the SASS actually emits

`cuobjdump -sass` on `cmake-build-release/.../philox_sample.cu.o`, `SamplePhiloxKernel`:

```
LDG.E.CONSTANT.SYS R25, [R10]          ; radial prob
LDG.E.CONSTANT.SYS R24, [R8]           ; theta  prob
@!P0 LDG.E.CONSTANT.SYS R23, [R10+0x4] ; radial alias
@!P1 LDG.E.CONSTANT.SYS R7,  [R8+0x4]  ; theta  alias
LDG.E.CONSTANT.SYS R23, [R28+0x8]      ; lo     (axis A)
LDG.E.CONSTANT.SYS R24, [R28+0xc]      ; width  (axis A)
LDG.E.CONSTANT.SYS R27, [R28+0x10]     ; y0     (axis A)
LDG.E.CONSTANT.SYS R0,  [R28+0x14]     ; y1     (axis A)
LDG.E.CONSTANT.SYS R19, [R6+0x8]       ; lo     (axis B)
LDG.E.CONSTANT.SYS R22, [R6+0xc]       ; width  (axis B)
LDG.E.CONSTANT.SYS R25, [R6+0x10]      ; y0     (axis B)
LDG.E.CONSTANT.SYS R26, [R6+0x14]      ; y1     (axis B)
@P0 STG.E.128.SYS [R2], R24            ; the 16 B output
```

**Twelve scalar 32-bit gathers per sample.** Not two 24-byte loads — twelve four-byte ones. That is
the whole story, and it is why ncu says 4.2 of 32 bytes per sector: a divergent `LDG.E` pulls a
32 B sector and consumes 4 B of it. `AliasBin` is `{float prob; int alias; float lo, width, y0, y1;}`
with **no `alignas`**, so its alignment is 4 and nvcc is not allowed to vector-load it — even though
`24 j` happens to be 8-byte aligned for every `j`. The struct's declared alignment, not its
addresses, is what the compiler is permitted to use.

### The L1TEX cycle model, calibrated on K7

For a fully divergent gather across 32 lanes into a ~98 KB table, no two lanes share a 128 B line,
so each load instruction costs **32 L1TEX wavefronts**. Per warp per sample:

- 10 unpredicated + 2 predicated (~0.35 taken) divergent gathers ≈ **10.7** effective
- 10.7 × 32 = **342** wavefronts
- one coalesced `STG.E.128`: 32 lanes × 16 B = 512 B = 4 lines = **4** wavefronts
- **≈ 346 L1TEX cycles per warp-sample**

Instruction side, from ncu: 13,281,292 executed instructions at 1e6 samples = 31,250 warps →
**425 warp-instructions per warp-sample**. Turing issues 4/cycle/SM → **106 SM-cycles**.

Predicted ratio 346 / 106 = **3.26**. ncu measured 96.14 / 25.90 = **3.71**. The model is within
13% of the report on the kernel it was built from, so I will use it forward. (The gap is the
uniform-address gathers inside the Laguerre/Legendre loops — trip count 0 for (1,0,0) — and
sectors shared by lucky lanes. Both make my 346 a slight under-count, which is the right direction
for the error.)

Two independent roofs fall out and both agree with the JSON:

- **Compute-issue roof**: 40 SM × 4 inst/cycle × 1.5 GHz ÷ (425/32 inst/sample) = **18.1 Gsamples/s**.
  Cross-check: 4.60 ÷ 0.2590 = 17.8. ✓
- **16 B DRAM roof**: 305 ÷ 16 = **19.06 Gsamples/s**.

### What K9 changes

Two arrays per axis, both naturally aligned:

```c++
struct alignas(8)  AliasDraw { float prob;  std::uint32_t alias; };   // LDG.E.64
struct alignas(8)  BinUniform{ float lo,    width; };                 // LDG.E.64
struct alignas(16) BinLinear { float lo, width, y0, y1; };            // LDG.E.128
```

The draw array is read once per axis and yields *both* fields, so the predicated `alias` load
disappears with it. The interior array is read once per axis, for the **winning** index only.

| | K7 (`AliasBin`) | K9 uniform | K9 linear |
|---|---|---|---|
| Divergent load instructions / sample | 10.7 | **4** | **4** |
| Logical table bytes / sample | 48 requested (4.2/32 used) | **32** | 48 |
| L1TEX wavefronts / warp-sample | 342 + 4 = **346** | 128 + 4 = **132** | 132 |
| Bytes used per 32 B sector | 4.2 | **8** | 16 |

L1TEX cycle ratio **346 / 132 = 2.62×**. Instruction side: uniform interior deletes two `sqrtf`,
two divides and the linear-interpolation FLOPs, plus eight loads and their address `IMAD`s —
call it 425 → ~370, so the compute-issue roof rises to 18.1 × 425/370 = **20.8 Gsamples/s**.

New binding pipe: L1TEX at 4.60 × 2.62 = **12.0 Gsamples/s**, with compute-issue at 20.8 and DRAM
at 19.06 both still above it. At 12.0 Gsamples/s the pipes would read: L1TEX ~96%, compute ~65%,
DRAM 192/356 = 54% of ncu peak. Three pipes in the 50–96% band is exactly the regime where a clean
ratio does not survive contact, and there is a second cost the model ignores: the gather chain is
**dependent** — `draw[j]` must return before `j_win` exists, and only then can `interior[j_win]`
issue. Two dependent L1 round trips per sample stay, at 2/3 the warps' worth of other work to hide
them. This repo's last two central numbers came in at 0.77× and 0.54×. I take a 0.87× haircut.

### Shared memory (secondary experiment)

Both draw arrays are 4095×8 + 2047×8 = **49,136 B**, one byte under Turing's 48 KB static limit.
Staging them at block start removes the two global gathers from the global path. It does not remove
them from **L1TEX** — on Turing shared memory and L1 are the same 96 KB SRAM behind the same
wavefront machinery, and a random 8 B `LDS` across 32 lanes serialises into bank-conflict phases
much as an `LDG` serialises into wavefronts. What it *does* remove is the tag lookup, and what it
costs is occupancy: 48 KB/block against 64 KB shared/SM is one block per SM, so 256-thread blocks
would run at 25% occupancy. Only a 1024-thread block gets the occupancy back. K3's retro already
warned that shared can lose on occupancy; this time it is 2 loads rather than a 23-step walk, so
the trade is closer, but the SRAM is the same SRAM.

### ILP (conditional experiment)

K8 measured S>1 as a loss while the stall was "LG queue full". If ncu says that queue drained, the
question is reopened and S ∈ {2,4} gets one sweep on the winner. If the LG throttle is still the
top stall, it is settled and I do not rerun it.

## Predicted number

- **K9 `packed`** — global, uniform interior, 8 B + 8 B, 256 threads, S=1: **10.5 Gsamples/s**
  (band 8.0–14.0). **2.3×** K7. 168 GB/s of physics: 55% of `bw_copy`, 64% of cuRAND's 264 GB/s.
- **K9 `packed_linear`** — global, 8 B draw + 16 B interior: **9.5 Gsamples/s** (band 7.0–13.0).
  Same four loads, wider sectors, two `sqrtf` back. The alias part measured uniform/linear at
  1.95/1.76 = 1.11× when the kernel was state-bound; expect about the same or a little more now
  that compute is nearer the roof.
- **K9 `packed_shared`** — draw arrays in 48 KB of shared, 1024-thread blocks: **8.5 Gsamples/s**
  (band 5.0–13.0). **I expect this to lose.** It is measured because K3's retro says measure it.
- **ILP S ∈ {2,4}**: noise (±5%), and only run if the LG throttle moved.
- **64M vs 8M**: same rate ±3%. Run both so the kernel duration is comparable to the 67M cuRAND
  microbench and the 8M series numbers stay comparable to K1–K8.
- **Per-orbital sweep**: (1,0,0) fastest; higher *n* and *l* pay only for the Laguerre/Legendre
  loop trip counts, which are compute, not L1TEX — expect ≤20% spread across the matrix, much
  flatter than the inverse-table part's spread.
- **(3,1) node-window χ²**: unchanged at ~30–37, limit ~83. The bins are the same bins.

## What would falsify this

- K9 lands at 4.6–5.5 Gsamples/s **and** ncu still reports L1/TEX ≥ 90% → the unit of the L1TEX
  cost is not the load instruction. Twelve scalar loads and four wide loads would then cost the
  same, and the whole 2.62× is imaginary.
- K9 ≥ 15 Gsamples/s with DRAM ≥ 75% → the tables were nearly free once vectorised and 10.5 is
  badly low; the 24-byte record was costing more than the wavefront model can account for.
- ncu Compute (SM) ≥ 80% at the measured rate → math, not L1TEX, became the roof. Then `packed`
  must beat `packed_linear` by **more** than 11%, and if it does not, the compute accounting is
  wrong too.
- `packed_shared` beats `packed` by >1.3× → the "shared and L1 are the same SRAM" argument is
  wrong and K3's occupancy warning does not generalise.
- Uncoalesced extra sectors stay ≥ 80%, or ncu still reports ~4 of 32 bytes utilised → the
  `alignas` did not produce `LDG.E.64`; check the SASS before believing any throughput number.
- (3,1) node-window χ² > 83 at 1e7 with uniform interior → the 8 B interior record is not enough
  resolution and `packed_linear` becomes the primary variant. The tables are not to be rebuilt;
  if χ² opens, the layout is right and the interior rule is wrong.
- Any two launch configs disagree bitwise at the same seed and sample index → the counter got
  tied to a thread id while adding the shared-memory block shape.
- Reconstructed masses from the split arrays differ from `ReconstructMasses` on the original
  `AliasBin` vector → the split forked the Vose tables. Do not fork the tables.
