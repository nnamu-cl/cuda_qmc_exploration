# Retro — packed alias records

Predicted: 10.5 Gsamples/s packed, 9.5 packed_linear, 8.5 packed_shared ("I expect this to lose"), ILP noise, 64M ±3%
Measured: **10.39** packed (128), 9.15 packed_linear (128), **14.60** packed_shared (512), 9.45 ILP S=2, 64M **−42%**
Ratio: 0.99×, 0.96×, **1.72×**, —, falsified

Release, RTX 2070 Super Max-Q, SM 1500 MHz ceiling, 8×10⁶ samples of (1,0,0), median of 20 after
3 warmups, **one process per cell** (see The instrument, below). Setup 0 ms. JSON in `results/`.

## The verdict

**packed_shared, 512 threads: 14.60 Gsamples/s, 233.6 GB/s of physics.** That is 77% of the 16 B
output roof (19.06) and **88% of cuRAND Philox's 264 GB/s** raw-bit emission rate, measured on the
same box in the same session (`results/philox_64m_referee.json` and the cpu-baseline microbenches).
K7 was 4.60. The 24-byte record was worth 3.2× — 2.26× from splitting it, another 1.41× from
putting the draw arrays in shared memory. The series headline is earned at burst; the power
governor owns everything past two milliseconds (below).

## Where the model was right

- **The wavefront model priced the global path to 1%.** Predicted 10.5 from 346→132 L1TEX
  wavefronts; measured 10.39. ncu on `packed`: L1TEX **98.5%**, LG throttle still the top stall at
  **44.7%** of a 14.8-cycle issue gap. Four aligned vector loads instead of twelve scalar gathers,
  and the pipe is still the pipe — exactly the "L1TEX stays the roof at about a third of the
  cycles" claim.
- **Uniform beats linear by 1.14×** (10.39 / 9.15). Predicted "about the same or a little more
  than 1.11×."
- **ILP still loses** (9.45 at S=2, 9.43 at S=4). The hypothesis said rerun only if the LG
  throttle drained; on the global path it did not drain.
- **Orbital spread ≤20%.** (1,0,0) 14.71 → (5,0,0) 12.13, a 17% spread across the verify matrix
  (`results/orbital_sweep.json`). The recurrence trip counts are compute, and compute is not the roof.
- **Correctness moved nowhere.** Node-window χ² 29.7–33.1 (limit ~83), bitwise determinism across
  128/256/512/1024 and shared/global, moments and KS pass on the full matrix. Same bins, same
  counters, verify-full green before any number in this file.

## Where the model was wrong

**Shared memory won by 1.72×, past its own 1.3× falsification line.** The "shared and L1 are the
same SRAM" argument was wrong in the way the falsification clause anticipated: the SRAM is shared,
the *access machinery* is not. A random 8 B `LDS` across 32 lanes costs bank-conflict phases —
ncu counts **68%** excess shared wavefronts — but it skips the tag stage and never touches the
LG queue, and the difference shows up as Warp Cycles Per Issued Instruction falling from **14.76**
(packed) to **6.27** (packed_shared). The occupancy tax I priced (49.6% achieved, one 512-thread
block per SM behind 48 KB of tables) is real and irrelevant: IPC 2.53 with three pipes in the
50–95% band (L1TEX 94.7%, compute 62.4%, DRAM 54.5%). K3's "shared loses on occupancy" lesson
does not generalise from a 23-step dependent walk to two independent draws. The remaining global
uncoalesced sectors (58%) are the interior-array gathers; they are the next target if there ever
is one.

**"64M vs 8M: same rate ±3%" was falsified at −42%, and the investigation ate the day.** The
chain, because the *why* matters more than the number:

1. `experiments/alloc_churn.json`: an identical 8M cell repeated in-process decays 10.40 → ~6.8
   and never recovers; scratch reuse changes nothing; every fresh process reads 10.40.
2. `experiments/footprint_ladder.json`: packed_shared is flat at ~14.8–15.0 to 24M samples, then
   falls to 8.7 at 64M. Looked exactly like a TLB-reach cliff at ~400 MiB.
3. But cuRAND is **flat at 272 GB/s writing 1 GiB** — a pure store stream doesn't pay the
   "cliff", which killed the TLB story.
4. The rep arrays killed the footprint story too: the first 64M rep runs **4.22 ms = 15.2
   Gsamples/s — full speed** — and rep 20 takes 10.2 ms. The slowdown is *progressive within the
   run*, not a property of the buffer.
5. ncu is immune (8M and 64M identical per-cycle: same SOL, same IPC, same occupancy) — because
   ncu runs at base clocks with replay gaps.
6. nvidia-smi sampled mid-rep: **750 MHz SM, 95.6 W, throttle reason 0x4 (SW_POWER_CAP)**.

This kernel pulls ~95 W at 1500 MHz. The 80 W governor averages over a short window: reps under
~1.6 ms (n ≤ 24M) with normal inter-rep gaps never trip it; longer reps back-to-back do, and
clocks bounce between 750 and 1500. An 8M cell launched *immediately* after a hot 64M run reads
14.79 — the ~2 s of process setup drains the window, which is why "fresh processes are always
fast" looked like process magic in step 1. `nvidia-smi -lgc 1500` is a ceiling, not a floor. Even
cuRAND decays once its reps are long enough (3.87 → 4.73 ms across 20 reps at 1 GiB); its median
just hides it.

So the honest reading: **14.60 Gsamples/s is the locked-clock burst rate**, comparable to every
number in this series (K1–K8 all ran ≤ 14 ms bursts at 8M); sustained production throughput on
this 80 W card is a different quantity, roughly 8.5–9 Gsamples/s, bounded by watts rather than by
any pipe ncu can see. That is a mobile-card fact, not a kernel fact — and it is the single best
argument in the series for the H100 session.

## The instrument

Every published cell runs in its own process, one kernel per invocation, n ≤ 8M for series
numbers — which is how K1–K8 were measured, so the series is internally comparable. In-process
multi-cell sweeps under-read later cells by up to 35% on this box (the governor, not the code);
`experiments/layout_sweep.json` is kept as the cautionary artifact, and the per-process cells in
`results/` supersede it. Retroactive caveat: K8's in-process launch sweep and K3's occupancy sweep
ranked short cells back-to-back; their orderings survive spot-checks per-process, but their
absolute later-cell numbers should not be quoted.

## What this changes going into the next part

The generation kernel is done: 77% of the output roof, 88% of cuRAND's payload bandwidth, gated
next by watts. Do not chase the last 23% — the remaining global interior gathers are worth maybe
10% and the governor takes it back at steady state. Drop 1 reports. Drop 2's walkers change the
memory story completely, and the power-governor instrument rules carry forward as-is.
