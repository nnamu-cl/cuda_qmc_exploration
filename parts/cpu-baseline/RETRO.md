# Retro — CPU hydrogen sampler

Predicted: 4.0 Msamples/s single-thread, 22 Msamples/s OpenMP
Measured: 2.12 Msamples/s single-thread, 15.1 Msamples/s OpenMP (16 threads)
Ratio: 0.53 single, 0.69 OpenMP

Release, i7-10875H, 8e6 samples of (1,0,0), SM graphics clock 1500 MHz. JSON in `results/`.

## Where the model was right
- Binding constraint is serial CPU work, not DRAM. 32 B/sample at 2.12 Msamples/s is 68 MB/s.
- OpenMP scales: 15.1 / 2.12 = 7.1× on 8 cores / 16 threads. Above the 4× “something is broken” line.
- Ballpark was the right order (a few million/s, not tens).

## Where the model was wrong
I priced the sample at 400–700 cycles and assumed a 4.8 GHz busy core. 2.12 Msamples/s is ~2.3e9 / 2.12e6 ≈ 1100 cycles at base clock, or ~2300 cycles if turbo stuck near 4.8 GHz. Either libm (`exp`/`pow`/`sin`/`cos` for the weight and Cartesian map) is heavier than the 150–300 cycle guess, or the core is not at 4.8 GHz for a 4 s loop. The OpenMP miss is mostly the same per-thread overestimate; scaling itself was fine.

No Nsight on this part — there is no kernel.

## GPU referees (same clock policy, quote these later)
- `bw_copy` 16,777,216 float4: **305 GB/s** → 16 B/sample roof **19.1 Gsamples/s**
- `fma32`: **3.75 TFLOPS**
- `fma64`: **88.2 GFLOPS**
- cuRAND Philox: **66.0 Gsamples/s** (264 GB/s at 4 B)
- cuRAND XORWOW: **67.1 Gsamples/s** (269 GB/s)

cuRAND Philox / CPU single-thread ≈ 3.1×10⁴. The series gap is real; the 10⁵ slogan was a bit high.

## What this changes going into the next part
Use 305 GB/s and 66 Gsamples/s as the measured roofs, not the 448 GB/s spec. Naive CUDA should beat 15 Msamples/s without trying; the interesting question is how close K1 gets to the FP64 pipe (88 GFLOPS) vs the bandwidth roof.
