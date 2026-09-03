# Part 1 — The problem, the lab, and the CPU reference

I am sampling the hydrogen orbital density |ψ_nlm|². That is the simulation: a stream of points drawn from the Born rule, not a picture. The visualizer this started from lived in OpenGL and never saw a GPU.

ψ_nlm(r,θ,φ) = R_nl(r) Y_lm(θ,φ). The density factorizes, so the draw splits into three independent 1D problems:

- P(r) ∝ r² R_nl(r)²
- P(θ) ∝ sinθ |P_l^m(cosθ)|²
- P(φ) uniform, because |e^{imφ}|² = 1

That factorization is why this drop needs no Markov chain. Helium will destroy it.

R_nl and P_l^m come from the DLMF upward recurrences (Laguerre §18.9, associated Legendre §14.10). The CPU reference is FP64 on purpose. It is ground truth, not a contestant.

## Tests before trust

I ported the old sampler as a test double and pointed the correctness kit at it. Three defects, three named tests:

1. **Stale CDFs.** `static` tables built for the first (n,l,m) and never rebuilt. KS on a second orbital: D = 0.15, rejected.
2. **Bin-edge quantization.** Returns `idx * dr`. 100k draws sat on 2191 discrete radii (table has 4096). The corrected inverse-CDF interpolates; 100k draws → 100k unique radii.
3. **Negative m.** The Legendre seed loop ran only for `m > 0`. KS on (3,1,−1): D = 0.054, rejected. Density depends on |m|; sample with |m|.

The corrected reference rebuilds per orbital, inverts a piecewise-linear CDF (Devroye ch. II), and uses |m|. Goldens for R_nl and P_l^m (scipy grid, mpmath closed forms) match to 1e-12. At N = 10⁷, six orbitals including (3,1,−1) and (5,0,0): KS D_r ≈ 2.8×10⁻⁴ (gate 5×10⁻⁴), moments ⟨r⟩, ⟨r²⟩, ⟨1/r⟩ inside 5σ of Griffiths / Bethe & Salpeter.

## CPU baseline

I predicted 4.0 Msamples/s on one i7-10875H thread and 22 Msamples/s with OpenMP. I got **2.12 Msamples/s** single-thread and **15.1 Msamples/s** on all 16 threads (Release, 8×10⁶ samples of 1s, median of 20). Factor-of-two miss on the serial number: libm plus the 23-step search cost more than the 400–700 cycle sketch. Scaling was 7.1× — not the thing that was wrong.

Format of this series is borrowed from Boehm's CUDA matmul worklog: predict, measure, admit the miss.

## The gap

Same machine, graphics clock locked at 1500 MHz:

| Referee | Measured |
|---|---|
| CPU single-thread | 2.12 Msamples/s |
| CPU OpenMP (16 threads) | 15.1 Msamples/s |
| Device copy bandwidth | 305 GB/s |
| 16 B/sample roof | 19.1 Gsamples/s |
| cuRAND Philox uniforms | 66.0 Gsamples/s |

cuRAND emits about 3×10⁴ more raw uniforms per second than the single-thread loop consumes. The gap is the series.
