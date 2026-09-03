# Original visualizer sampler — bugs the tests must catch

The OpenGL app in Atoms-main is not the reference. These defects live in
`src/atom_realtime.cpp` (copied in `src/atom_raytracer.cpp`). The CPU ground
truth is `cpu-reference/hydrogen.*`. `cpu-reference/original_buggy.*` is only
a test double so the suite goes red on the old code and green on the fix.

1. Stale CDFs — `static` tables built for the first `(n,l,m)` and never rebuilt.
2. Bin-edge quantization — returns `idx * dr` with no interpolation inside the bin.
3. Negative m — associated Legendre seed loop only runs for `m > 0`. Density depends on `|m|`.

Fixes in the reference: rebuild per orbital, piecewise-linear inverse CDF, sample with `|m|`, FP64, weight `w = |ψ|²`.
