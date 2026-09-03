# Reading — CPU baseline

### Griffiths — Introduction to Quantum Mechanics, ch. 4
- Link / DOI: textbook
- What it originated: hydrogen ψ_nlm = R_nl(r) Y_lm(θ,φ); density factorization; closed-form ⟨r⟩, ⟨r²⟩, ⟨1/r⟩
- What I actually need from it: the three 1D sampling distributions and the moment formulas (a₀ = 1)
- Read before: this part

### Bethe & Salpeter (1957) — Quantum Mechanics of One- and Two-Electron Atoms
- Link / DOI: classic monograph
- What it originated: complete one-electron expectation-value tables
- What I actually need from it: quote ⟨r⟩ = (a₀/2)(3n² − l(l+1)), ⟨r²⟩ = (a₀² n²/2)(5n² + 1 − 3l(l+1)), ⟨1/r⟩ = 1/(n² a₀) — do not re-derive
- Read before: writing only

### DLMF §18.9 Laguerre recurrences
- Link: https://dlmf.nist.gov/18.9
- What it originated: (n+1) L_{n+1}^α(x) = (2n+α+1−x) L_n^α(x) − (n+α) L_{n−1}^α(x), with L_0^α = 1, L_1^α = 1+α−x
- What I actually need from it: the upward recurrence used in R_nl for n ≤ 6
- Read before: this part

### DLMF §14.10 associated Legendre recurrences
- Link: https://dlmf.nist.gov/14.10
- What it originated: stable upward recurrence in l from P_m^m
- What I actually need from it: P_m^m(x) = (−1)^m (2m−1)!! (1−x²)^{m/2} (Condon–Shortley), then the l-step used in Numerical Recipes `plgndr`
- Read before: this part

### Devroye (1986) — Non-Uniform Random Variate Generation, ch. II
- Link: https://luc.devroye.org/rnbookindex.html
- What it originated: inverse-transform sampling; piecewise-linear CDF inversion
- What I actually need from it: interpolate inside the hit bin instead of returning the bin edge
- Read before: this part

### Kolmogorov–Smirnov / Pearson χ² (Massey 1951 for the practical KS form)
- Link / DOI: standard
- What it originated: distribution tests used as regression alarms, not p-value theater
- What I actually need from it: at N = 10⁷, gate on D < 5×10⁻⁴ against a high-resolution FP64 radial CDF; χ² on bins finer than the table so quantization shows up
- Read before: this part

### Simon Boehm — How to Optimize a CUDA Matmul Kernel
- Link: https://siboehm.com/articles/22/CUDA-MMM
- What it originated: the worklog format this series copies (prediction → kernel → measure → retro)
- What I actually need from it: voice and structure, not GEMM
- Read before: writing only
