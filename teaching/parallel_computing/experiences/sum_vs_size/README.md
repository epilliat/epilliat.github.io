# Experiment — Sum vs size (the staircase of Base's `sum`)

Goal: measure the time of a CPU sum **as a function of the vector's size N**, in the steady state
(compilation done once and for all — see [`../warmup_jit/`](../warmup_jit/)), for two Julia
variants:

- **`mysum`**: scalar native loop → **~ linear and smooth** time in N;
- **Base's `sum`**: **pairwise summation + SIMD** → much faster, and reveals a slight
  **staircase** (the steps of the recursion / of the SIMD blocking). Anecdotal, but pretty.

For each N we take **REPS = 10** measurements → **mean ± standard deviation** (the CI drawn on
the curves).

## Measured variants

| Source file  | Variant         | Expected                                            |
|--------------|-----------------|-----------------------------------------------------|
| `scaling.jl` | `julia_mysum`   | smooth line, slope ≈ cost per element (scalar)      |
| `scaling.jl` | `julia_basesum` | lower, **staircase** (SIMD / pairwise steps)        |

## Running

```bash
cd teaching/parallel_computing/experiences/sum_vs_size
julia   scaling.jl       # -> results/scaling_julia_{mysum,basesum}.csv
python3 plot_scaling.py  # -> results/scaling.png (means + ±std bands)
```

Cumulative frames for the progressive reveal of the slides (mysum → base → CI):

```bash
python3 plot_scaling.py --frames ../../images   # -> scaling_frame_0..3.png (committed)
```

Every generated artifact (CSV + `scaling.png`) goes into **`results/`** (git-ignored). Common CSV
format `n,mean_s,std_s`. **Linear** axes: that is what makes Base's staircase visible.

## Why Base's `sum` beats `mysum` (code reflection)

```bash
julia reflection.jl   # -> results/{mysum,sum,mapreduce_impl}.asm + @simd benchmark
```

Dumps the native code (Intel asm) and times `mysum` / `mysum_simd` / `sum`. We see that `mysum`
stays **scalar** (`vaddsd`, a single `xmm0` accumulator) whereas the `mapreduce_impl` kernel of
`sum` is **vectorized** (`vaddpd` on `ymm0..3` = 4-wide SIMD × 4 accumulators). Adding `@simd` to
`mysum` vectorizes it in turn (~10× faster). This is the source of the asm excerpts of chapter 1
(`slides/1_compilation_and_types.qmd`).

## Parameters (at the top of `scaling.jl`)

`N_START = 10_000`, `N_STOP = 100_000`, `N_STEP = 100` (901 points), `REPS = 10`. The compilation
is taken out of the measurement by a warmup call before the loop; each `randn(n)` vector serves
both variants (fair comparison).
