# Experiment — JIT warmup (decay of the sum's time, CPU)

Goal: *show* the JIT warmup. When the **same** computation (a sum over a CPU vector) is called in
a loop, the first calls are slow and the time then **decays** towards a steady state. Two causes
overlap:

- **just-in-time compilation (JIT)**: the very first call of a JIT-compiled function pays for the
  compilation → a huge spike, isolated on call #0;
- **hardware warmup**: the CPU ramping up its frequency, the caches filling up → a smooth decay
  over the first calls, present in *every* language.

The experiment runs the same sum in several variants, times every call, and overlays the curves
to compare the numbers (the pedagogical engine of the course).

## Measured variants

| Source file | Variant           | Expected                                          |
|-------------|-------------------|---------------------------------------------------|
| `warmup.py` | `python_numba`    | `@njit`: **big spike** on the 1st call (JIT), then a fast plateau |
| `warmup.py` | `python_pure`     | pure Python loop: no JIT → **slow** plateau, no spike |
| `warmup.py` | `python_numpy`    | `numpy.sum`: already-compiled reference → **fast** plateau |
| `warmup.jl` | `julia_mysum`     | native loop compiled just in time: **spike** (JIT) on the 1st call, then a plateau (≈ `numba`) |
| `warmup.jl` | `julia_basesum`   | Base's `sum(x)`: **spike** (JIT) on the 1st call as well, then the fastest plateau |

The key contrast: `numba` (Python) and **both** Julia variants pay for the compilation on the
1st call (spike), whereas `numpy` (precompiled C) shows almost nothing and the **pure** Python
loop is flat **but high** (interpreted, never compiled — no Julia equivalent, and that is exactly
the point).

> **Julia lesson (visible on `basesum`)**: *everything* is JIT in Julia. Even Base's `sum` gets
> (re)compiled on the 1st call **for a given concrete type** if the specialization is not already
> in the system image → a spike, just like `mysum`. The steady states then differ (`sum` is faster
> than `mysum` thanks to pairwise summation / SIMD).

> **Measurement detail (Julia)**: capturing the 1st call's spike requires a trick — see the comment
> on `measure` in `warmup.jl`. Called directly, `fn(x)` would be compiled *before* the loop (while
> `measure` itself is being compiled) and the spike would escape the timer; we therefore route the
> call through an abstract `Function[fn]` container to defer the compilation to the 1st actual call.
> numba, for its part, compiles naturally on the 1st call.

## Running

```bash
cd teaching/parallel_computing/experiences/warmup_jit
python3 warmup.py        # -> results/warmup_python_{numba,pure,numpy}.csv
julia   warmup.jl        # -> results/warmup_julia_{mysum,basesum}.csv
python3 plot_warmup.py   # -> results/warmup.png (overlays every results/warmup_*.csv)
```

Every generated artifact (CSV + `warmup.png`) goes into the **`results/`** subfolder
(git-ignored). For the cumulative frames of the slides:

```bash
python3 plot_warmup.py --frames ../../images   # -> warmup_frame_0..5.png (committed)
```

Each CSV uses the common `iteration,time_s` format. The plot uses a log-y scale because the JIT
spike of the 1st call flattens a linear axis.

Parameters at the top of the scripts: `N = 1_000_000` (Float64 vector), `R = 2000` fast
iterations, `R_PURE = 200` for the pure Python loop (each call ~0.1 s). `N` is deliberately small
so that the compilation cost (~0.1–0.5 s) **dominates** the steady state (~1 ms) and makes the JIT
spike clearly visible.

## Coming up: the same experiment on GPU

On a GPU, timing from the host side (`time_ns` / `perf_counter`) mixes the GPU time with the
asynchronous kernel launches, the transfers and the synchronization → a lot of noise. We will
therefore measure the **GPU time alone** with a profiler / GPU events:

- **Julia**: `CUDA.@profile`, or `CUDA.@elapsed` (based on CUDA events), + Nsight Systems.
- **Python**: `torch.cuda.Event(enable_timing=True)` around the kernel (with `synchronize`),
  or `nsys profile` to keep only the kernel durations.

We will see the GPU warmup there (kernel compilation/loading, first allocation of the CUDA
context) with a cleaner signal. The CSVs produced will follow the same `iteration,time_s` format
and will be dropped here (`warmup_gpu_*.csv`); `plot_warmup.py` will overlay them automatically.
Connects to module 4 (memory & GPU) of the course.
