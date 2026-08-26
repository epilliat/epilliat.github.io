# Experiment — Thread scaling (the honest Amdahl curve)

Goal: measure the **speedup of Monte-Carlo π as a function of the number of threads**, and put the
course's central distinction on the same plot:

- **`Threads.@spawn`** (return + reduce) → real **parallelism**: the curve climbs, then **bends away
  from the diagonal** — that's **Amdahl**;
- **`@sync`/`@async`** → **concurrency only**: same syntax, same work, and the curve stays **flat at
  ~1×**. There is nothing to interleave in CPU work; nobody is waiting.

That second curve is the point of the threads module, *measured* instead of defined. It is also
question **Q1** of the module — students bet before they look.

## ⚠️ Why one *process* per point

The tempting version is a single session started with `--threads=auto`, looping over
`ntasks = 1, 2, 4, 8…`. **Don't.** It measures **placement luck**, not scaling.

On a modern hybrid CPU (P-cores at ~5 GHz, LP-E cores at ~2.5 GHz), a handful of tasks may land on
fast cores or on slow ones. In one such session we measured **1.3× at 4 tasks but 4.7× at 6** — an
artefact of *where* the OS put the tasks, not of how the algorithm scales.

Launching `julia --threads=k` once per point gives the scheduler **exactly k threads**, which makes
the points comparable. That's what `run.sh` does.

## Measured variants

| Source file | Variant | Expected |
|---|---|---|
| `scaling_threads.jl` | `seq` | the baseline — one plain sequential loop |
| `scaling_threads.jl` | `spawn` | climbs, then flattens (Amdahl: spawn/fetch/reduce stay sequential) |
| `scaling_threads.jl` | `async` | **~1× whatever k is** — concurrency ≠ parallelism |

Each point: `REPS = 5` measurements → mean ± standard deviation (the error bars).
`π ≈ …` is printed for every run: `spawn` and `async` must agree with `seq` — no race here, the
counts are returned and reduced, never shared.

## Running

```bash
cd teaching/parallel_computing/experiences/threads_scaling
bash run.sh                 # k = 1..nproc (capped at 16), one process per point
bash run.sh 1 2 4 8 16      # or an explicit list
python3 plot_threads.py     # -> results/threads_scaling.png
```

`run.sh` removes `results/threads_scaling.csv` first, so each run starts clean.

> **Measure on an idle machine.** These timings are bandwidth- and scheduler-sensitive; a busy
> machine flattens the `spawn` curve and makes the whole thing look worse than it is. Check
> `uptime` first.

## Reading the result

The gap between the dashed diagonal and the `spawn` curve *is* Amdahl's law: whatever stays
sequential (splitting, spawning, `fetch`, the final reduce — plus the tasks that finish early and
then idle) caps the gain. In the notebook the same thing shows up as the profiler's
`Utilization: ~60%` line.

Related: [`../sum_vs_size/`](../sum_vs_size/) (single-core scaling),
[`../warmup_jit/`](../warmup_jit/) (the JIT spike).
