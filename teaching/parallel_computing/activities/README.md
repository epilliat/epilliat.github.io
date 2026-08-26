# Activities — short, targeted, ungraded

The goal is **not** to test students — it's to hand them one striking number with the least possible
friction. Each activity is **~5 minutes**: fill **one** blank, run everything, read the payoff. No
grade, no setup to fight, no long worksheet. Encouragement, not assessment.

Each module has (or will have) a matched pair — **Python and Julia** — so a student runs whichever they
like, or both, and compares the numbers. That comparison is the whole pedagogical engine of the course.

## How it works

One file per activity, written **with** its solution, wrapped in a marker. `build_activities.py`
strips the answer to a friendly stub. Single source per activity — no drift.

```bash
cd teaching/parallel_computing
python3 activities/build_activities.py activities/1_warmup_python.ipynb
python3 activities/build_activities.py activities/1_warmup_julia.jl
#   -> activities/release/1_warmup_python_todo.ipynb
#   -> activities/release/1_warmup_julia_todo.jl
```

- The **instructor files** (`activities/N_*.ipynb` / `.jl`) are committed — they hold the answer.
- The **student skeletons** land in `activities/release/` — **gitignored**, regenerate anytime.
  Distribute by uploading the `_python` one to Colab and handing out the `_julia` one for VS Code.

## The marker

Wrap only the answer. Everything outside stays verbatim — the setup and the timing/print that reveals
the payoff. Put the **hint** in the marker; it becomes the student's `# ✏️ your turn:` line.

Python / Jupyter code cell:

```python
def mysum(x):
    ## SOLUTION: add up the elements of x with a for loop ##
    s = 0.0
    for xi in x:
        s += xi
    return s
    ## END ##
```

Julia `#%%` script:

```julia
function mysum(x)
    #= SOLUTION: add up the elements of x with a for loop =#
    s = zero(eltype(x))
    for i in eachindex(x); s += x[i]; end
    return s
    #= END =#
end
```

The skeleton replaces the block with the hint + a loud stub (`raise NotImplementedError` /
`error(...)`) so the notebook stops **exactly at the blank** — the student sees precisely where to type.

## Design rules (keep them tiny)

- **Few holes** — one or a small handful, each carrying one "aha". Never a quiz.
- **Scaffold does everything else** — imports, data, timing, the print that shows the number. The
  student never wrestles with setup. A deliberately-broken *given* demo (e.g. the race) is run, not
  written — you don't ask a student to type a bug.
- **Immediate, striking payoff.** The cell right after each hole prints the factor (`~70× faster`, a
  race that lies, a GPU that overtakes). That number is the point. When it can be **concrete**, make
  it so — the concurrency activity downloads *real* Binance price files and plots the returns; the
  ~20× overlap is felt, not simulated.
- **A one-line bridge to the other language** ("in Python threads can't do this — the GIL; Julia has
  real threads").

## Module 1 — the worked example

`1_warmup_python.ipynb` (2 holes, ~8 min):
1. write the pure-Python sum loop → `numpy` is ~70× faster (the interpreter tax);
2. write the same loop `@njit`-compiled → ~30× faster than your pure loop (Julia's trick on Python).

`1_warmup_julia.jl` (3 holes, ~12 min) — sums the integers `1:N` (so a wrong answer is obvious):
1. write `mysum` → your compiled loop ≈ Base `sum` (in Python the same loop is ~100× slower);
2. **Part 2 — threads.** A *given* wild `@threads` on a shared counter loses ~95 % of the sum and
   changes every run (a race); then you write the `@async` version — correct but **no speedup**
   (concurrency doesn't help pure computation); then the `@spawn` version — correct and **~4× faster**
   (real parallelism). Same split, one word changed, opposite result.

The Julia activity spans the course arc (compiled loop → the concurrency-vs-parallelism distinction)
on one tiny example; the Python one stays on the compilation half, because Python threads can't
parallelize CPU work anyway.

## The full set

Each is verified: the solution prints the payoff, the skeleton stops at the hole. All numbers below
were measured on this machine (loaded — expect more on an idle one).

| module | file | hole(s) | the aha |
|---|---|---|---|
| 1 compilation | `1_warmup_python.ipynb` | pure loop, numba | numpy ~70×, numba ~30× — the interpreter tax, then removed |
| 1 compilation | `1_warmup_julia.jl` | mysum, @async, @spawn | compiled ≈ Base; race lies (~95 % lost); @async ~1×, @spawn ~4× |
| 2 dispatch | `2_dispatch_julia.jl` | the two `area` methods | concrete `Vector{Circle}` **13×** faster than abstract `Vector{Shape}` |
| 3 concurrency | `3_concurrency_julia.jl` | the `@async` line | download 20 daily Binance closes → **~10×** overlap, then plots the returns |
| 3 concurrency | `3_concurrency_python.ipynb` | the `ThreadPoolExecutor` map | download 20 real Binance daily files → **~20×** overlap, then plots the returns |
| 4 locality | `4_locality_julia.jl` | the loop order | down columns **12×** faster than along rows (column-major) |
| 4 locality | `4_locality_python.ipynb` | the loop order | with the grain **12×** faster (numba, row-major → *opposite* direction) |
| 5 GPU | `5_gpu_python.ipynb` | draw the points on the GPU | π: at 10 k points GPU **0.6× (loses!)**, at 1 M **~90×**, at 50 M **~40×** — feed it |
| 5 GPU | `5_gpu_julia.jl` | `CUDA.rand` instead of `rand` | same π: 10 k → **0.4×**, 1 M → **65×**, 50 M → **83×**; the *same* `count(...)` runs on the card by multiple dispatch |

Notes on language coverage:

- **Module 2 is Julia-only.** Its aha is a *measurement* of dispatch specialization; Python always
  resolves methods at run time, so the concrete-vs-abstract contrast has no Python twin — that's the
  module's whole point.
- **Module 4 is bilingual, and the twin makes the lesson sharper.** Plain `np.sum` *hides* locality
  (it reorders iteration into memory order), but a **numba** hand-loop does not — so it shows the same
  ~12× as Julia, in the **opposite** direction (numpy is row-major, Julia column-major). "A library
  can hide the penalty; it cannot remove it."
- **Module 5 (GPU) needs hardware.** The Python track is the common one: it runs on Colab's **free
  GPU** (*Runtime ▸ Change runtime type ▸ GPU*) — an "Open in Colab" link on the notebook (hosted on
  GitHub) opens it in one click. The Julia track needs a **local CUDA machine**, so it's the deep-dive.
  Neither writes a kernel — `torch` / `CUDA.jl` run high-level array code on the card. **A custom
  kernel (`@cuda` / raw CUDA) is a separate bonus module (6).**
- **The GPU activity closes the red thread:** π went sequential (module 1) → threads (module 3) → GPU
  (module 5), the *same* Monte-Carlo estimate carried onto three kinds of hardware.

### Distribution

Students **download the notebook/script from the course site** and run it locally (Jupyter / VS
Code) — the site links every one. **Colab matters only for module 5's GPU** (most laptops have no
NVIDIA card); the CPU modules 1–4 need nothing but a local Python or Julia.
