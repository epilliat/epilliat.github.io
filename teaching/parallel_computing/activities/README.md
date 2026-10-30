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
- The **student skeletons** land in `activities/release/` — committed and published (see *Distribution*).

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
  ~10× overlap is felt, not simulated.
- **A one-line bridge to the other language** ("in Python threads can't do this — the GIL; Julia has
  real threads").

## Module 1 — the worked example

`1_warmup_python.ipynb` (2 holes, ~8 min):
1. write the pure-Python sum loop → `numpy` is ~70× faster (the interpreter tax);
2. write the same loop `@njit`-compiled → ~30× faster than your pure loop (Julia's trick on Python).

`1_warmup_julia.jl` (1 hole, ~4 min) — sums the integers `1:N`, N = 10⁷ (so a wrong answer is
obvious): write `mysum` → your compiled loop ≈ Base `sum` (in Python the same loop is ~70-100× slower
than numpy). No threads here: going parallel is Part B of the concurrency activity.

## The full set

Each is verified: the solution prints the payoff, the skeleton stops at the hole. The numbers below
were measured on this machine; expect different values on another one — compare the ratios.

| module | file | hole(s) | the aha |
|---|---|---|---|
| compilation | `1_warmup_python.ipynb` | pure loop, numba | numpy ~70×, numba ~30× — the interpreter tax, then removed |
| compilation | `1_warmup_julia.jl` | `mysum` | your compiled loop ≈ Base `sum` |
| dispatch | `2_dispatch_julia.jl` | the two `area` methods | concrete `Vector{Circle}` **~10×** faster than abstract `Vector{Shape}` |
| threads & concurrency | `3_concurrency_julia.jl` | A: the `@async` line · B: the `@spawn` split | A: 20 daily Binance closes → **~10×** overlap, then plots the returns · B: π on a shared counter lies (race); `@spawn` + `fetch` is right and several × faster; the *same* code with `@async` stays ~1× |
| threads & concurrency | `3_concurrency_python.ipynb` | the `ThreadPoolExecutor` map | download 20 real Binance daily files → **~10-20×** overlap, then plots the returns |
| memory & locality | `4_locality_julia.jl` | the loop order | down columns faster than along rows (column-major): **~12×** at 8000² (7.2× at 4000²) |
| memory & locality | `4_locality_python.ipynb` | the loop order | with the grain several × faster at 4000² (numba, row-major → *opposite* direction); the ratio is printed |
| GPU | `5_gpu_python.ipynb` | draw the points on the GPU | π: at 10 k points GPU **0.6× (loses!)**, at 1 M **~90×**, at 50 M **~40×** — feed it |
| GPU | `5_gpu_julia.jl` | `CUDA.rand` instead of `rand` | same π (Float32 on both sides): small sizes lose, millions of points win by tens of ×; the *same* `count_in_disk(x, y)` runs on the card by multiple dispatch |

Notes on language coverage:

- **The dispatch activity is Julia-only.** Its aha is a *measurement* of dispatch specialization;
  Python always resolves methods at run time, so the concrete-vs-abstract contrast has no Python
  twin — that's the module's whole point.
- **The concurrency activity's Part B is Julia-only.** Python threads cannot run CPU work in
  parallel (the GIL); the Python twin keeps to I/O, where the GIL is released.
- **The locality activity is bilingual, and the twin makes the lesson sharper.** Plain `np.sum`
  *hides* locality (it reorders iteration into memory order), but a **numba** hand-loop does not —
  so it shows the same effect as Julia, in the **opposite** direction (numpy is row-major, Julia
  column-major). "A library can hide the penalty; it cannot remove it."
- **The GPU activity needs hardware.** The Python track is the common one: it runs on Colab's **free
  GPU** (*Runtime ▸ Change runtime type ▸ GPU*) — an "Open in Colab" link on the notebook (hosted on
  GitHub) opens it in one click. The Julia track needs a **local CUDA machine**, so it's the deep-dive.
  Neither *activity* writes a kernel — `torch` / `CUDA.jl` run high-level array code on the card;
  the coalescing test in the GPU module shows a hand-written `@cuda` kernel.
- **The GPU activity closes the running example:** the scripts count darts in the disk
  (`count_in_disk`) sequentially (Julia basics, compilation), then on threads (the threads module
  and Part B of the concurrency activity), then on the GPU.

### Distribution

The **student skeletons** (`release/*_todo.*`) are **committed and published**; the instructor
sources (`N_*.ipynb` / `.jl`, with the answers) are committed but **never linked**, so Quarto never
copies them to `docs/`. The course page (`index.qmd`, *Activities* tab) links each skeleton, plus an
**Open in Colab** link for the Python ones (Colab reads the `docs/…/activities/release/` copy on
GitHub `master`, so it works once pushed). **Colab matters only for the GPU activity**; the CPU
activities need nothing but a local Python or Julia.

Flow after editing an activity:

1. edit the source `activities/N_*.{ipynb,jl}` (answer inside the markers);
2. rebuild its skeleton: `python3 activities/build_activities.py activities/N_*.{ipynb,jl}`;
3. `quarto render index.qmd` (copies the skeletons into `docs/teaching/parallel_computing/activities/release/`);
4. commit source + skeleton + `docs/` together. **Never link a source file** from a page — it
   would publish the answers.
