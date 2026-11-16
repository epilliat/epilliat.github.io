# Activities — predict, verify, measure, explain

See [the three-session plan](../sessions.qmd) for the exact 3 × 180-minute budget.
Activities take 20–30 minutes including prediction, coding, verification and interpretation.
Each one is launched and debriefed by a slide of its deck (A1–A6); files state the question, not the conclusion:
the explanation is produced by the students and collected in the debrief.
They are formative, not graded. A gain or a slowdown is acceptable when explained.

| Session | Core activities | Optional alternative |
|---|---|---|
| 1 | Python warmup (20 min), diagnostic (20 min) | Julia warmup; dispatch extension |
| 2 | Python I/O (20 min), CPU lots/reduction (30 min), race discussion | Julia concurrency parts A/B; A3d slow local server, Python or Julia (no Internet) |
| 3 | Python locality (20 min), batching (20 min), GPU/analysis (30 min) | Julia locality/GPU |

A3d's optional class π needs `registry.py` or `registry.jl` on the instructor's laptop (address on the board; open it in a browser to project names and points).
Python CPU work uses the importable `../python/parallel_workers.py`; download it alongside a standalone notebook.
I/O supports synthetic values with controlled waits, explicitly labelled. CPU work is independent of I/O.
A GPU is not guaranteed: use [the analysis activity](../gpu-fallback.qmd) without one.
For local Python prepare NumPy, Numba, matplotlib and threadpoolctl; GPU notebooks use PyTorch.
Record hardware, dtype, size, workers/BLAS threads, warmup, repetition and included costs.
Compare ratios on the same machine, not language rankings across computers.

## Distribution

Instructor `.jl`/`.ipynb` files include solutions between `SOLUTION` and `END` markers.
`build_activities.py` replaces these blocks with explicit errors for students to complete.
For a notebook it also writes `release/<name>_todo.py`, the same cells as `# %%` blocks, for
students who work locally (linked next to Colab on the slides and the course page).
Both sources and generated `release` files are committed. Rebuild after every source edit:

```bash
python3 teaching/parallel_computing/activities/build_activities.py teaching/parallel_computing/activities/3_cpu_python.ipynb
```

Publish release notebooks and the worker module with the course page. Instructor solutions stay outside
the student activity links. There is no grading framework or additional abstraction to learn.
