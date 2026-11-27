# Activities — predict, verify, measure, explain

See [the three-session plan](../sessions.qmd) for the exact 3 × 180-minute budget.
Activities take 20–30 minutes including prediction, coding, verification and interpretation.
Each one is launched and debriefed by a slide of its deck; files state the question, not the conclusion:
the explanation is produced by the students and collected in the debrief.
They are formative, not graded. A gain or a slowdown is acceptable when explained.

| Session | Core activities | Optional alternative |
|---|---|---|
| 1 | Python warmup (20 min), diagnostic (20 min) | Julia warmup; dispatch extension |
| 2 | Python I/O (20 min), CPU lots/reduction (30 min), race discussion | Julia concurrency parts A/B; a slow local server, Python or Julia (no Internet) |
| 3 | Python locality (20 min), batching (20 min), GPU/analysis (30 min) | Julia locality/GPU |

The slow-server activity's optional class π: the instructor runs the last cell (`start_registry()`) and opens http://localhost:9000/ to project the board (the address students type, names, points, π). A student who runs it gets a board of their own.
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

## Optional final activity — Qwen on GPU

`6_llm_batching_python.ipynb` is the instructor version; distribute the generated notebook/script
in `release/`. Allow 30 minutes **after the core course, only if time remains**. Prepare
`transformers>=4.56,<5`, matplotlib, CUDA PyTorch and the cached `Qwen/Qwen2.5-0.5B-Instruct`
weights in advance. Run the complete instructor notebook on the classroom GPU before teaching.
The instructor notebook keeps the outputs of a run on an RTX 1000 Ada laptop GPU as reference timings.

Students first edit ordinary strings through `answer` and `answer_batch`. Those cells work
before the single exercise (two metric formulas). Benchmarking uses repeated equal prompts,
fixed output length, greedy decoding, resident inputs, synchronized warm measurements and medians.
Compare B=1…128, then equal total work (successive requests versus a batch). The model is fully on
GPU in FP16; no automatic CPU offload or quantization. On OOM the sweep stops at the last working
size. If B=1 fails, restart on a suitable GPU. Without a GPU, interpret another group's recorded
measurements and name their source; the activity has no invented GPU results.

Debrief: T/B is amortized time, not user latency; the sequential comparison includes the waiting
of requests that arrived together. A batch can reuse weights and improve throughput while each
response still depends on its previous tokens. The benchmark includes prefill and decoding, not
first-token latency. Do not infer utilization or a specific bottleneck from timing alone.
For a short 15-minute demonstration, run the prepared cells, give the metric formulas and keep
only the prediction and final discussion. No GUI/server setup is needed.
