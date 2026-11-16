# Local version of 3_cpu_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # A3c — CPU work: split, compute, reduce — 30 minutes
# Needs `parallel_workers.py` next to the notebook (the setup cell downloads it if missing).

# %%
# Provided setup: workers must be importable by a fresh Python process.
from pathlib import Path
import sys, urllib.request
candidates = [Path.cwd(), *list(Path.cwd().parents)[:4]]
worker_dir = next((p / suffix for p in candidates
                   for suffix in ("", "python", "teaching/parallel_computing/python")
                   if (p / suffix / "parallel_workers.py").is_file()), None)
if worker_dir is None:  # a standalone notebook, e.g. Colab
    url = "https://raw.githubusercontent.com/epilliat/epilliat.github.io/master/docs/teaching/parallel_computing/python/parallel_workers.py"
    urllib.request.urlretrieve(url, "parallel_workers.py")
    worker_dir = Path.cwd()
sys.path.insert(0, str(worker_dir))
from parallel_workers import cpu_load, count_in_disk, make_jobs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time, platform
print(platform.python_version(), "start method: spawn; timings include pool startup")

# %%
jobs = make_jobs(1_000_003, lots=8)
assert sum(n for n, seed in jobs) == 1_000_003
start = time.perf_counter()
expected = list(map(count_in_disk, jobs))
t_seq = time.perf_counter() - start

# %%
# One lot: (size, seed), and its count of points in the disk
jobs[0], count_in_disk(jobs[0])

# %%
# The same lot, computed by ANOTHER process
with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
    future = pool.submit(count_in_disk, jobs[0])
    print(future.result())

# %% [markdown]
# Your turn: submit the lots to processes, sum the counts.

# %%
def parallel_hits(jobs, workers):
    # ✏️ your turn: use ProcessPoolExecutor with spawn, map count_in_disk over jobs, sum returned counts
    raise NotImplementedError("remove this line and write your code")

# %%
# Try it; then change 2 to 1, 4…
start = time.perf_counter()
hits = parallel_hits(jobs, 2)
hits == sum(expected), time.perf_counter() - start

# %%
# The whole table
for workers in (1, 2, 4):
    start = time.perf_counter()
    hits = parallel_hits(jobs, workers)
    elapsed = time.perf_counter() - start
    assert hits == sum(expected)
    n = sum(size for size, seed in jobs)
    print(workers, "workers", "π ≈", 4*hits/n, "seconds", round(elapsed, 3), "speedup", round(t_seq/elapsed, 2))
