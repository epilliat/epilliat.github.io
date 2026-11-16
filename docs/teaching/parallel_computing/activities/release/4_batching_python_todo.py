# Local version of 4_batching_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # A5 — batching the same workload — 20 minutes
# One matrix `W` applied to 128 examples: one by one, then in batches of B. Needs NumPy and threadpoolctl.

# %%
import time, platform
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
print(platform.python_version(), np.__version__)
threadpool_limits(limits=1, user_api="blas")  # one BLAS thread for ALL timings below
print(threadpool_info())
rng = np.random.default_rng(2026)
m, total = 1024, 128  # increase m only if time/memory allow
W = rng.random((m, m), dtype=np.float32)
X = np.asfortranarray(rng.random((m, total), dtype=np.float32))

def individual():
    # A true matrix-vector product for EACH of the same total examples.
    return np.column_stack([W @ X[:, i] for i in range(total)])

def elapsed(fn):
    fn()  # warm up
    times = []
    for _ in range(3):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.median(times)

# %%
# One example, one batch of 8
W.shape, (W @ X[:, 0]).shape, (W @ X[:, 0:8]).shape

# %% [markdown]
# Your turn: `W` times successive B-column slices of `X`.

# %%
def batched(B):
    # ✏️ your turn: multiply W by successive B-column slices of X, concatenate the results
    raise NotImplementedError("remove this line and write your code")

# %%
# Try it: one timing each; then change 8 to 1, 64…
expected = individual()
print(np.allclose(batched(8), expected, rtol=1e-5, atol=1e-4))
elapsed(individual), elapsed(lambda: batched(8))

# %%
# The whole table
expected = individual()
reference_time = elapsed(individual)
print("Float32; BLAS threads=1; same", total, "examples; W bytes:", W.nbytes)
print("reference GEMV: seconds", reference_time, "µs/example", reference_time/total*1e6)
print("B | total seconds | µs/example | examples/s | speedup vs GEMV")
for B in (1, 8, 64):
    assert np.allclose(batched(B), expected, rtol=1e-5, atol=1e-4)
    seconds = elapsed(lambda: batched(B))
    print(B, seconds, seconds/total*1e6, total/seconds, reference_time/seconds)
