# Local version of 1_diagnostic_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # Diagnose before optimizing — 20 minutes
# Time the three stages of a pipeline, then speed up one.

# %%
import time, platform
import numpy as np
print(platform.python_version(), np.__version__)
raw = list(range(200_000))

def transform(x):
    return np.array([float(value) ** 2 for value in x])

def pipeline(transform):
    start = time.perf_counter()
    x = np.asarray(raw, dtype=np.float64)
    prepared = time.perf_counter()
    y = transform(x)
    transformed = time.perf_counter()
    result = float(y.sum())
    finished = time.perf_counter()
    return result, np.array([prepared-start, transformed-prepared, finished-transformed])

# %%
# One run: the result, then the three stage times (s)
pipeline(transform)

# %%
# Now 3 runs, median of each stage
pipeline(transform)  # warm up
baseline = [pipeline(transform) for _ in range(3)]
reference = baseline[0][0]
before = np.median([times for result, times in baseline], axis=0)
print("prepare / transform / summarize (s):", before)
print("total (s):", before.sum(), "largest share:", before.max()/before.sum())

# %% [markdown]
# Your turn: the same squares with one NumPy operation.

# %%
def faster_transform(x):
    # ✏️ your turn: compute the same squares with one NumPy array operation
    raise NotImplementedError("remove this line and write your code")

# %%
# Try it small first
faster_transform(np.arange(5.0)), transform(np.arange(5.0))

# %%
faster_transform(np.asarray(raw, dtype=np.float64))  # warm up
runs = [pipeline(faster_transform) for _ in range(3)]
assert all(np.isclose(value, reference, rtol=1e-12) for value, times in runs)
after = np.median([times for value, times in runs], axis=0)
fraction = before[1] / before.sum()
local_gain = before[1] / after[1]
predicted = 1 / (1 - fraction + fraction/local_gain)
print("after stage times (s):", after)
print("local gain", local_gain, "predicted total gain", predicted, "observed total gain", before.sum()/after.sum())
