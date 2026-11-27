# Local version of 4_locality_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # Two compiled loop orders — 20 minutes
# Numba compiles both loops before measurement. Predict which order is faster, then inspect
# strides and compare sums within tolerance. This tests traversal order without the Python
# interpreter dominating every addition.

# %%
import numpy as np, time
from numba import njit

A = np.random.rand(4000, 4000)     # default NumPy layout

def best(f, r=5):
    f(A)                            # numba compiles on the 1st call
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); f(A); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3            # ms

# %%
@njit
def sum_j_outer(A):                 # j outer, i inner
    s = 0.0
    for j in range(A.shape[1]):
        for i in range(A.shape[0]):
            s += A[i, j]
    return s

# %% [markdown]
# Your turn: the same sum, i outer, j inner.

# %%
@njit
def sum_i_outer(A):
    s = 0.0
    # ✏️ your turn: i outer, j inner — for i in range(A.shape[0]): for j in range(A.shape[1])
    raise NotImplementedError("remove this line and write your code")
    return s

# %%
assert np.isclose(sum_j_outer(A), sum_i_outer(A))   # equal up to rounding (the order of the + differs)
t_j, t_i = best(sum_j_outer), best(sum_i_outer)
print("j outer, i inner :", f"{t_j:.0f} ms")
print("i outer, j inner :", f"{t_i:.0f} ms")
print(f"ratio : {max(t_j, t_i)/min(t_j, t_i):.1f}×   strides: {A.strides}   C-contiguous: {A.flags['C_CONTIGUOUS']}")

# %% [markdown]
# Explain the ratio using this array’s layout. Try a smaller matrix and an F-contiguous array. Cache capacity, prefetching and translation can all contribute; a small timing difference does not prove that traversal order is irrelevant.
