# Local version of 1_warmup_python_todo.ipynb: run it cell by cell (VS Code: Shift+Enter on each `# %%`).


# %% [markdown]
# # What does a loop cost? — 20 minutes
# The same 10⁶ numbers summed three ways: your `for` loop, `numpy`, `numba`.
#
# Before running, predict the ranking of the three variants and the first-call cost.

# %%
import time, timeit, numpy as np

v = np.random.rand(1_000_000)     # a million numbers
v_list = v.tolist()               # the SAME data, as a plain Python list

def ms(t): return f"{t*1e3:.2f} ms"
def best(f, r=5):                 # time per call, %timeit-style
    n = 1
    while timeit.timeit(f, number=n) < 0.05: n *= 10
    return min(timeit.repeat(f, number=n, repeat=r)) / n

# %% [markdown]
# ## 1. Your turn — the pure Python loop
# Write `mysum` with a plain `for` loop (no `sum`, no `numpy`).

# %%
def mysum(x):
    # ✏️ your turn: add up the elements of x with a for loop
    raise NotImplementedError("remove this line and write your code")

# %%
# Try it small first
mysum([1.0, 2.0, 3.0])

# %%
# Run me — no need to edit.
assert abs(mysum(v_list) - sum(v_list)) < 1e-6, 'mysum should return the sum'
t_loop = best(lambda: mysum(v_list))
t_np   = best(lambda: np.sum(v))
print("your Python loop :", ms(t_loop))
print("numpy.sum        :", ms(t_np))
print(f"speedup vs Python loop (NumPy) : {t_loop/t_np:.0f}×")

# %% [markdown]
# ## 2. Your turn — the same loop, compiled
# The same loop again, in `sum_numba`.

# %%
try:
    from numba import njit
except ImportError:                       # Colab already has numba
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numba'])
    from numba import njit

# %%
@njit
def sum_numba(x):
    # ✏️ your turn: the SAME loop as mysum — numba compiles it to native code
    raise NotImplementedError("remove this line and write your code")

# %%
# Run me — the very first call is timed on its own, then the warmed-up calls.
t0 = time.perf_counter(); first = sum_numba(v); t_first = time.perf_counter() - t0
assert abs(first - np.sum(v)) < 1e-6, 'sum_numba should return the sum'
t_nb = best(lambda: sum_numba(v))
print("numba, first call  :", ms(t_first))
print("numba, warmed up     :", ms(t_nb))
print("your pure loop   :", ms(t_loop))
print(f"speedup vs Python loop (warmed-up Numba) : {t_loop/t_nb:.0f}×")
print(f"first-call / warmed-up time     : {t_first/t_nb:.0f}×")

# %% [markdown]
# **Explain:** report first-call and warmed-up times separately. Where does each variant execute the loop? List one difference besides compilation between the Python-list and NumPy-array paths. Record your measurement protocol.
