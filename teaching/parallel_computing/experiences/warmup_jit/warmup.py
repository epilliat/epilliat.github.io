#!/usr/bin/env python3
"""Experiment "JIT warmup" — Python.

We time R successive calls of the *same* sum over a CPU vector and observe the
decay of the time per call. Three variants:

  - numba @njit  : compiled just in time → big spike on the 1st call (JIT compilation),
                   then a fast steady state. This is the Python mirror of Julia.
  - pure         : pure Python loop → no JIT, a flat but slow line.
  - numpy        : numpy.sum → already-compiled reference, flat and fast.

Each variant writes a CSV in the common `iteration,time_s` format (see README),
read back by plot_warmup.py.
"""

import os
import time
import numpy as np
from numba import njit

# --- common parameters (same values as warmup.jl) --------------------------
N = 1_000_000        # vector size (float64)
R = 2000             # iterations for the fast variants (njit, numpy)
R_PURE = 200         # iterations for the pure Python loop (each call ~0.1 s)
SEED = 0

# All results go to warmup_jit/results/ (anchored to the script's folder).
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@njit(cache=False)
def sum_loop_njit(x):
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i]
    return s


def sum_pure(x):
    s = 0.0
    for v in x:
        s += v
    return s


def sum_numpy(x):
    return float(np.sum(x))


def measure(fn, x, reps):
    """Returns the list of times (s) of `reps` successive calls of fn(x).

    The result is accumulated into a sink to prevent the computation from being
    eliminated.
    """
    times = []
    sink = 0.0
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        s = fn(x)
        t1 = time.perf_counter_ns()
        sink += s
        times.append((t1 - t0) * 1e-9)
    return times, sink


def write_csv(path, times):
    with open(path, "w") as f:
        f.write("iteration,time_s\n")
        for i, t in enumerate(times):
            f.write(f"{i},{t:.9e}\n")


def summary(name, times):
    first_call = times[0]
    tail = sorted(times[-max(1, len(times) // 10):])
    stable = tail[len(tail) // 2]  # median of the last 10 %
    ratio = first_call / stable if stable > 0 else float("inf")
    print(f"  {name:<14} 1st call = {first_call*1e3:9.3f} ms | "
          f"stable = {stable*1e3:9.4f} ms | ratio = {ratio:8.1f}x")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(N)  # allocated once: we measure the warmup, not the allocation

    print(f"JIT warmup experiment (Python) — N={N}, R={R}, R_pure={R_PURE}")

    variants = [
        ("python_numba", sum_loop_njit, R),
        ("python_numpy", sum_numpy, R),
        ("python_pure", sum_pure, R_PURE),
    ]

    for tag, fn, reps in variants:
        times, sink = measure(fn, x, reps)
        path = os.path.join(RESULTS, f"warmup_{tag}.csv")
        write_csv(path, times)
        summary(tag, times)
        print(f"    -> results/warmup_{tag}.csv  (sink={sink:.6e})")


if __name__ == "__main__":
    main()
