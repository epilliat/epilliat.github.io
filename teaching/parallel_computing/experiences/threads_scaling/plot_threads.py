#!/usr/bin/env python3
"""Speedup vs number of threads — the honest Amdahl curve.

  python3 plot_threads.py    # -> results/threads_scaling.png

Reads results/threads_scaling.csv (`threads,variant,mean_s,std_s,pi_hat`), produced by
run.sh (one julia process per point).

Three curves, and the whole lesson is in the gap between them:

  - ideal  : y = k, the dashed diagonal nobody reaches;
  - spawn  : Threads.@spawn — real parallelism, bends away from the diagonal (Amdahl);
  - async  : @sync/@async — same syntax, same work, flat at ~1x. CPU work has nothing
             to interleave: concurrency is not parallelism.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CSV = os.path.join(RESULTS, "threads_scaling.csv")

STYLE = {
    "spawn": ("Threads.@spawn (parallelism)", "tab:blue"),
    "async": ("@sync/@async (concurrency)", "tab:orange"),
}


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} — run `bash run.sh` first")

    # variant -> {threads: (mean, std)}
    data = defaultdict(dict)
    with open(CSV) as f:
        for row in csv.DictReader(f):
            data[row["variant"]][int(row["threads"])] = (
                float(row["mean_s"]),
                float(row["std_s"]),
            )

    ks = sorted(data["spawn"])
    if not ks:
        raise SystemExit("no rows for the `spawn` variant")

    # Baseline: the SEQUENTIAL time at k=1 — the only honest reference for a speedup.
    base = data["seq"][min(data["seq"])][0]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ks, ks, "k--", lw=1, alpha=0.5, label="ideal (y = k)")

    for variant, (label, color) in STYLE.items():
        if variant not in data:
            continue
        pts = sorted(data[variant])
        speed = [base / data[variant][k][0] for k in pts]
        # propagate the std through the ratio: d(base/t) = base/t * (dt/t)
        err = [
            (base / data[variant][k][0]) * (data[variant][k][1] / data[variant][k][0])
            for k in pts
        ]
        ax.errorbar(pts, speed, yerr=err, marker="o", ms=4, capsize=3,
                    color=color, label=label)

    ax.set_xlabel("threads (one julia process per point)")
    ax.set_ylabel(f"speedup vs sequential ({base:.2f} s)")
    ax.set_title("Monte-Carlo π: speedup vs threads")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(RESULTS, "threads_scaling.png")
    fig.savefig(out, dpi=140)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
