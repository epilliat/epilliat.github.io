#!/usr/bin/env python3
"""Comparative plot of the warmup curves.

Two modes:

  python3 plot_warmup.py              # overlays every results/warmup_*.csv -> results/warmup.png
  python3 plot_warmup.py --frames DIR # cumulative frames warmup_frame_0..5.png -> DIR/
                                       # (for the progressive reveal of the reveal.js slides)

Reads the `warmup_*.csv` files of the results/ subfolder (format `iteration,time_s`). Log-y scale
(the JIT spike of the 1st call flattens a linear axis).

Extensible: the GPU experiment only has to drop further warmup_gpu_*.csv files into results/.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt

# All results (CSV + warmup.png) live in warmup_jit/results/.
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Fixed order and style of the series for the cumulative frames (color pinned per series,
# otherwise the matplotlib cycle reassigns the colors depending on the number of curves drawn).
SERIES = [
    ("python_pure", "python pure", "tab:purple"),
    ("python_numba", "python numba", "tab:green"),
    ("python_numpy", "python numpy", "tab:red"),
    ("julia_mysum", "julia mysum", "tab:orange"),
    ("julia_basesum", "julia basesum", "tab:blue"),
]

TITLE = "JIT warmup: decay of a CPU sum's time"
XLABEL = "iteration (call #)"
YLABEL = "time per call (ms, log scale)"


def read_csv(path):
    its, ts = [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, t = line.split(",")
            its.append(int(i))
            ts.append(float(t))
    return its, ts


def label_from(path):
    # warmup_python_numba.csv -> "python numba"
    base = os.path.basename(path)
    return base[len("warmup_"):-len(".csv")].replace("_", " ")


def plot_all():
    """Default mode: a single plot overlaying every CSV present."""
    paths = sorted(glob.glob(os.path.join(RESULTS, "warmup_*.csv")))
    if not paths:
        raise SystemExit("No results/warmup_*.csv found. Run warmup.py / warmup.jl first.")

    plt.figure(figsize=(9, 5.5))
    for path in paths:
        its, ts = read_csv(path)
        ms = [t * 1e3 for t in ts]
        plt.plot(its, ms, marker=".", markersize=2, linewidth=0.8, label=label_from(path))

    plt.yscale("log")
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(TITLE)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS, "warmup.png")
    plt.savefig(out, dpi=130)
    print("-> results/warmup.png")
    if os.environ.get("DISPLAY"):
        plt.show()


def make_frames(outdir):
    """Cumulative frames with pinned axes, for the progressive reveal of the slides.

    frame_0 = empty axes; frame_k = the first k series of SERIES (in order).
    The images are opaque (white background) and of identical dimensions: stacked in a
    `::: {.r-stack}`, the top one hides the previous ones, which gives the illusion
    of adding a curve on every click.
    """
    os.makedirs(outdir, exist_ok=True)

    # Load the available series in the fixed order; compute common axes.
    loaded = []
    xmax, ymin, ymax = 0, float("inf"), 0.0
    for key, label, color in SERIES:
        path = os.path.join(RESULTS, f"warmup_{key}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path} — run warmup.py / warmup.jl first.")
        its, ts = read_csv(path)
        ms = [t * 1e3 for t in ts]
        loaded.append((its, ms, label, color))
        xmax = max(xmax, max(its))
        ymin = min(ymin, min(ms))
        ymax = max(ymax, max(ms))

    # Margins: a bit of air around the extremes (log axis -> multiplicative factors).
    ylim = (ymin / 1.6, ymax * 1.6)
    xlim = (-xmax * 0.02, xmax * 1.02)

    def base_axes():
        fig = plt.figure(figsize=(9, 5.5))
        ax = fig.add_subplot(111)
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(YLABEL)
        ax.set_title(TITLE)
        ax.grid(True, which="both", alpha=0.3)
        return fig, ax

    for k in range(len(loaded) + 1):
        fig, ax = base_axes()
        for its, ms, label, color in loaded[:k]:
            ax.plot(its, ms, marker=".", markersize=2, linewidth=0.8,
                    color=color, label=label)
        if k > 0:
            # Anchored legend of stable size (reserves the 5 entries from the start).
            ax.legend(loc="upper right", framealpha=1.0)
        fig.tight_layout()
        out = os.path.join(outdir, f"warmup_frame_{k}.png")
        fig.savefig(out, dpi=130, facecolor="white")
        plt.close(fig)
        print(f"-> {out}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--frames":
        make_frames(sys.argv[2])
    elif len(sys.argv) == 1:
        plot_all()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
