#!/usr/bin/env python3
"""Plot of the sum's time as a function of the size N.

Two modes:

  python3 plot_scaling.py              # mean curves + ±std bands -> results/scaling.png
  python3 plot_scaling.py --frames DIR # cumulative frames scaling_frame_0..3.png -> DIR/
                                        # (progressive reveal: mysum -> base -> CI)

Reads results/scaling_julia_{mysum,basesum}.csv (format `n,mean_s,std_s`).
Linear axes: we want to see the "staircase" of Base's `sum` (SIMD / pairwise steps).
"""

import os
import sys

import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Fixed order and style (color pinned per series).
SERIES = [
    ("julia_mysum", "julia mysum", "tab:orange"),
    ("julia_basesum", "julia sum (Base)", "tab:blue"),
]

TITLE = "Time of the sum as a function of the size N (Julia)"
XLABEL = "N (number of elements)"
YLABEL = "time per call (µs)"


def read_csv(path):
    ns, means, stds = [], [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            n, m, s = line.split(",")
            ns.append(int(n))
            means.append(float(m) * 1e6)   # s -> µs
            stds.append(float(s) * 1e6)
    return ns, means, stds


def load():
    data = []
    for key, label, color in SERIES:
        path = os.path.join(RESULTS, f"scaling_{key}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path} — run scaling.jl first.")
        ns, means, stds = read_csv(path)
        data.append((ns, means, stds, label, color))
    return data


def axis_limits(data):
    xmax = max(max(ns) for ns, *_ in data)
    ytop = max(m + s for ns, means, stds, *_ in data for m, s in zip(means, stds))
    return (0, xmax * 1.02), (0, ytop * 1.08)


def plot_all():
    data = load()
    xlim, ylim = axis_limits(data)
    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    for ns, means, stds, label, color in data:
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(ns, lo, hi, color=color, alpha=0.25, linewidth=0)
        ax.plot(ns, means, color=color, linewidth=1.0, label=label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE)
    ax.legend(loc="upper left", framealpha=1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS, "scaling.png")
    plt.savefig(out, dpi=130)
    print("-> results/scaling.png")
    if os.environ.get("DISPLAY"):
        plt.show()


def make_frames(outdir):
    """Cumulative frames with pinned axes: 0=empty, 1=mysum, 2=+base, 3=+±std bands.

    Opaque images of the same dimension: stacked in a `::: {.r-stack}`, the top one
    hides the previous ones -> illusion of adding an element on every click.
    """
    os.makedirs(outdir, exist_ok=True)
    data = load()
    xlim, ylim = axis_limits(data)

    def base_axes():
        fig = plt.figure(figsize=(9, 5.5))
        ax = fig.add_subplot(111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(YLABEL)
        ax.set_title(TITLE)
        ax.grid(True, alpha=0.3)
        return fig, ax

    def draw(ax, upto_mean, with_bands):
        for ns, means, stds, label, color in data[:upto_mean]:
            if with_bands:
                lo = [m - s for m, s in zip(means, stds)]
                hi = [m + s for m, s in zip(means, stds)]
                ax.fill_between(ns, lo, hi, color=color, alpha=0.25, linewidth=0)
            ax.plot(ns, means, color=color, linewidth=1.0, label=label)
        if upto_mean > 0:
            ax.legend(loc="upper left", framealpha=1.0)

    # (upto_mean, with_bands): 0 empty, 1 mysum, 2 +base, 3 +CI on both.
    specs = [(0, False), (1, False), (2, False), (2, True)]
    for k, (upto, bands) in enumerate(specs):
        fig, ax = base_axes()
        draw(ax, upto, bands)
        fig.tight_layout()
        out = os.path.join(outdir, f"scaling_frame_{k}.png")
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
