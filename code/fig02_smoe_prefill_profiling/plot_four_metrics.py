#!/usr/bin/env python3
"""Plot SMOE 1x3 same-forward four metrics (CPU / GPU / Transfer / layer).

Clean re-run smoe_fig_four_metrics_cpu10 (warmup=2, timed=5, median).
Timing-only; expert-load bars live in plot_expert_load_layer3.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Hardcoded primary table (same forward; Transfer = host_pack + H2D; ms)
# Matches SUMMARY.md under results/smoe_fig_four_metrics_cpu10/
# ---------------------------------------------------------------------------
QLENS = [512, 1024, 2048, 4096]
LAYERS = [3, 32, 60]  # subplot column order

# (qlen, layer) -> (CPU, GPU, Transfer, layer_wall)
DATA = {
    (512, 3): (175.87, 42.31, 129.24, 184.66),
    (512, 32): (241.68, 42.16, 132.03, 243.87),
    (512, 60): (165.25, 47.01, 141.05, 200.88),
    (1024, 3): (134.92, 74.28, 218.07, 317.04),
    (1024, 32): (88.29, 104.91, 312.55, 445.04),
    (1024, 60): (87.42, 81.22, 236.26, 340.52),
    (2048, 3): (108.88, 103.23, 279.57, 429.28),
    (2048, 32): (34.53, 127.42, 358.51, 527.53),
    (2048, 60): (70.57, 101.71, 282.07, 429.67),
    (4096, 3): (73.90, 133.67, 334.85, 543.53),
    (4096, 32): (27.22, 144.31, 366.25, 579.00),
    (4096, 60): (67.39, 121.80, 307.47, 513.36),
}

OUT_ROOT = Path(__file__).resolve().parents[2] / "figures_out" / "fig02"
OUT_PNG = OUT_ROOT / "smoe_stage_service_times.png"
OUT_PDF = OUT_ROOT / "smoe_stage_service_times.pdf"

COLORS = {
    "CPU": "#1f77b4",
    "GPU": "#d62728",
    "Transfer": "#2ca02c",
    "layer": "#111111",
}

STYLES = {
    "CPU": dict(color=COLORS["CPU"], linestyle="-", marker="o", linewidth=1.5, markersize=4.5),
    "GPU": dict(color=COLORS["GPU"], linestyle="-", marker="o", linewidth=1.5, markersize=4.5),
    "Transfer": dict(color=COLORS["Transfer"], linestyle="-", marker="o", linewidth=1.5, markersize=4.5),
    "layer": dict(color=COLORS["layer"], linestyle="--", marker="o", linewidth=1.6, markersize=4.5),
}
LABELS = {
    "CPU": "CPU computation",
    "GPU": "GPU computation",
    "Transfer": "Weight transfer",
    "layer": "Layer time",
}


def series_for_layer(layer: int):
    cpu, gpu, xfer, wall = [], [], [], []
    for q in QLENS:
        c, g, t, w = DATA[(q, layer)]
        cpu.append(c)
        gpu.append(g)
        xfer.append(t)
        wall.append(w)
    return {
        "CPU": np.asarray(cpu),
        "GPU": np.asarray(gpu),
        "Transfer": np.asarray(xfer),
        "layer": np.asarray(wall),
    }


def annotate_last(ax, xs, items):
    """items: (ys, color). Stack labels whose endpoints sit within one line."""
    x = xs[-1]
    ordered = sorted((float(ys[-1]), color) for ys, color in items)
    shift = [0] * len(ordered)
    for i in range(1, len(ordered)):
        if ordered[i][0] - ordered[i - 1][0] < 80:
            shift[i] = shift[i - 1] + 11
    for (y, color), dy in zip(ordered, shift):
        ax.annotate(
            f"{int(round(y))}",
            xy=(x, y),
            xytext=(7, dy),
            textcoords="offset points",
            fontsize=9,
            color=color,
            va="center",
            ha="left",
            annotation_clip=False,
        )


def main():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True)
    keys = ["CPU", "GPU", "Transfer", "layer"]

    for ax, layer in zip(axes, LAYERS):
        ser = series_for_layer(layer)
        xs = np.asarray(QLENS, dtype=float)
        ends = []
        for k in keys:
            ys = ser[k]
            ax.plot(xs, ys, label=LABELS[k], **STYLES[k])
            ends.append((ys, COLORS[k]))
        annotate_last(ax, xs, ends)

        ax.set_xscale("log", base=2)
        ax.set_xticks(QLENS)
        ax.set_xticklabels([str(q) for q in QLENS])
        ax.set_xlim(QLENS[0] * 0.85, QLENS[-1] * 1.55)
        ax.set_ylim(0, 650)
        ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Input tokens")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.grid(False, axis="x")

    axes[0].set_ylabel("Time (ms)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.2,
        handletextpad=0.4,
    )

    fig.text(
        0.5,
        0.005,
        "Transfer = host staging + H2D; component busy ≠ layer wall-clock; "
        "timing from same forward (cpu10).",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28, wspace=0.18)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
