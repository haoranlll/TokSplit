#!/usr/bin/env python3
"""Single-column prefill ablation figures (SMOE vs TokSplit-T vs TokSplit-T+G, 10T & 64T).

Generates two single-column layouts for comparison:
  - Version C (prefill_thread_ablation_C_1x2.{pdf,png}):
      Compact 1x2 side-by-side panels (left: 10 threads, right: 64 threads),
      1K-16K x-ticks, narrow bars, large fonts matching end_to_end_prefill_decode_2x4.py.
  - Version D (prefill_thread_ablation_D_split.{pdf,png}):
      Single axes with left block (10 threads) and right block (64 threads)
      separated by a dashed divider, 3 narrow bars per prompt length.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Data (Prompt length = 1024, 2048, 4096, 8192, 16384; unit: tokens/s)
# =========================================================
prompts = np.array([1024, 2048, 4096, 8192, 16384], dtype=float)
x_labels = ["1K", "2K", "4K", "8K", "16K"]

data = {
    10: {
        "SMOE":         np.array([45.71, 64.451, 103.809, 168.12, 224.84]),
        "TokSplit-T":   np.array([68.98, 125.94, 200.72, 310.87, 312.09]),
        "TokSplit-T+G": np.array([78.65, 138.87, 233.96, 332.02, 332.82]),
    },
    64: {
        "SMOE":         np.array([44.57, 66.65, 107.02, 169.89, 225.19]),
        "TokSplit-T":   np.array([108.55, 201.05, 300.57, 405.39, 357.02]),
        "TokSplit-T+G": np.array([180.10, 286.47, 398.19, 462.69, 397.40]),
        "KTransformers": np.array([96.53, 112.60, 122.45, 121.84, 115.21]),
    },
}
data[10]["KTransformers"] = np.array([26.54, 27.17, 27.07, 27.21, 27.01])

# One point smaller than end_to_end_prefill_decode_2x4.py.
BASE_FONT = 12.0
TITLE_FONT = 14.0
TICK_FONT = 15.5
X_LABEL_FONT = 15.0
Y_LABEL_FONT = 15.0
LEGEND_FONT = 14.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
    "axes.unicode_minus": False,
    "font.size": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "axes.titlesize": TITLE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": LEGEND_FONT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Keep TokSplit-T+G identical to TokSplit in end_to_end_prefill_decode_2x4.py (#7FB069, "..");
# use distinct color + hatch for SMOE and TokSplit-T so they don't collide with llama.cpp / KTransformers.
series_specs = [
    ("SMOE",         "SMOE",         "#9B84B8", "\\\\"),
    ("TokSplit-T",   "TokSplit-T",   "#D67D79", "++"),
    ("TokSplit-T+G", "TokSplit-T+L", "#7FB069", ".."),
]

y_max = max(v.max() for panel in data.values() for v in panel.values())
out_dir = Path(__file__).resolve().parent


def make_version_c():
    """Version C: Compact 1x2 side-by-side bar chart for single column."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

    bar_width = 0.20
    idx = np.arange(len(prompts))
    offsets = [-bar_width, 0.0, bar_width]

    for col, (ax, threads) in enumerate(zip(axes, (10, 64))):
        panel = data[threads]
        for offset, (key, label, color, hatch) in zip(offsets, series_specs):
            ax.bar(
                idx + offset,
                panel[key],
                width=bar_width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.65,
                hatch=hatch,
                zorder=3,
            )
        ax.set_ylim(0, 520)
        ax.set_yticks([0, 100, 200, 300, 400, 500])
        ax.set_xlim(idx[0] - 0.52, idx[-1] + 0.52)
        ax.set_title(f"{threads} Threads", fontsize=TITLE_FONT, fontweight="bold", pad=6)
        ax.set_xticks(idx)
        ax.set_xticklabels(x_labels, fontsize=TICK_FONT)
        ax.set_xlabel("Prompt length", fontsize=X_LABEL_FONT, fontweight="bold")
        ax.tick_params(axis="x", labelsize=TICK_FONT, length=3.5, pad=3)
        ax.tick_params(axis="y", labelsize=TICK_FONT, length=3.5, pad=3)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col > 0:
            ax.tick_params(axis="y", left=False)

    axes[0].set_ylabel("Throughput (tokens/s)", fontsize=Y_LABEL_FONT, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        bbox_to_anchor=(0.53, 1.05),
        prop={"size": LEGEND_FONT, "weight": "bold"},
        handlelength=1.3,
        handletextpad=0.45,
        columnspacing=1.0,
        borderpad=0.32,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.89], w_pad=0.6)

    pdf_path = out_dir / "prefill_thread_ablation_C_1x2.pdf"
    png_path = out_dir / "prefill_thread_ablation_C_1x2.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Version C: {pdf_path} / {png_path}")


def make_version_d():
    """Version D: Single axes with left (10T) and right (64T) blocks separated by dashed line."""
    fig, ax = plt.subplots(figsize=(7.2, 3.7))

    n = len(prompts)
    gap = 0.9
    x_10 = np.arange(n, dtype=float)
    x_64 = np.arange(n, dtype=float) + n + gap
    x_all = np.concatenate([x_10, x_64])
    labels_all = x_labels + x_labels

    # KTransformers matches end_to_end_prefill_decode_2x4.py: #E3A14A, hatch "xx".
    # Draw it first: it is the lowest series in every group.
    d_series = [("KTransformers", "KTransformers", "#E3A14A", "xx")] + series_specs
    bar_width = 0.16
    offsets = (np.arange(len(d_series)) - (len(d_series) - 1) / 2.0) * bar_width

    for offset, (key, label, color, hatch) in zip(offsets, d_series):
        vals = np.concatenate([data[10][key], data[64][key]])
        ax.bar(
            x_all + offset,
            vals,
            width=bar_width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.65,
            hatch=hatch,
            zorder=3,
        )

    sep = (x_10[-1] + x_64[0]) / 2.0
    ax.axvline(sep, linestyle="--", linewidth=1.1, color="#555555", alpha=0.8, zorder=2)

    ax.set_ylim(0, 555)
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_xlim(x_10[0] - 0.55, x_64[-1] + 0.55)
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels_all, fontsize=TICK_FONT)
    ax.tick_params(axis="x", labelsize=TICK_FONT, length=3.5, pad=3)
    ax.tick_params(axis="y", labelsize=TICK_FONT, length=3.5, pad=3)
    ax.set_xlabel("Prompt length", fontsize=X_LABEL_FONT, fontweight="bold", labelpad=6)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=Y_LABEL_FONT, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Block headers inside top of each half
    ax.text(
        np.mean(x_10),
        512,
        "10 Threads",
        ha="center",
        va="center",
        fontsize=TITLE_FONT,
        fontweight="bold",
    )
    ax.text(
        np.mean(x_64),
        512,
        "64 Threads",
        ha="center",
        va="center",
        fontsize=TITLE_FONT,
        fontweight="bold",
    )

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        bbox_to_anchor=(0.53, 1.04),
        prop={"size": LEGEND_FONT, "weight": "bold"},
        handlelength=1.3,
        handletextpad=0.45,
        columnspacing=1.0,
        borderpad=0.32,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    pdf_path = out_dir / "prefill_thread_ablation_D_split.pdf"
    png_path = out_dir / "prefill_thread_ablation_D_split.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    # Also update default output name to Version D
    fig.savefig(out_dir / "prefill_thread_ablation.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "prefill_thread_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Version D: {pdf_path} / {png_path}")


if __name__ == "__main__":
    make_version_c()
    make_version_d()
