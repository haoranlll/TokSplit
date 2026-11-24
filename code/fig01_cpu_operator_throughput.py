"""Stack the KT-Kernel MoE curve and the dense / 20%-sparse curve.

Both panels use the same equal-spaced token axis, so the plot boxes match.
Tick labels stop at 1024; 2048 and 4096 remain on the curves.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

tokens = np.array([1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)

kt_ms = np.array(
    [
        0.9882,
        1.7527,
        3.3478,
        6.6870,
        8.1069,
        10.2387,
        16.5884,
        23.4479,
        29.3484,
        41.3152,
        72.4204,
        140.6998,
        274.5742,
        548.5140,
    ],
    dtype=float,
)
fastest_dense_ms = np.array(
    [0.3477, 0.3414, 0.4065, 0.485, 0.5963, 0.700, 1.197, 2.23, 3.65, 7.21, 14.28, 29.14, 61.32, 122.0],
    dtype=float,
)
kairox20_ms = np.array(
    [0.5190, 0.7825, 0.9572, 1.769, 2.221, 2.989, 5.015, 10.38, 19.60, 38.34, 77.06, 156.7, 318.1, 630.5],
    dtype=float,
)

kt_tps = tokens / kt_ms * 1000.0
dense_tps = tokens / fastest_dense_ms * 1000.0
kairox_tps = tokens / kairox20_ms * 1000.0

# Match end_to_end_prefill_decode_2x4.py so the two paper figures share one type scale.
BASE_FONT = 13.0
TITLE_FONT = 15.0
TICK_FONT = 16.5
X_LABEL_FONT = 16.0
Y_LABEL_FONT = 16.0
LEGEND_FONT = 15.5

plt.rcParams.update(
    {
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
    }
)

x = np.arange(len(tokens))
labeled = tokens <= 1024
tick_labels = [str(int(t)) for t in tokens[labeled]]

fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.4), sharex=True)
# Fixed margins keep the two axes the same width despite different y-tick digits.
fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.11, hspace=0.38)

ax = axes[0]
ax.plot(x, kt_tps, color="#9A6248", marker="o", markersize=7, linewidth=2.2, label="KT-Kernel (dense)")
ax.set_xlim(-0.45, len(tokens) - 0.45)
ax.set_ylabel("Tokens/s", fontsize=Y_LABEL_FONT, fontweight="bold")
ax.set_title("(a) MoE operator throughput of KT-Kernel (Top-8)", fontsize=TITLE_FONT, fontweight="bold", pad=6)
ax.set_ylim(0, 8200)
ax.set_yticks([2000, 4000, 6000])
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
ax.legend(frameon=True, loc="lower right", prop={"size": LEGEND_FONT, "weight": "bold"})
ax.tick_params(axis="both", labelsize=TICK_FONT, labelbottom=True)

ax = axes[1]
ax.plot(x, dense_tps, color="#5B8DB8", marker="o", markersize=7, linewidth=2.2, label="llama.cpp Dense-Kernel")
ax.plot(x, kairox_tps, color="#9A6248", marker="o", markersize=7, linewidth=2.2, label="KAIROX-Kernel 80%-Sparse")
ax.set_ylabel("Tokens/s", fontsize=Y_LABEL_FONT, fontweight="bold")
ax.set_xlabel("Number of input tokens", fontsize=X_LABEL_FONT, fontweight="bold", labelpad=4)
ax.set_title("(b) Single-expert FFN operator throughput", fontsize=TITLE_FONT, fontweight="bold", pad=6)
ax.set_ylim(0, 40000)
ax.set_yticks([10000, 20000, 30000])
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
ax.legend(frameon=True, loc="upper left", prop={"size": LEGEND_FONT, "weight": "bold"})
ax.tick_params(axis="both", labelsize=TICK_FONT)

for ax in axes:
    ax.set_xticks(x[labeled])
    ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=TICK_FONT)

out_dir = os.path.dirname(os.path.abspath(__file__))
png = os.path.join(out_dir, "combined_operator_throughput.png")
pdf = os.path.join(out_dir, "combined_operator_throughput.pdf")
fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
print(png)
print(pdf)
