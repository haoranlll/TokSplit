#!/usr/bin/env python3
"""End-to-end Prefill (top 1x4) and Decode (bottom 1x4) throughput across 4 MoE models."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Data (Batch = 1, 2, 4, 8, 16, 32; unit: tokens/s)
# =========================================================
batches = np.array([1, 2, 4, 8, 16, 32], dtype=float)
x_labels = ["1", "2", "4", "8", "16", "32"]

models = [
    {
        "name": "Mixtral-8x22B",
        "prefill": {
            "llama.cpp":     np.array([96.44, 173.19, 280.44, 335.99, 346.19, 353.35]),
            "KTransformers": np.array([119.23, 118.51, 119.47, 118.91, 120.22, 123.02]),
            "TokSplit":      np.array([334.64, 412.48, 457.65, 453.85, 467.17, 486.46]),
        },
        "decode": {
            "llama.cpp":     np.array([10.10, 13.16, 18.11, 24.72, 32.08, 37.86]),
            "KTransformers": np.array([13.93, 16.98, 19.27, 24.89, 32.00, 37.91]),
            "TokSplit":      np.array([14.26, 18.11, 26.48, 31.02, 33.36, 37.90]),
        },
    },
    {
        "name": "Qwen3-235B",
        "prefill": {
            "llama.cpp":     np.array([203.21, 345.42, 545.33, 527.32, 495.58, 439.85]),
            "KTransformers": np.array([191.40, 188.75, 186.95, 188.20, 188.69, 188.27]),
            "TokSplit":      np.array([572.06, 567.31, 772.83, 905.19, 970.61, 971.48]),
        },
        "decode": {
            "llama.cpp":     np.array([9.71, 13.30, 18.25, 25.04, 35.07, 46.26]),
            "KTransformers": np.array([22.37, 28.15, 32.96, 37.18, 65.16, 100.09]),
            "TokSplit":      np.array([23.46, 36.27, 50.14, 65.01, 114.62, 156.87]),
        },
    },
    {
        "name": "Qwen3-Coder-480B",
        "prefill": {
            "llama.cpp":     np.array([110.47, 191.38, 316.29, 308.05, 292.58, 266.136]),
            "KTransformers": np.array([99.12, 110.41, 110.70, 110.46, 110.77, 113.60]),
            "TokSplit":      np.array([363.12, 345.95, 458.16, 525.80, 540.90, 582.48]),
        },
        "decode": {
            "llama.cpp":     np.array([6.52, 8.74, 11.80, 15.80, 21.80, 27.42]),
            "KTransformers": np.array([16.99, 17.05, 18.30, 20.41, 24.47, 39.24]),
            "TokSplit":      np.array([21.12, 26.26, 33.58, 43.58, 52.11, 59.24]),
        },
    },
    {
        "name": "DeepSeek-R1",
        "prefill": {
            "llama.cpp":     np.array([63.34, 112.84, 185.34, 171.23, 148.60, 117.63]),
            "KTransformers": np.array([121.47, 111.98, 113.18, 112.51, 110.94, 110.44]),
            "TokSplit":      np.array([184.21, 261.89, 365.06, 334.78, 329.60, 312.61]),
        },
        "decode": {
            "llama.cpp":     np.array([6.82, 8.39, 10.29, 12.21, 16.68, 22.49]),
            "KTransformers": np.array([15.99, 19.83, 20.96, 21.45, 28.46, 31.91]),
            "TokSplit":      np.array([17.61, 21.66, 27.49, 28.86, 34.47, 37.85]),
        },
    },
]

# =========================================================
# Print speedup summary
# =========================================================
for m in models:
    name = m["name"]
    pf_vs_llama = m["prefill"]["TokSplit"] / m["prefill"]["llama.cpp"]
    pf_vs_kt    = m["prefill"]["TokSplit"] / m["prefill"]["KTransformers"]
    dc_vs_llama = m["decode"]["TokSplit"]  / m["decode"]["llama.cpp"]
    dc_vs_kt    = m["decode"]["TokSplit"]  / m["decode"]["KTransformers"]
    print(f"===== {name} =====")
    print(
        f"  Prefill speedup vs llama.cpp:     avg={pf_vs_llama.mean():.2f}x, max={pf_vs_llama.max():.2f}x"
    )
    print(
        f"  Prefill speedup vs KTransformers: avg={pf_vs_kt.mean():.2f}x, max={pf_vs_kt.max():.2f}x"
    )
    print(
        f"  Decode  speedup vs llama.cpp:     avg={dc_vs_llama.mean():.2f}x, max={dc_vs_llama.max():.2f}x"
    )
    print(
        f"  Decode  speedup vs KTransformers: avg={dc_vs_kt.mean():.2f}x, max={dc_vs_kt.max():.2f}x"
    )

# =========================================================
# Style (English paper figure, consistent with paper_picture/)
# =========================================================
BASE_FONT = 13.0
TITLE_FONT = 15.0
TICK_FONT = 16.5
X_LABEL_FONT = 16.0
Y_LABEL_FONT = 16.0
LEGEND_FONT = 15.5

X_LABEL_TEXT = "Batch size"
Y_LABEL_TEXT = "Throughput (tokens/s)"
PREFILL_PREFIX = "Prefill"
DECODE_PREFIX = "Decode"

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

series_specs = [
    ("llama.cpp",     "#6C8EBF", "//"),
    ("KTransformers", "#E3A14A", "xx"),
    ("TokSplit",      "#7FB069", ".."),
]

# =========================================================
# Plot: 2 rows × 4 columns (top = Prefill, bottom = Decode)
# =========================================================
fig, axes = plt.subplots(2, 4, figsize=(18.6, 7.6), sharex=True)

bar_width = 0.24
idx = np.arange(len(batches))
offsets = [-bar_width, 0.0, bar_width]

for col, m in enumerate(models):
    model_name = m["name"]

    # -----------------------------
    # Row 0: Prefill throughput
    # -----------------------------
    ax = axes[0, col]
    for offset, (sys_name, color, hatch) in zip(offsets, series_specs):
        ax.bar(
            idx + offset,
            m["prefill"][sys_name],
            width=bar_width,
            label=sys_name,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch,
            zorder=3,
        )

    pf_max = max(v.max() for v in m["prefill"].values())
    ax.set_ylim(0, pf_max * 1.12)
    ax.set_title(f"{PREFILL_PREFIX}: {model_name}", fontsize=TITLE_FONT, fontweight="bold", pad=8)
    ax.set_xticks(idx)
    ax.set_xticklabels(x_labels, fontsize=TICK_FONT)
    ax.tick_params(axis="x", labelbottom=True, labelsize=TICK_FONT)
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if col == 0:
        ax.set_ylabel(Y_LABEL_TEXT, fontsize=Y_LABEL_FONT, fontweight="bold")

    # -----------------------------
    # Row 1: Decode throughput
    # -----------------------------
    ax = axes[1, col]
    for offset, (sys_name, color, hatch) in zip(offsets, series_specs):
        ax.bar(
            idx + offset,
            m["decode"][sys_name],
            width=bar_width,
            label=sys_name,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch,
            zorder=3,
        )

    dc_max = max(v.max() for v in m["decode"].values())
    ax.set_ylim(0, dc_max * 1.12)
    ax.set_title(f"{DECODE_PREFIX}: {model_name}", fontsize=TITLE_FONT, fontweight="bold", pad=8)
    ax.set_xticks(idx)
    ax.set_xticklabels(x_labels, fontsize=TICK_FONT)
    ax.set_xlabel(X_LABEL_TEXT, fontsize=X_LABEL_FONT, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if col == 0:
        ax.set_ylabel(Y_LABEL_TEXT, fontsize=Y_LABEL_FONT, fontweight="bold")

# =========================================================
# Shared legend (top)
# =========================================================
handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    bbox_to_anchor=(0.5, 1.03),
    prop={"size": LEGEND_FONT, "weight": "bold"},
    handlelength=1.6,
    columnspacing=1.8,
)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=1.8, w_pad=1.2)

# =========================================================
# Save PDF + PNG
# =========================================================
out_dir = Path(__file__).resolve().parent
pdf_path = out_dir / "e2e_prefill_decode_2x4.pdf"
png_path = out_dir / "e2e_prefill_decode_2x4.png"

fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved PDF: {pdf_path}")
print(f"Saved PNG: {png_path}")
