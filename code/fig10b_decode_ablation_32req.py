#!/usr/bin/env python3
"""Normalized 32K decode speed ablation across Qwen3-235B, Qwen3-Coder-480B, and DeepSeek-R1.

Compares Base (1.0x), Base+ED (Expert-Deferral), and Base+ED+DS
(Expert-Deferral + Dual-granularity Sparsity) in a single three-section bar chart,
styled as the decode counterpart to prefill_component_ablation_32k.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Raw 32K decode throughput (tokens/s) & normalized speedups
# =========================================================
models = ["Qwen3-235B", "Qwen3-Coder-480B", "DeepSeek-R1"]

raw_tps = {
    "Base":       np.array([54.59,  26.14, 28.94], dtype=float),
    "Base+ED":    np.array([100.09, 38.95, 31.85], dtype=float),
    "Base+ED+DS": np.array([156.87, 59.24, 37.85], dtype=float),
}

norm_speed = {k: v / raw_tps["Base"] for k, v in raw_tps.items()}

print(f"{'Model':<18} | {'Base':>14} | {'Base+ED':>16} | {'Base+ED+DS':>16}")
print("-" * 74)
for i, m in enumerate(models):
    print(
        f"{m:<18} | "
        f"{raw_tps['Base'][i]:6.2f} (1.00x) | "
        f"{raw_tps['Base+ED'][i]:6.2f} ({norm_speed['Base+ED'][i]:.2f}x) | "
        f"{raw_tps['Base+ED+DS'][i]:6.2f} ({norm_speed['Base+ED+DS'][i]:.2f}x)"
    )

# =========================================================
# Style (twin layout to prefill_component_ablation_32k.py)
# =========================================================
BASE_FONT = 14.0
TITLE_FONT = 15.5
TICK_FONT = 15.0
Y_LABEL_FONT = 15.2
LEGEND_FONT = 14.2
ANNOT_FONT = 12.0

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

# Unified color palette:
# - Base:       #E3A14A ("xx") -- identical to Base in prefill_component_ablation_32k.py
# - Base+ED:    #8A96A3 ("||") -- cool silver-slate + vertical stripes for Expert-Deferral
# - Base+ED+DS: #C87594 ("oo") -- distinct muted rose-berry + ring hatch for ED + Dual-granularity Sparsity
series_specs = [
    ("Base",       "#E3A14A", "xx"),
    ("Base+ED",    "#8A96A3", "||"),
    ("Base+ED+DS", "#C87594", "oo"),
]

fig, ax = plt.subplots(figsize=(7.6, 3.8))

group_step = 1.25
x = np.arange(len(models), dtype=float) * group_step
bar_width = 0.25
offsets = (np.arange(len(series_specs)) - (len(series_specs) - 1) / 2.0) * bar_width

for offset, (name, color, hatch) in zip(offsets, series_specs):
    vals = norm_speed[name]
    bars = ax.bar(
        x + offset,
        vals,
        width=bar_width,
        label=name,
        color=color,
        edgecolor="black",
        linewidth=0.75,
        hatch=hatch,
        zorder=3,
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.06,
            f"{v:.2f}×",
            ha="center",
            va="bottom",
            fontsize=ANNOT_FONT,
            fontweight="bold",
            color="#222222",
            zorder=5,
        )

# Horizontal 1.0x reference line
ax.axhline(1.0, color="#B85636", linestyle="--", linewidth=1.3, alpha=0.75, zorder=2)

# Vertical dividers splitting the figure into 3 model sections
for i in range(len(models) - 1):
    sep = 0.5 * (x[i] + x[i + 1])
    ax.axvline(sep, color="#555555", linestyle="--", linewidth=1.1, alpha=0.85, zorder=2)

ax.set_xlim(x[0] - 0.60, x[-1] + 0.60)
ax.set_ylim(0, 3.40)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["0", "1×", "2×", "3×"], fontsize=TICK_FONT)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=14.5, fontweight="bold")
ax.tick_params(axis="x", length=0, pad=7)
ax.tick_params(axis="y", labelsize=TICK_FONT, length=4.0, width=1.0, pad=3)
ax.set_ylabel("Normalized Decode Speed", fontsize=Y_LABEL_FONT, fontweight="bold")
ax.text(
    0.985,
    0.96,
    "(b)",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=TITLE_FONT,
    fontweight="bold",
    color="#222222",
    zorder=6,
)

ax.grid(axis="y", which="major", linestyle="--", linewidth=0.55, color="#D4D4D4", alpha=0.85, zorder=0)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.9)
    spine.set_color("#444444")

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    bbox_to_anchor=(0.53, 1.04),
    prop={"size": LEGEND_FONT, "weight": "bold"},
    handlelength=1.45,
    handletextpad=0.4,
    columnspacing=1.3,
    borderpad=0.32,
)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout(rect=[0, 0, 1, 0.90])

out_dir = Path(__file__).resolve().parent
pdf_path = out_dir / "decode_component_ablation_32k.pdf"
png_path = out_dir / "decode_component_ablation_32k.png"
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved PDF: {pdf_path}")
print(f"Saved PNG: {png_path}")
