#!/usr/bin/env python3
"""LoadGear speedup over KT-Kernel across token counts (1 to 16384) in a single merged bar chart.

Left block: Mixtral-8x22B dense; Right block: DeepSeek-R1.
Y-axis: Speedup (×) = Throughput(LoadGear) / Throughput(KT-Kernel) = Latency(KT-Kernel) / Latency(LoadGear),
with a horizontal dashed reference line at 1.0× (KT-Kernel baseline).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

# =========================================================
# Raw latency data (unit: ms) -> Throughput (tokens/s) & Speedup
# =========================================================
tokens = np.array(
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    dtype=float,
)

# Mixtral-8x22B dense (token=12 excluded so x-axis is pure powers of 2)
mx_loadgear_ms = np.array([
    1.2269, 2.2743, 3.6558, 5.0384, 6.4242, 9.3248, 15.2257,
    18.7583, 25.3704, 39.3267, 67.8874, 125.5049, 244.9504, 446.0599, 932.877,
], dtype=float)
mx_kt_ms = np.array([
    1.3706, 2.6662, 5.2525, 10.8951, 6.7434, 8.885, 15.6376,
    31.5496, 57.0088, 123.3954, 243.3453, 504.6989, 997.3557, 1890.349, 3677.491,
], dtype=float)

# DeepSeek-R1
ds_kt_ms = np.array([
    0.9882, 1.7527, 3.3478, 6.687, 10.2387, 16.5884, 23.4479,
    29.3484, 41.3152, 72.4204, 140.6998, 274.5742, 548.514, 1090.356, 2151.462,
], dtype=float)
ds_loadgear_ms = np.array([
    0.788, 1.4872, 2.8843, 5.6942, 10.2116, 15.6252, 22.9281,
    28.5227, 38.7361, 61.8845, 86.1637, 117.6664, 192.3464, 351.5373, 693.0815,
], dtype=float)

# Throughput (tokens/s) and Speedup (LoadGear_tps / KT_tps)
mx_kt_tps = tokens / mx_kt_ms * 1000.0
mx_lg_tps = tokens / mx_loadgear_ms * 1000.0
mx_speedup = mx_lg_tps / mx_kt_tps

ds_kt_tps = tokens / ds_kt_ms * 1000.0
ds_lg_tps = tokens / ds_loadgear_ms * 1000.0
ds_speedup = ds_lg_tps / ds_kt_tps

print(f"{'tokens':>6} | {'Mixtral speedup':>15} | {'DeepSeek-R1 speedup':>19}")
for t, sm, sd in zip(tokens, mx_speedup, ds_speedup):
    print(f"{int(t):6d} | {sm:14.2f}x | {sd:18.2f}x")

# =========================================================
# Style (unified with paper_picture/ figures, single-column ready)
# =========================================================
BASE_FONT = 14.0
TITLE_FONT = 13.5
TICK_FONT = 15.0
X_LABEL_FONT = 14.0
Y_LABEL_FONT = 12.8
LEGEND_FONT = 12.5

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

# Unified colors: LoadGear (#5BA8A0) matches Base+L in prefill_component_ablation_32k.py,
# keeping #7FB069 reserved exclusively for full TokSplit (Base+T+L) across the paper.
BAR_COLOR = "#5BA8A0"
BAR_EDGE  = "#2E5E5A"
BASE_LINE = "#B85636"

n = len(tokens)
gap = 1.6
x_mx = np.arange(n, dtype=float)
x_ds = np.arange(n, dtype=float) + n + gap
x_all = np.concatenate([x_mx, x_ds])
speedup_all = np.concatenate([mx_speedup, ds_speedup])

fig, ax = plt.subplots(figsize=(7.6, 3.85))

bar_width = 0.64
bars = ax.bar(
    x_all,
    speedup_all,
    width=bar_width,
    color=BAR_COLOR,
    edgecolor=BAR_EDGE,
    linewidth=0.75,
    label="LoadGear",
    zorder=3,
)

# Horizontal 1.0x baseline (KT-Kernel)
base_line = ax.axhline(
    1.0,
    color=BASE_LINE,
    linestyle="--",
    linewidth=2.0,
    label="KT-Kernel (1.0×)",
    zorder=4,
)

# Vertical divider between Mixtral-8x22B and DeepSeek-R1
sep = (x_mx[-1] + x_ds[0]) / 2.0
ax.axvline(sep, color="#555555", linestyle="--", linewidth=1.1, alpha=0.85, zorder=2)

# Major ticks at [1, 8, 64, 512, 4096, 16384] (rotated 45°, like reference figure);
# minor ticks at all intermediate token counts.
major_tokens = [1, 8, 64, 512, 4096, 16384]
major_local_idx = [int(np.where(tokens == v)[0][0]) for v in major_tokens]
major_labels_one = ["1", "8", "64", "512", "4K", "16K"]
minor_local_idx = [i for i in range(n) if i not in major_local_idx]

major_xticks = [x_mx[i] for i in major_local_idx] + [x_ds[i] for i in major_local_idx]
major_xticklabels = major_labels_one + major_labels_one
minor_xticks = [x_mx[i] for i in minor_local_idx] + [x_ds[i] for i in minor_local_idx]

ax.set_xticks(major_xticks)
ax.set_xticklabels(major_xticklabels, rotation=45, ha="right", rotation_mode="anchor", fontsize=TICK_FONT - 0.5)
ax.xaxis.set_minor_locator(FixedLocator(minor_xticks))
ax.tick_params(axis="x", which="major", length=5.5, width=1.0, pad=2)
ax.tick_params(axis="x", which="minor", length=3.0, width=0.75)
ax.tick_params(axis="y", which="major", labelsize=TICK_FONT, length=4.0, width=1.0, pad=3)

ax.set_xlim(x_mx[0] - 0.85, x_ds[-1] + 0.85)
ax.set_ylim(0, 4.75)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(["0", "1×", "2×", "3×", "4×"])
ax.set_xlabel("Number of input tokens", fontsize=X_LABEL_FONT, fontweight="bold", labelpad=5)
ax.set_ylabel("Normalized Operator Speed", fontsize=Y_LABEL_FONT, fontweight="bold")

# Subtle major gridlines matching reference style
ax.grid(axis="y", which="major", linestyle="-", linewidth=0.55, color="#D8D8D8", alpha=0.85, zorder=0)
ax.grid(axis="x", which="major", linestyle="--", linewidth=0.55, color="#DCDCDC", alpha=0.85, zorder=0)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.9)
    spine.set_color("#444444")

# Model section headers placed in clean open space of each half
ax.text(
    x_mx[6],
    4.36,
    "Mixtral-8x22B",
    ha="center",
    va="center",
    fontsize=TITLE_FONT,
    fontweight="bold",
)
ax.text(
    np.mean(x_ds),
    4.36,
    "DeepSeek-R1",
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
    ncol=2,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    bbox_to_anchor=(0.53, 1.04),
    prop={"size": LEGEND_FONT, "weight": "bold"},
    handlelength=1.6,
    handletextpad=0.45,
    columnspacing=1.5,
    borderpad=0.32,
)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout(rect=[0, 0, 1, 0.90])

out_dir = Path(__file__).resolve().parent
for stem in ("loadgear_operator_speedup", "loadgear_operator_throughput"):
    fig.savefig(out_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved PDF: {out_dir / 'loadgear_operator_speedup.pdf'}")
print(f"Saved PNG: {out_dir / 'loadgear_operator_speedup.png'}")
