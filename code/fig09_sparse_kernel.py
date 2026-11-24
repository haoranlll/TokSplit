#!/usr/bin/env python3
"""Single-column 1x2 figure matching the visual style of the Motivation operator
throughput plot:
  - 4-sided closed box spines (#555555)
  - Muted academic palette (steel-blue #5689B8, warm brown #9B6750, sage #639E7D,
    warm ochre #CC8E4B), solid lines with circular markers
  - Faint horizontal solid + vertical dashed gridlines
  - Panel marks (a) and (b) sit under each subplot; axis titles are not bold
  - Framed rounded in-axes legends (no "(Ours)" tag)
  - Horizontal numeric token ticks + shared bottom "Number of input tokens"
  - Y-axis: "Normalized Operator Speed"

Generates both:
  1. sparse_op_vs_kt_kairox.{png,pdf}       (left line + right bar)
  2. sparse_op_vs_kt_kairox_lines.{png,pdf} (both line charts)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "figures_out" / "fig09"

# ---------------------------------------------------------------------------
# Raw measurements (ms)
# ---------------------------------------------------------------------------
TOKENS = np.array([1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

# (a) Qwen3-Coder-480B-A35B-Instruct-Q4_K_M (ms)
KT_DENSE_MS = np.array([
    0.9608, 1.8275, 3.5717, 7.0796, 8.4200, 10.4025, 15.1958,
    18.8773, 23.9844, 38.8578, 72.1941, 141.9237, 282.7328, 554.5536,
])
TRANS_MS = np.array([
    0.5910, 0.9634, 1.7078, 3.0622, 4.2742, 5.2954, 8.6556,
    12.6993, 20.2922, 31.6472, 48.0044, 80.0263, 142.9336, 269.4678,
])
NEU_MS = np.array([
    0.8810, 1.5789, 2.8674, 5.0533, 6.9131, 8.6215, 12.8288,
    17.0655, 24.2125, 37.8984, 53.8722, 78.4071, 124.1674, 215.0228,
])
KEEP_MS = np.array([
    0.5914, 0.9627, 1.7198, 3.0581, 4.2767, 5.3187, 8.6411,
    12.8642, 20.3923, 31.8756, 48.1592, 78.9747, 139.2630, 258.9094,
])

# (b) DeepSeek-R1 Q4_K_M, layer 10, 64 threads, 20% keep
SPEEDUP_VS_KAIROX = np.array([
    1.93, 2.72, 2.37, 1.85, 1.72, 2.16, 3.12,
    4.39, 4.85, 5.85, 6.22, 6.53, 6.73, 6.50,
])

# Muted palette harmonized with the Motivation operator throughput figure
C_DUAL = "#5689B8"   # Muted steel blue
C_TRANS = "#639E7D"  # Muted sage green
C_NEU = "#CC8E4B"    # Muted warm ochre
C_BASE = "#9B6750"   # Muted warm brown
SPINE_COLOR = "#555555"


def style_axes(ax, x, major_idx, major_lbl):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.75)
    ax.set_xticks(major_idx)
    ax.set_xticklabels(major_lbl, rotation=0, ha="center")
    ax.set_xticks(x, minor=True)
    ax.tick_params(axis="x", which="major", length=2.8, width=0.7, color=SPINE_COLOR, pad=1.5)
    ax.tick_params(axis="x", which="minor", length=1.6, width=0.5, color="#888888")
    ax.tick_params(axis="y", which="major", length=2.8, width=0.7, color=SPINE_COLOR, pad=2.0)
    ax.grid(True, axis="y", linestyle="-", color="#E2E5E9", linewidth=0.65)
    ax.grid(True, axis="x", which="major", linestyle="--", color="#E6E9ED", linewidth=0.55)
    ax.set_axisbelow(True)


def _panel_tag(ax, tag: str) -> None:
    """Corner tag inside the axes, so the letter does not add figure height."""
    ax.text(
        0.98,
        0.97,
        tag,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        fontweight="normal",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.9),
        zorder=6,
        clip_on=False,
    )


def build_figure(right_mode: str = "bar"):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 10.8,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.2,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Wider and shorter than a single column: the two panels read left-to-right.
    # Panel letters sit inside each axes so they do not add a caption row.
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(5.8, 2.58))
    fig.subplots_adjust(left=0.098, right=0.992, top=0.96, bottom=0.20, wspace=0.16)

    x = np.arange(len(TOKENS))
    major_idx = [0, 3, 7, 10, 13]
    major_lbl = ["1", "8", "64", "512", "4096"]

    # ------------------------------------------------------------------
    # (a) Line chart: vs. KT-Kernel
    # ------------------------------------------------------------------
    sp_keep = KT_DENSE_MS / KEEP_MS
    sp_trans = KT_DENSE_MS / TRANS_MS
    sp_neu = KT_DENSE_MS / NEU_MS

    ax_l.axhline(
        1.0, color=C_BASE, linestyle="--", linewidth=1.1,
        label="KT-Kernel", zorder=2,
    )
    ax_l.plot(
        x, sp_neu, color=C_NEU, linestyle="-", linewidth=1.35,
        marker="o", markersize=3.1, label="Native-Only", zorder=3,
    )
    ax_l.plot(
        x, sp_trans, color=C_TRANS, linestyle="-", linewidth=1.65,
        marker="o", markersize=3.7, label="Transpose-Only", zorder=4,
    )
    ax_l.plot(
        x, sp_keep, color=C_DUAL, linestyle="-", linewidth=1.35,
        marker="o", markersize=2.9, label="Dual-Layout", zorder=5,
    )

    style_axes(ax_l, x, major_idx, major_lbl)
    ax_l.set_xlim(-0.6, len(TOKENS) - 0.4)
    ax_l.set_ylim(0.78, 3.55)
    ax_l.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0], ["1.0", "1.5", "2.0", "2.5", "3.0"])
    ax_l.set_ylabel(
        "Normalized Operator Speed",
        labelpad=2.5,
        fontsize=9.8,
        fontweight="bold",
        y=0.42,
    )

    handles_l, labels_l = ax_l.get_legend_handles_labels()
    order_l = [3, 2, 1, 0]
    leg_l = ax_l.legend(
        [handles_l[i] for i in order_l],
        [labels_l[i] for i in order_l],
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D6D9DD",
        borderpad=0.30,
        labelspacing=0.22,
        handlelength=1.45,
        handletextpad=0.38,
    )
    leg_l.get_frame().set_linewidth(0.65)
    _panel_tag(ax_l, "(a)")

    # ------------------------------------------------------------------
    # (b) Right panel: vs. KAIROX (bar or line)
    # ------------------------------------------------------------------
    if right_mode == "bar":
        ax_r.axhline(
            1.0, color=C_BASE, linestyle="--", linewidth=1.1,
            label="KAIROX", zorder=4,
        )
        ax_r.bar(
            x, SPEEDUP_VS_KAIROX, width=0.66, color=C_DUAL, alpha=0.92,
            edgecolor="#3E6B94", linewidth=0.35, label="Dual-Layout", zorder=3,
        )
        ax_r.set_xlim(-0.7, len(TOKENS) - 0.3)
        ax_r.set_ylim(0, 8.5)
        ax_r.set_yticks([0, 2, 4, 6], ["0", "2", "4", "6"])
        handles_r, labels_r = ax_r.get_legend_handles_labels()
        order_r = [1, 0]
    else:
        ax_r.axhline(
            1.0, color=C_BASE, linestyle="--", linewidth=1.1,
            label="KAIROX", zorder=2,
        )
        ax_r.plot(
            x, SPEEDUP_VS_KAIROX, color=C_DUAL, linestyle="-", linewidth=1.5,
            marker="o", markersize=3.3, label="Dual-Layout", zorder=4,
        )
        ax_r.set_xlim(-0.6, len(TOKENS) - 0.4)
        ax_r.set_ylim(0.5, 8.2)
        ax_r.set_yticks([1, 2, 3, 4, 5, 6, 7], ["1", "2", "3", "4", "5", "6", "7"])
        handles_r, labels_r = ax_r.get_legend_handles_labels()
        order_r = [1, 0]

    style_axes(ax_r, x, major_idx, major_lbl)

    leg_r = ax_r.legend(
        [handles_r[i] for i in order_r],
        [labels_r[i] for i in order_r],
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D6D9DD",
        borderpad=0.30,
        labelspacing=0.24,
        handlelength=1.45,
        handletextpad=0.38,
    )
    leg_r.get_frame().set_linewidth(0.65)
    _panel_tag(ax_r, "(b)")

    fig.canvas.draw()
    pos_l = ax_l.get_position()
    pos_r = ax_r.get_position()
    mid_x = 0.5 * (pos_l.x0 + pos_r.x1)
    fig.text(
        mid_x, 0.012, "Number of input tokens",
        ha="center", va="bottom", fontsize=10.0, fontweight="bold",
    )

    return fig, ax_l, ax_r


def save_variant(right_mode: str, stem: str):
    fig, ax_l, ax_r = build_figure(right_mode=right_mode)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"{stem}.png"
    out_pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(out_png, dpi=350, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main():
    save_variant("bar", "sparse_op_vs_kt_kairox")
    save_variant("line", "sparse_op_vs_kt_kairox_lines")


if __name__ == "__main__":
    main()
