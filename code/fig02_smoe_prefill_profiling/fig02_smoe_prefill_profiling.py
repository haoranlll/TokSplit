#!/usr/bin/env python3
"""Combine Layer-3 expert-load (1x4, top) + three-layer timing (1x3, bottom).

Does not modify plot_expert_load_layer3.py or plot_four_metrics.py.
Imports helpers/data from those modules so styling stays in sync with the
user-tuned load figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from plot_expert_load_layer3 import (  # noqa: E402
    BAR_ALPHA,
    BAR_COLORS,
    N_EXPERTS,
    QLENS,
    bar_colors,
    load_rows,
    pack_by_rank,
)
from plot_four_metrics import (  # noqa: E402
    COLORS,
    LABELS,
    LAYERS,
    STYLES,
    series_for_layer,
)

OUT_ROOT = Path(__file__).resolve().parents[2] / "figures_out" / "fig02"
OUT_STEM = "smoe_load_and_stage_times"


def style_rc():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def main():
    style_rc()

    # ---- load packs (rank-sorted, matches user-tuned figure) ----
    packs = {}
    for q in QLENS:
        tokens, devices, n_gpu, n_cpu, boundary = pack_by_rank(load_rows(q))
        packs[q] = (tokens, devices, n_gpu, n_cpu, boundary)
        print(
            f"[rank] q{q}: n_gpu={n_gpu} n_cpu={n_cpu} "
            f"max_tokens={int(tokens.max())} boundary={boundary}"
        )
    ymax = max(float(p[0].max()) for p in packs.values()) * 1.08
    xs_bar = np.arange(N_EXPERTS)

    fig = plt.figure(figsize=(7.16, 5.85))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.05, 1.0],
        hspace=1.05,
        left=0.10, right=0.985, top=0.96, bottom=0.22,
    )
    gs_top = outer[0].subgridspec(1, 4, wspace=0.16)
    gs_bot = outer[1].subgridspec(1, 3, wspace=0.28)

    # ---- top: Layer 3 expert load ----
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(4)]
    for i, (ax, q) in enumerate(zip(axes_top, QLENS)):
        if i > 0:
            ax.sharey(axes_top[0])
        tokens, devices, n_gpu, n_cpu, boundary = packs[q]
        ax.bar(
            xs_bar,
            tokens,
            width=1.0,
            color=bar_colors(tokens, devices),
            alpha=BAR_ALPHA,
            edgecolor="none",
            align="center",
            linewidth=0,
        )
        if boundary is not None:
            ax.axvline(
                boundary - 0.5,
                color="#111111",
                linestyle="--",
                linewidth=0.9,
                alpha=0.85,
                zorder=3,
            )
        ax.set_xlim(-0.5, N_EXPERTS - 0.5)
        ax.set_xticks([0, 128, 255])
        # Avoid colliding "255" with next panel's "0"
        ax.set_xticklabels(["0", "128", "255" if ax is axes_top[-1] else ""])
        ax.set_ylim(0, ymax)
        ax.set_title(f"{q} tokens", pad=4)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.grid(False, axis="x")
        if i > 0:
            ax.tick_params(axis="y", labelleft=False)
        # Sit on the 400-token gridline, inside the panel.
        ax.text(
            0.97, 400,
            f"GPU:{n_gpu} CPU:{n_cpu}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=11, color="#222222",
            clip_on=True, zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.9),
        )
    axes_top[0].set_ylabel("Tokens per expert")
    fig.canvas.draw()
    top_bottom = min(ax.get_position().y0 for ax in axes_top)
    fig.text(0.53, top_bottom - 0.058, "Expert rank", ha="center", va="top", fontsize=12)

    leg_load = fig.legend(
        handles=[
            Patch(
                facecolor=BAR_COLORS["gpu"], edgecolor="none",
                alpha=BAR_ALPHA, label="GPU experts",
            ),
            Patch(
                facecolor=BAR_COLORS["cpu"], edgecolor="none",
                alpha=BAR_ALPHA, label="CPU experts",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.53, top_bottom - 0.088),
        columnspacing=1.4,
        handletextpad=0.4,
        fontsize=12,
    )

    # ---- bottom: three-layer timing ----
    axes_bot = [fig.add_subplot(gs_bot[0, i]) for i in range(3)]
    keys = ["CPU", "GPU", "Transfer", "layer"]
    for i, (ax, layer) in enumerate(zip(axes_bot, LAYERS)):
        if i > 0:
            ax.sharey(axes_bot[0])
        ser = series_for_layer(layer)
        xs = np.asarray(QLENS, dtype=float)
        for k in keys:
            ys = ser[k]
            ax.plot(xs, ys, label=LABELS[k], **STYLES[k])
        ax.set_xscale("log", base=2)
        ax.set_xticks(QLENS)
        ax.set_xticklabels([str(q) for q in QLENS])
        ax.set_xlim(QLENS[0] * 0.85, QLENS[-1] * 1.08)
        ax.set_ylim(0, 650)
        ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Input tokens")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.grid(False, axis="x")
        if i > 0:
            ax.tick_params(axis="y", labelleft=False)
    axes_bot[0].set_ylabel("Time (ms)")

    fig.canvas.draw()
    bot_y0 = min(ax.get_position().y0 for ax in axes_bot)
    handles, labels = axes_bot[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.53, bot_y0 - 0.082),
        columnspacing=1.2,
        handletextpad=0.4,
        fontsize=12,
    )
    fig.text(
        0.53, bot_y0 - 0.168,
        "(b) Stage time across layers",
        ha="center", va="top", fontsize=12,
    )
    # Just under the load legend, not floating in the gap above the timing row.
    fig.text(
        0.53, top_bottom - 0.148,
        "(a) Dynamic expert assignment on layer 3",
        ha="center", va="top", fontsize=12,
    )

    # Keep load legend from being GC'd
    fig.add_artist(leg_load)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_png = OUT_ROOT / f"{OUT_STEM}.png"
    out_pdf = OUT_ROOT / f"{OUT_STEM}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
