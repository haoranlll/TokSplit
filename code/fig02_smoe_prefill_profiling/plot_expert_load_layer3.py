#!/usr/bin/env python3
"""Layer-3 expert-load 1x4 panels: by-rank (load-sorted) and by-expert-ID variants.

Data: results/smoe_layer_timing_cpu10/q{qlen}_blk_3/activation_load.csv
      columns rank,expert_id,tokens,device

Locked plan for --mode rank (primary):
- 1x4, Layer 3 only, qlen 512/1024/2048/4096
- Sort by tokens descending; pad inactive tokens=0 to length 256
- Colors GPU #d62728 / CPU #1f77b4, alpha~0.85; zero bars white/invisible
- Vertical dashed line at first CPU after sort
- Shared x label: Expert rank
- X ticks 0, 128, 255; Y: Tokens per expert, sharey
- Panel titles: '{q} tokens'; Suptitle: Layer 3
- Per-panel annotation GPU:{n} CPU:{n}
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

QLENS = [512, 1024, 2048, 4096]
LAYER = 3
N_EXPERTS = 256

LOAD_ROOT = Path(__file__).resolve().parents[2] / "data" / "smoe_layer_timing_cpu10"
OUT_ROOT = Path(__file__).resolve().parents[2] / "figures_out" / "fig02"

BAR_COLORS = {
    "gpu": "#d62728",
    "cpu": "#1f77b4",
}
BAR_ALPHA = 0.85


def load_rows(qlen: int):
    path = LOAD_ROOT / f"q{qlen}_blk_{LAYER}" / "activation_load.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "expert_id": int(r["expert_id"]),
                "tokens": float(r["tokens"]),
                "device": r["device"].strip().lower(),
            })
    return rows


def pack_by_id(rows):
    """Natural expert-ID order 0..255; missing -> tokens=0, device=None."""
    tokens = np.zeros(N_EXPERTS, dtype=float)
    devices = [None] * N_EXPERTS
    for r in rows:
        eid = r["expert_id"]
        if 0 <= eid < N_EXPERTS:
            tokens[eid] = r["tokens"]
            devices[eid] = r["device"]
    n_gpu = sum(1 for d, t in zip(devices, tokens) if d == "gpu" and t > 0)
    n_cpu = sum(1 for d, t in zip(devices, tokens) if d == "cpu" and t > 0)
    return tokens, devices, n_gpu, n_cpu


def pack_by_rank(rows):
    """Sort by tokens descending; pad inactive (tokens=0) to length 256 at the tail."""
    present = {r["expert_id"] for r in rows}
    items = [(r["tokens"], r["device"], r["expert_id"]) for r in rows]
    for eid in range(N_EXPERTS):
        if eid not in present:
            items.append((0.0, None, eid))

    def key(it):
        tok, dev, eid = it
        # GPU before CPU before None on ties so the cut stays clean
        dev_ord = 0 if dev == "gpu" else (1 if dev == "cpu" else 2)
        return (-tok, dev_ord, eid)

    items.sort(key=key)
    tokens = np.asarray([t for t, _, _ in items], dtype=float)
    devices = [d for _, d, _ in items]
    n_gpu = sum(1 for d, t in zip(devices, tokens) if d == "gpu" and t > 0)
    n_cpu = sum(1 for d, t in zip(devices, tokens) if d == "cpu" and t > 0)
    boundary = None
    for i, d in enumerate(devices):
        if d == "cpu":
            boundary = i
            break
    return tokens, devices, n_gpu, n_cpu, boundary


def bar_colors(tokens, devices):
    colors = []
    for d, tok in zip(devices, tokens):
        if tok <= 0 or d is None:
            colors.append("#ffffff")
        elif d == "gpu":
            colors.append(BAR_COLORS["gpu"])
        elif d == "cpu":
            colors.append(BAR_COLORS["cpu"])
        else:
            colors.append("#999999")
    return colors


def style_rc():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_variant(mode: str):
    """mode: 'rank' or 'id'."""
    assert mode in ("rank", "id")
    style_rc()

    packs = {}
    for q in QLENS:
        rows = load_rows(q)
        if mode == "rank":
            tokens, devices, n_gpu, n_cpu, boundary = pack_by_rank(rows)
            packs[q] = (tokens, devices, n_gpu, n_cpu, boundary)
        else:
            tokens, devices, n_gpu, n_cpu = pack_by_id(rows)
            packs[q] = (tokens, devices, n_gpu, n_cpu, None)

    ymax = max(float(p[0].max()) for p in packs.values()) * 1.08
    xs = np.arange(N_EXPERTS)

    # Slightly taller for shared xlabel + footer under legend
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.85), sharey=True)

    for ax, q in zip(axes, QLENS):
        tokens, devices, n_gpu, n_cpu, boundary = packs[q]
        ax.bar(
            xs,
            tokens,
            width=1.0,
            color=bar_colors(tokens, devices),
            alpha=BAR_ALPHA,
            edgecolor="none",
            align="center",
            linewidth=0,
        )
        if mode == "rank" and boundary is not None:
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
        # Four panels share one row; the right-edge "255" would collide with the next "0".
        ax.set_xticklabels(["0", "128", "255" if ax is axes[-1] else ""])
        ax.set_ylim(0, ymax)
        ax.set_title(f"{q} tokens", pad=4)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.grid(False, axis="x")
        # Per-panel GPU/CPU counts
        ax.text(
            0.98,
            0.94,
            f"GPU:{n_gpu}  CPU:{n_cpu}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
            clip_on=False,
        )
        print(
            f"[{mode}] q{q}: n_gpu={n_gpu} n_cpu={n_cpu} "
            f"max_tokens={int(tokens.max())} boundary={boundary}"
        )

    axes[0].set_ylabel("Tokens per expert")

    # Single shared x label under the row (not per-panel)
    if mode == "rank":
        fig.supxlabel("Expert rank", fontsize=11, y=0.11)
    else:
        fig.supxlabel("Expert ID", fontsize=11, y=0.11)

    fig.suptitle("Layer 3", fontsize=12, y=0.98)

    legend = fig.legend(
        handles=[
            Patch(
                facecolor=BAR_COLORS["gpu"],
                edgecolor="none",
                alpha=BAR_ALPHA,
                label="GPU experts",
            ),
            Patch(
                facecolor=BAR_COLORS["cpu"],
                edgecolor="none",
                alpha=BAR_ALPHA,
                label="CPU experts",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        columnspacing=1.4,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.28, wspace=0.12)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    stem = f"smoe_expert_load_layer3_by_{mode}"
    out_png = OUT_ROOT / f"{stem}.png"
    out_pdf = OUT_ROOT / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")

    # Canonical aliases for the primary (rank) figure
    if mode == "rank":
        alias_png = OUT_ROOT / "smoe_expert_load_layer3.png"
        alias_pdf = OUT_ROOT / "smoe_expert_load_layer3.pdf"
        fig.savefig(alias_png, bbox_inches="tight")
        fig.savefig(alias_pdf, bbox_inches="tight")
        print(f"Wrote {alias_png}")
        print(f"Wrote {alias_pdf}")

    plt.close(fig)
    return out_png, out_pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("rank", "id", "both"),
        default="rank",
        help="rank=load-sorted (primary); id=expert-ID order; both=write both",
    )
    args = ap.parse_args()
    modes = ("rank", "id") if args.mode == "both" else (args.mode,)
    for m in modes:
        plot_variant(m)


if __name__ == "__main__":
    main()
