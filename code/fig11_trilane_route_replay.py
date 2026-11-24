#!/usr/bin/env python3
"""Drive fixed-backend route-replay sweeps and plot Oracle-regret & ablation figures.

Addresses pre-submission review recommendations:
  1) Fixed-backend route replay across candidate prefix lengths ``n`` under the
     exact same MoE layer, routing counts, resident cache state, and prefetched
     ring state.
  2) Direct comparison against Oracle empirical optimum ``min_n T_real(n)``:
     reports suboptimality gap ("how much slower than the empirical best prefix")
     and exposed (unhidden) H2D stall time on the GPU compute stream.
  3) Step-wise policy ablation under identical calibrated operator primitives:
     Fixed-k, Per-expert Local Rule, 2-Lane Compute-only, Apex-style 2-Lane
     ``max(T_cpu, T_h2d + T_gpu)``, TriLane w/o Prefetch-ready priority,
     TriLane w/o Last-group tail, Legacy Eq.(10), and Full TriLane Eq.(10').

Server launch example (before running --collect-url):
  export KT_TRILANE_REPLAY_JSONL=/tmp/trilane_replay.jsonl
  export KT_TRILANE_REPLAY_LAYERS=3,32,60
  export KT_TRILANE_REPLAY_TRIGGER_FILE=/tmp/trilane_replay.arm
  export KT_TRILANE_REPLAY_STRIDE=8
  export KT_TRILANE_REPLAY_REPEATS=3
  # launch sglang server as usual...

Client collection + plotting example:
  python code/fig11_trilane_route_replay.py \
      --jsonl data/fig11/trilane_replay.jsonl \
      --curve-layer 60 \
      --out-prefix figures_out/fig11/trilane_route_replay
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _send_generate(
    endpoint: str, input_ids: List[int], timeout_s: float = 900.0
) -> float:
    import requests

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(endpoint, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    _ = resp.json()
    return time.perf_counter() - t0


def collect_replay_runs(
    server_url: str,
    tokenizer_path: str,
    token_lengths: Sequence[int],
    warmup_rounds: int,
    trigger_file: Optional[str],
    jsonl_path: str,
    timeout_s: float = 1200.0,
) -> None:
    """Warm up cost model & LRU caches, arm replay trigger, and run exact prompts."""
    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from stress_test_prompts import PROMPTS

    # Disarm trigger during warmup so calibration converges at full speed
    if trigger_file and os.path.exists(trigger_file):
        os.remove(trigger_file)

    if os.path.exists(jsonl_path):
        print(f"[collect] removing stale replay file {jsonl_path}")
        os.remove(jsonl_path)

    print(f"[collect] loading tokenizer from {tokenizer_path} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    first_four = list(PROMPTS[:4])
    encoded_four = [
        (idx, tok.encode(text, add_special_tokens=False))
        for idx, text in enumerate(first_four)
    ]
    joined_fallback = tok.encode("\n\n".join(first_four), add_special_tokens=False)

    ids_by_length: Dict[int, List[int]] = {}
    for length in token_lengths:
        cands = [(i, ids) for i, ids in encoded_four if len(ids) >= length]
        if cands:
            _, chosen = min(cands, key=lambda kv: (len(kv[1]), kv[0]))
            ids_by_length[length] = chosen[:length]
        else:
            reps = max(1, (length // max(1, len(joined_fallback))) + 1)
            ids_by_length[length] = (joined_fallback * reps)[:length]

    endpoint = server_url.rstrip("/") + "/generate"
    flush_endpoint = server_url.rstrip("/") + "/flush_cache"

    def _flush_kv_cache() -> None:
        import requests

        try:
            requests.post(flush_endpoint, timeout=15.0)
        except Exception:
            pass

    def _set_armed(armed: bool) -> None:
        if not trigger_file:
            return
        if armed:
            Path(trigger_file).parent.mkdir(parents=True, exist_ok=True)
            Path(trigger_file).write_text("armed\n", encoding="utf-8")
        elif os.path.exists(trigger_file):
            os.remove(trigger_file)

    per_len_warmups = max(1, int(warmup_rounds))
    _set_armed(False)
    try:
        for length in token_lengths:
            prompt_ids = ids_by_length[length]
            print(
                f"\n[stage tokens={length}] running {per_len_warmups} steady-state "
                f"same-distribution warmup(s) (trigger disarmed) ..."
            )
            _set_armed(False)
            for w in range(per_len_warmups):
                _flush_kv_cache()
                dt = _send_generate(endpoint, prompt_ids, timeout_s=timeout_s)
                print(
                    f"  [warmup {w + 1}/{per_len_warmups}] tokens={length:5d} "
                    f"done in {dt:.2f}s"
                )

            _flush_kv_cache()
            _set_armed(True)
            print(
                f"  [replay] armed trigger -> sweeping candidate n at steady-state "
                f"for tokens={length:5d} ..."
            )
            dt = _send_generate(endpoint, prompt_ids, timeout_s=timeout_s)
            print(f"  [replay] tokens={length:5d} completed in {dt:.2f}s")
            _set_armed(False)
    finally:
        _set_armed(False)
        if trigger_file:
            print(f"[collect] disarmed replay trigger -> {trigger_file}")


def generate_synthetic_demo_jsonl(
    jsonl_path: str,
    layers: Sequence[int],
    token_lengths: Sequence[int],
) -> None:
    """Generate realistic synthetic replay sweeps for offline pipeline verification."""
    rng = np.random.default_rng(2026)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for seq_idx, tok in enumerate(token_lengths, start=4):
        for lid in layers:
            cap = min(224, int(130 + 28 * math.log2(max(1, tok / 512))))
            n_grid = sorted(set(list(range(0, cap + 1, 8)) + [cap]))
            n_pref = 0 if lid == 3 else (8 if tok <= 2048 else 12)

            # Underlying Zipfian expert route loads
            ranks = np.arange(1, cap + 1, dtype=np.float64)
            weights = 1.0 / np.power(ranks + 3.0, 0.88)
            weights /= weights.sum()
            non_res_pairs = tok * 8 * 0.76
            cand_counts = np.maximum(1.0, non_res_pairs * weights)
            cum_pairs = np.concatenate([[0.0], np.cumsum(cand_counts)])

            sweep = []
            for n in n_grid:
                rem_ratio = max(
                    0.0, (non_res_pairs - cum_pairs[n]) / max(non_res_pairs, 1.0)
                )
                cpu_ms = 8.0 + (0.115 * tok) * (rem_ratio ** 0.92)
                paid = max(0, n - n_pref)
                h2d_ms = (1.2 + 1.08 * (2 + paid)) if (2 + paid) > 0 else 0.0
                groups = math.ceil(n / 8)
                gpu_res_ms = 0.003 * (tok * 8 * 0.24)
                gpu_grp_ms = groups * 0.42 + 0.0055 * cum_pairs[n]
                gpu_busy_ms = gpu_res_ms + gpu_grp_ms
                last_grp_ms = (0.42 + 0.0055 * (cum_pairs[n] - cum_pairs[max(0, n - 8)])) if n > 0 else 0.0
                gpu_stream_span = max(gpu_busy_ms, h2d_ms + last_grp_ms)
                exposed_wait = max(0.0, gpu_stream_span - gpu_busy_ms)
                noise = float(rng.normal(0.0, 0.8))
                layer_ms = max(cpu_ms, gpu_stream_span) + 1.5 + max(-0.6, noise)
                sweep.append(
                    {
                        "n": int(n),
                        "prefetch_priority": True,
                        "n_paid": int(paid + 2),
                        "prefetch_hits": int(min(n, n_pref)),
                        "layer_ms": round(layer_ms, 4),
                        "layer_std_ms": round(abs(noise) + 0.3, 4),
                        "cpu_ms": round(cpu_ms, 4),
                        "gpu_busy_ms": round(gpu_busy_ms, 4),
                        "h2d_ms": round(h2d_ms, 4),
                        "gpu_exposed_h2d_wait_ms": round(exposed_wait, 4),
                        "layer_exposed_h2d_ms": round(
                            max(0.0, layer_ms - max(cpu_ms, gpu_busy_ms)), 4
                        ),
                    }
                )

            by_n = {pt["n"]: pt for pt in sweep}
            oracle_pt = min(sweep, key=lambda p: p["layer_ms"])
            oracle_n = int(oracle_pt["n"])
            oracle_ms = float(oracle_pt["layer_ms"])

            def _nearest(target_n: int, penalty_ms: float = 0.0) -> Dict[str, Any]:
                nearest_pt = min(sweep, key=lambda p: abs(p["n"] - target_n))
                real_ms = float(nearest_pt["layer_ms"]) + penalty_ms
                reg_ms = max(0.0, real_ms - oracle_ms)
                return {
                    "n": int(nearest_pt["n"]),
                    "prefetch_priority": penalty_ms == 0.0,
                    "est_layer_ms": round(real_ms, 4),
                    "real_layer_ms": round(real_ms, 4),
                    "real_cpu_ms": nearest_pt["cpu_ms"],
                    "real_gpu_busy_ms": nearest_pt["gpu_busy_ms"],
                    "real_h2d_ms": nearest_pt["h2d_ms"],
                    "gpu_exposed_h2d_wait_ms": nearest_pt["gpu_exposed_h2d_wait_ms"],
                    "layer_exposed_h2d_ms": nearest_pt["layer_exposed_h2d_ms"],
                    "prefetch_hits": nearest_pt["prefetch_hits"] if penalty_ms == 0.0 else max(0, nearest_pt["prefetch_hits"] - 4),
                    "n_paid": nearest_pt["n_paid"],
                    "regret_ms": round(reg_ms, 4),
                    "regret_pct": round(reg_ms / oracle_ms * 100.0, 3),
                    "delta_n_vs_oracle": int(nearest_pt["n"]) - oracle_n,
                }

            pref_pen = 5.5 if n_pref > 0 else 0.0
            policies = {
                "no_stream": _nearest(0),
                "fixed_32": _nearest(min(32, cap), pref_pen * 0.5),
                "fixed_64": _nearest(min(64, cap), pref_pen * 0.5),
                "smoe_rule": _nearest(min(cap, int(oracle_n * 1.65 + 32)), pref_pen * 0.3),
                "local_rule": _nearest(min(cap, int(oracle_n * 1.55 + 24)), pref_pen * 0.3),
                "two_lane_compute": _nearest(min(cap, int(oracle_n * 1.45 + 16)), pref_pen * 0.3),
                "apex_2lane": _nearest(max(8, int(oracle_n * 0.66 - n_pref)), pref_pen),
                "apex_2lane_grp": _nearest(max(8, int(oracle_n * 0.62 - n_pref)), pref_pen),
                "trilane_no_prefetch": _nearest(max(8, oracle_n - n_pref), pref_pen),
                "trilane_no_tail": _nearest(min(cap, oracle_n + 8)),
                "trilane_legacy_eq10": _nearest(oracle_n),
                "trilane": _nearest(oracle_n),
            }

            records.append(
                {
                    "ts": time.time(),
                    "chunk_seq": seq_idx,
                    "layer": lid,
                    "tokens": tok,
                    "top_k": 8,
                    "cap": cap,
                    "group_size": 8,
                    "repeats": 3,
                    "n_resident": 8,
                    "n_incoming_resident": 2,
                    "resident_pairs": int(tok * 8 * 0.24),
                    "retained_pairs": int(tok * 8 * 0.20),
                    "incoming_pairs": int(tok * 8 * 0.04),
                    "init_all_prefetched": list(range(n_pref)),
                    "init_landed_prefetched": list(range(n_pref)),
                    "oracle": {
                        "n": oracle_n,
                        "prefetch_priority": True,
                        "real_layer_ms": oracle_ms,
                        "real_cpu_ms": oracle_pt["cpu_ms"],
                        "real_gpu_busy_ms": oracle_pt["gpu_busy_ms"],
                        "real_h2d_ms": oracle_pt["h2d_ms"],
                        "gpu_exposed_h2d_wait_ms": oracle_pt["gpu_exposed_h2d_wait_ms"],
                        "layer_exposed_h2d_ms": oracle_pt["layer_exposed_h2d_ms"],
                    },
                    "policies": policies,
                    "sweep": sweep,
                }
            )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[demo] wrote {len(records)} synthetic replay records to {jsonl_path}")


def _derive_smoe_rule_from_sweep(rec: Dict[str, Any], tau_tokens: float = 22.0) -> Optional[Dict[str, Any]]:
    """Derive SMOE per-expert token-threshold selection directly on measured sweep points."""
    sweep = sorted(rec.get("sweep", []), key=lambda p: int(p["n"]))
    if not sweep or "stream_pairs" not in sweep[0]:
        return None
    chosen = sweep[0]
    n_pref = len(rec.get("init_landed_prefetched", []))
    for i in range(1, len(sweep)):
        dn = int(sweep[i]["n"]) - int(sweep[i - 1]["n"])
        dp = int(sweep[i]["stream_pairs"]) - int(sweep[i - 1]["stream_pairs"])
        if dn <= 0:
            continue
        if (dp / float(dn)) >= tau_tokens:
            chosen = sweep[i]
        elif int(sweep[i]["n"]) > n_pref + 8:
            break
    ora_ms = float(rec["oracle"]["real_layer_ms"])
    ora_n = int(rec["oracle"]["n"])
    real_ms = float(chosen["layer_ms"])
    reg_ms = max(0.0, real_ms - ora_ms)
    reg_pct = (reg_ms / ora_ms * 100.0) if ora_ms > 0 else 0.0
    return {
        "n": int(chosen["n"]),
        "prefetch_priority": bool(chosen.get("prefetch_priority", True)),
        "est_layer_ms": round(real_ms, 4),
        "real_layer_ms": round(real_ms, 4),
        "real_cpu_ms": round(float(chosen["cpu_ms"]), 4),
        "real_gpu_busy_ms": round(float(chosen["gpu_busy_ms"]), 4),
        "real_h2d_ms": round(float(chosen["h2d_ms"]), 4),
        "gpu_exposed_h2d_wait_ms": round(float(chosen["gpu_exposed_h2d_wait_ms"]), 4),
        "layer_exposed_h2d_ms": round(float(chosen.get("layer_exposed_h2d_ms", 0.0)), 4),
        "prefetch_hits": int(chosen.get("prefetch_hits", 0)),
        "n_paid": int(chosen.get("n_paid", 0)),
        "regret_ms": round(reg_ms, 4),
        "regret_pct": round(reg_pct, 3),
        "delta_n_vs_oracle": int(chosen["n"]) - ora_n,
    }


def load_replay_records(jsonl_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tau = float(os.environ.get("KT_SMOE_TOKEN_THRESHOLD", "22"))
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if "smoe_rule" not in rec.get("policies", {}):
                    derived = _derive_smoe_rule_from_sweep(rec, tau_tokens=tau)
                    if derived is not None:
                        rec.setdefault("policies", {})["smoe_rule"] = derived
                rows.append(rec)
    if not rows:
        raise RuntimeError(f"No replay records found in {jsonl_path}")
    return rows


POLICY_DISPLAY_ORDER: List[Tuple[str, str]] = [
    ("no_stream", "Resident-Only (n=0)"),
    ("fixed_32", "Fixed-32"),
    ("fixed_64", "Fixed-64"),
    ("smoe_rule", "SMOE-style (Token-Threshold)"),
    ("two_lane_compute", "2-Lane Compute-Only (No H2D)"),
    ("local_rule", "Per-Expert Marginal Rule"),
    ("apex_2lane", "2-Path Serial (H2D+GPU)"),
    ("trilane_no_prefetch", "TriLane w/o Prefetch-Ready"),
    ("trilane_no_tail", "TriLane w/o Last-Group Tail"),
    ("trilane_legacy_eq10", "TriLane (Legacy Eq.10)"),
    ("trilane", "TriLane (Ours)"),
]


def summarize_and_export_csv(
    records: List[Dict[str, Any]], csv_path: str
) -> None:
    """Print comprehensive Oracle-regret & exposed-H2D summary and write CSV."""
    print("\n" + "=" * 108)
    print(
        f"{'Policy':<30} | {'Mean Regret':>12} | {'Mean Gap %':>10} | "
        f"{'P90 Gap %':>9} | {'Mean |Δn|':>9} | {'Exposed H2D':>12} | {'CPU Time':>9}"
    )
    print("-" * 108)

    csv_rows: List[Dict[str, Any]] = []
    for p_key, p_label in POLICY_DISPLAY_ORDER:
        reg_ms_list: List[float] = []
        reg_pct_list: List[float] = []
        abs_dn_list: List[float] = []
        signed_dn_list: List[float] = []
        stall_list: List[float] = []
        cpu_list: List[float] = []

        for r in records:
            p = r["policies"].get(p_key)
            if p is None:
                continue
            reg_ms_list.append(float(p["regret_ms"]))
            reg_pct_list.append(float(p["regret_pct"]))
            dn = float(p["delta_n_vs_oracle"])
            abs_dn_list.append(abs(dn))
            signed_dn_list.append(dn)
            stall_list.append(float(p["gpu_exposed_h2d_wait_ms"]))
            cpu_list.append(float(p["real_cpu_ms"]))

        if not reg_ms_list:
            continue

        mean_reg_ms = float(np.mean(reg_ms_list))
        mean_reg_pct = float(np.mean(reg_pct_list))
        p90_reg_pct = float(np.percentile(reg_pct_list, 90))
        mean_abs_dn = float(np.mean(abs_dn_list))
        mean_signed_dn = float(np.mean(signed_dn_list))
        mean_stall = float(np.mean(stall_list))
        mean_cpu = float(np.mean(cpu_list))

        print(
            f"{p_label:<30} | {mean_reg_ms:9.2f} ms | {mean_reg_pct:8.2f} % | "
            f"{p90_reg_pct:7.2f} % | {mean_abs_dn:9.1f} | {mean_stall:9.2f} ms | {mean_cpu:6.1f} ms"
        )
        csv_rows.append(
            {
                "policy_key": p_key,
                "policy_label": p_label,
                "num_cases": len(reg_ms_list),
                "mean_regret_ms": round(mean_reg_ms, 3),
                "mean_regret_pct": round(mean_reg_pct, 3),
                "p90_regret_pct": round(p90_reg_pct, 3),
                "mean_abs_delta_n": round(mean_abs_dn, 2),
                "mean_signed_delta_n": round(mean_signed_dn, 2),
                "mean_gpu_exposed_h2d_wait_ms": round(mean_stall, 3),
                "mean_cpu_ms": round(mean_cpu, 3),
            }
        )

    # Check Eq.(10) vs Eq.(10') difference count
    diff_eq10 = sum(
        1
        for r in records
        if "trilane_legacy_eq10" in r["policies"]
        and "trilane" in r["policies"]
        and r["policies"]["trilane_legacy_eq10"]["n"]
        != r["policies"]["trilane"]["n"]
    )
    print("-" * 108)
    print(
        f"[Eq.(10) boundary check] Legacy Eq.(10) vs Exact Eq.(10') chose different n in "
        f"{diff_eq10}/{len(records)} replay snapshots."
    )
    print("=" * 108 + "\n")

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[summary] saved CSV table -> {csv_path}")


def plot_replay_figure(
    records: List[Dict[str, Any]],
    curve_layer: int,
    token_lengths: Sequence[int],
    out_prefix: str,
) -> None:
    """Render two-row publication figure:
      (a) Candidate-n empirical latency sweep + exposed H2D stall + policy choices
      (b) Left: Step-wise policy ablation (Oracle Regret % across token lengths)
          Right: Unhidden bottleneck breakdown (CPU excess vs Exposed H2D stall)
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.6,
            "axes.titlesize": 10.6,
            "axes.titleweight": "normal",
            "axes.labelsize": 9.6,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    by_layer_tok: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in records:
        by_layer_tok[(int(r["layer"]), int(r["tokens"]))] = r

    avail_layers = sorted({int(r["layer"]) for r in records})
    if curve_layer not in avail_layers:
        curve_layer = avail_layers[len(avail_layers) // 2]

    # Sized close to IEEE/ACM two-column textwidth (~7.0 in -> ~0.73x scale)
    fig = plt.figure(figsize=(9.6, 5.55))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.05, 1.05],
        hspace=0.74,
        left=0.068,
        right=0.992,
        top=0.855,
        bottom=0.095,
    )
    gs_top = gs[0].subgridspec(1, len(token_lengths), wspace=0.25)
    gs_bot = gs[1].subgridspec(1, 2, width_ratios=[1.36, 1.0], wspace=0.25)

    # Palette consistent with trilane_prefill_fig2
    col_layer = "#1D3557"
    col_cpu = "#577590"
    col_stall = "#F4A261"
    col_oracle = "#FFB703"
    col_trilane = "#D62828"
    col_apex = "#7209B7"
    col_local = "#2A9D8F"
    col_fixed = "#6C757D"

    def _short_tok(t: int) -> str:
        return f"{t // 1024}K" if t >= 1024 and t % 1024 == 0 else str(t)

    # ---------------- Panel (a): Candidate-n sweeps ----------------
    for idx, tok in enumerate(token_lengths):
        ax = fig.add_subplot(gs_top[0, idx])
        rec = by_layer_tok.get((curve_layer, int(tok)))
        if rec is None:
            # Fallback to any available layer at this token count
            cands = [r for r in records if int(r["tokens"]) == int(tok)]
            rec = cands[0] if cands else None
        if rec is None:
            ax.set_title(f"{tok} tokens (missing)")
            continue

        sweep = sorted(rec["sweep"], key=lambda p: int(p["n"]))
        ns = np.array([int(p["n"]) for p in sweep], dtype=np.float64)
        layer_ms = np.array([float(p["layer_ms"]) for p in sweep], dtype=np.float64)
        layer_std = np.array(
            [float(p.get("layer_std_ms", 0.0)) for p in sweep], dtype=np.float64
        )
        cpu_ms = np.array([float(p["cpu_ms"]) for p in sweep], dtype=np.float64)
        stall_ms = np.array(
            [float(p["gpu_exposed_h2d_wait_ms"]) for p in sweep], dtype=np.float64
        )

        ax.fill_between(
            ns,
            0.0,
            stall_ms,
            color=col_stall,
            alpha=0.35,
            zorder=2,
            label="Exposed H2D stall",
        )
        ax.plot(
            ns,
            stall_ms,
            color="#E76F51",
            linewidth=1.25,
            linestyle="-",
            zorder=2,
        )
        ax.plot(
            ns,
            cpu_ms,
            color=col_cpu,
            linewidth=1.6,
            linestyle="--",
            zorder=3,
            label="CPU compute",
        )
        ax.fill_between(
            ns,
            np.maximum(0.0, layer_ms - layer_std),
            layer_ms + layer_std,
            color=col_layer,
            alpha=0.12,
            zorder=3,
        )
        ax.plot(
            ns,
            layer_ms,
            color=col_layer,
            linewidth=2.1,
            zorder=4,
            label="Measured layer latency",
        )

        # Plot policy selections on their REAL measured hardware latency
        pols = rec["policies"]
        ora = rec["oracle"]

        def _mark(p_dict: Optional[Dict[str, Any]], marker: str, color: str, size: float, z: int, edge: str = "white"):
            if p_dict is None:
                return
            ax.scatter(
                [float(p_dict["n"])],
                [float(p_dict["real_layer_ms"])],
                marker=marker,
                s=size,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.95,
                zorder=z,
            )

        _mark(pols.get("fixed_64"), "s", col_fixed, 52, 6)
        _mark(pols.get("smoe_rule"), "D", col_local, 54, 7)
        _mark(pols.get("apex_2lane"), "^", col_apex, 68, 8)
        _mark(ora, "*", col_oracle, 175, 9, edge="#222222")
        _mark(pols.get("trilane"), "o", col_trilane, 62, 10)

        shown_pts_max = max(
            float(ora["real_layer_ms"]),
            *(
                float(pols[k]["real_layer_ms"])
                for k in ("fixed_64", "smoe_rule", "apex_2lane", "trilane")
                if k in pols
            ),
        )
        ymax = max(
            float(np.max(layer_ms)) * 1.34,
            shown_pts_max * 1.34,
            float(ora["real_layer_ms"]) * 1.95,
        )

        left_h = float(layer_ms[0])
        right_h = float(layer_ms[-1])
        if right_h > left_h * 1.15:
            box_x, box_ha = 0.04, "left"
        elif left_h > right_h * 1.25:
            box_x, box_ha = 0.96, "right"
        else:
            box_x, box_ha = 0.50, "center"

        tri = pols.get("trilane", {})
        ax.text(
            box_x,
            0.95,
            f"n*={ora['n']} ({ora['real_layer_ms']:.0f}ms)\n"
            f"Ours={tri.get('n', -1)} (+{tri.get('regret_pct', 0.0):.1f}%)",
            transform=ax.transAxes,
            ha=box_ha,
            va="top",
            fontsize=8.6,
            color="#1D1D1D",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="#BBBBBB",
                alpha=0.92,
            ),
            zorder=11,
        )

        ax.set_title(f"Layer {rec['layer']} @ {_short_tok(int(tok))}", fontsize=10.5, pad=5)
        ax.set_xlabel("Streamed prefix n", fontsize=9.5)
        if idx == 0:
            ax.set_ylabel("Latency / Stall (ms)", fontsize=9.5)
        ax.set_xlim(0, max(16.0, float(ns[-1]) * 1.04))
        ax.set_xticks([0, 80, 160, 240])
        ax.set_ylim(0.0, ymax)
        ax.grid(True, linestyle=":", linewidth=0.65, alpha=0.6)

    legend_handles_a = [
        Line2D([0], [0], color=col_layer, lw=2.2, label="Measured $T_{\\mathrm{real}}(n)$"),
        Line2D([0], [0], color=col_cpu, lw=1.6, ls="--", label="CPU compute"),
        Patch(facecolor=col_stall, alpha=0.45, edgecolor="#E76F51", label="Exposed H2D stall"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=col_oracle, markeredgecolor="#222", markersize=11.5, label="Oracle Best"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col_trilane, markersize=8.0, label="TriLane (Ours)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=col_apex, markersize=8.5, label="2-Path (Serial)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=col_local, markersize=7.5, label="SMOE-style"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=col_fixed, markersize=7.5, label="Fixed-64"),
    ]
    fig.legend(
        handles=legend_handles_a,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.992),
        ncol=4,
        fontsize=9.6,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.4,
        labelspacing=0.36,
    )

    # ---------------- Panel (b) Left: Step-wise Oracle Regret (%) ----------------
    ax_reg = fig.add_subplot(gs_bot[0, 0])
    ablation_specs = [
        ("fixed_64", "Fixed-64", "#8D99AE"),
        ("smoe_rule", "SMOE-style", "#2A9D8F"),
        ("apex_2lane", "2-Path (Serial)", "#7209B7"),
        ("trilane_no_prefetch", "w/o Prefetch", "#457B9D"),
        ("trilane", "TriLane (Ours)", "#D62828"),
    ]

    x_groups = [_short_tok(int(t)) for t in token_lengths] + ["Mean"]
    x_pos = np.arange(len(x_groups), dtype=np.float64)
    n_bars = len(ablation_specs)
    width = 0.82 / n_bars

    max_bar_val = 5.0
    for b_idx, (p_key, p_lbl, b_col) in enumerate(ablation_specs):
        vals: List[float] = []
        for tok in token_lengths:
            tok_vals = [
                float(r["policies"][p_key]["regret_pct"])
                for r in records
                if int(r["tokens"]) == int(tok) and p_key in r["policies"]
            ]
            vals.append(float(np.mean(tok_vals)) if tok_vals else 0.0)
        all_vals = [
            float(r["policies"][p_key]["regret_pct"])
            for r in records
            if p_key in r["policies"]
        ]
        vals.append(float(np.mean(all_vals)) if all_vals else 0.0)
        max_bar_val = max(max_bar_val, max(vals))

        offset = (b_idx - (n_bars - 1) / 2.0) * width
        bars = ax_reg.bar(
            x_pos + offset,
            vals,
            width=width * 0.92,
            color=b_col,
            edgecolor="#222222" if p_key == "trilane" else "none",
            linewidth=0.8 if p_key == "trilane" else 0.0,
            label=p_lbl,
            zorder=3,
        )
        if p_key == "trilane":
            for rect, v in zip(bars, vals):
                ax_reg.text(
                    rect.get_x() + rect.get_width() / 2.0 + 0.025,
                    rect.get_height() + max_bar_val * 0.02,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.6,
                    fontweight="bold",
                    color=col_trilane,
                )

    ax_reg.set_xticks(x_pos)
    ax_reg.set_xticklabels([f"{_short_tok(int(t))}\ntokens" for t in token_lengths] + ["Overall\nMean"], fontsize=9.3)
    ax_reg.set_ylabel("Suboptimality vs. Oracle (%)", fontsize=9.5)
    ax_reg.set_title("(b) Suboptimality Gap vs. Oracle Best", fontsize=10.6, fontweight="bold", pad=6)
    ax_reg.set_ylim(0.0, max_bar_val * 1.54)
    ax_reg.grid(True, axis="y", linestyle=":", linewidth=0.65, alpha=0.6, zorder=0)
    ax_reg.legend(ncol=3, frameon=True, loc="upper right", fontsize=8.8, columnspacing=0.75, handletextpad=0.35)

    # ---------------- Panel (c) Right: Diagnostic Mechanism Breakdown ----------------
    ax_diag = fig.add_subplot(gs_bot[0, 1])
    diag_specs = [
        ("fixed_64", "Fixed\n64"),
        ("smoe_rule", "SMOE\nstyle"),
        ("apex_2lane", "2-Path\nSerial"),
        ("trilane_no_prefetch", "w/o\nPref."),
        ("trilane", "TriLane\n(Ours)"),
        ("oracle", "Oracle\nBest"),
    ]

    d_labels: List[str] = []
    exposed_stalls: List[float] = []
    cpu_over_ora: List[float] = []

    for p_key, p_lbl in diag_specs:
        d_labels.append(p_lbl)
        st_arr: List[float] = []
        cpu_ex_arr: List[float] = []
        for r in records:
            ora = r["oracle"]
            p = ora if p_key == "oracle" else r["policies"].get(p_key)
            if p is None:
                continue
            st_arr.append(float(p["gpu_exposed_h2d_wait_ms"]))
            cpu_ex_arr.append(
                max(0.0, float(p["real_cpu_ms"]) - float(ora["real_cpu_ms"]))
            )
        exposed_stalls.append(float(np.mean(st_arr)) if st_arr else 0.0)
        cpu_over_ora.append(float(np.mean(cpu_ex_arr)) if cpu_ex_arr else 0.0)

    dx = np.arange(len(d_labels), dtype=np.float64)
    bw = 0.37
    ax_diag.bar(
        dx - bw / 2.0,
        cpu_over_ora,
        width=bw,
        color=col_cpu,
        label="Excess CPU tail (ms)",
        zorder=3,
    )
    ax_diag.bar(
        dx + bw / 2.0,
        exposed_stalls,
        width=bw,
        color=col_stall,
        edgecolor="#E76F51",
        linewidth=0.8,
        label="Exposed H2D stall (ms)",
        zorder=3,
    )
    diag_max = max(max(cpu_over_ora), max(exposed_stalls))
    ax_diag.set_ylim(0.0, diag_max * 1.44)
    ax_diag.set_xticks(dx)
    ax_diag.set_xticklabels(d_labels, fontsize=9.1)
    ax_diag.set_ylabel("Mean Penalty Time (ms)", fontsize=9.5)
    ax_diag.set_title("(c) CPU Tail vs. Exposed H2D Stall", fontsize=10.6, fontweight="bold", pad=6)
    ax_diag.grid(True, axis="y", linestyle=":", linewidth=0.65, alpha=0.6, zorder=0)
    ax_diag.legend(frameon=True, loc="upper right", fontsize=8.9)

    fig.text(
        0.52,
        0.478,
        "(a) Fixed-Backend Candidate Prefix Replay Sweep: Empirical Latency, CPU Tail & Exposed H2D Stall",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="bold",
    )

    out_pdf = f"{out_prefix}.pdf"
    out_png = f"{out_prefix}.png"
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[plot] saved publication figure -> {out_pdf} and {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and plot fixed-backend TriLane route-replay Oracle evaluation."
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "fig11"
            / "trilane_replay.jsonl"
        ),
        help="Path to replay sweep JSONL file.",
    )
    parser.add_argument(
        "--collect-url",
        type=str,
        default="",
        help="Optional running SGLang server URL (e.g. http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Tokenizer directory for slicing exact-length prompts.",
    )
    parser.add_argument(
        "--trigger-file",
        type=str,
        default="/tmp/trilane_replay.arm",
        help="Trigger file matching server's KT_TRILANE_REPLAY_TRIGGER_FILE.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=3,
        help="Same-distribution warmup runs immediately before each length's replay.",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="1024,2048,4096,8192",
        help="Comma-separated exact prefill token counts.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="3,32,60",
        help="Comma-separated layer indices (used by --synthetic-demo).",
    )
    parser.add_argument(
        "--curve-layer",
        type=int,
        default=60,
        help="Representative MoE layer shown in Panel (a) sweep curves.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "figures_out"
            / "fig11"
            / "trilane_route_replay"
        ),
        help="Output path prefix for .pdf, .png, and _summary.csv.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Generate synthetic replay data to preview charts without server.",
    )
    args = parser.parse_args()

    token_lengths = _parse_int_list(args.tokens)
    layers = _parse_int_list(args.layers)

    if args.synthetic_demo:
        generate_synthetic_demo_jsonl(args.jsonl, layers, token_lengths)
    elif args.collect_url:
        collect_replay_runs(
            server_url=args.collect_url,
            tokenizer_path=args.tokenizer,
            token_lengths=token_lengths,
            warmup_rounds=args.warmup_rounds,
            trigger_file=args.trigger_file or None,
            jsonl_path=args.jsonl,
        )

    records = load_replay_records(args.jsonl)
    summarize_and_export_csv(records, f"{args.out_prefix}_summary.csv")
    plot_replay_figure(
        records=records,
        curve_layer=args.curve_layer,
        token_lengths=token_lengths,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
