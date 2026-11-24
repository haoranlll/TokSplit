# Figure-generation code

Scripts are listed in the order of the final PDF. Each entry was checked against the caption, axis text, and reported numbers in *TokSplit: Workload-Aware CPU–GPU Inference for Large Quantized Mixture-of-Experts Models*. Figures 3–5 are schematic diagrams and have no measurement scripts.

For Figures 1, 6, 7, 8, and 10, the plotted series are stored in the script, and PDF and PNG files are written next to the script. For Figures 2, 9, and 11, the scripts read files in this repository and write to `figures_out/`.

Python with `matplotlib` and `numpy` is required. Run the commands from the repository root. Measurement definitions for each figure are in [README.md](README.md).

## Figure 1. CPU operator throughput under different token workloads

The manuscript uses `combined_operator_throughput.pdf`. The two panel titles are “(a) MoE operator throughput of KT-Kernel (Top-8)” and “(b) Single-expert FFN operator throughput”.

- Script: `code/fig01_cpu_operator_throughput.py`
- Plotted series: `data/fig01_cpu_operator_throughput.csv`

```bash
python code/fig01_cpu_operator_throughput.py
```

## Figure 2. Profiling SMOE’s prefill scheduling under different input lengths

The manuscript uses `smoe_load_and_stage_times.pdf`. The top row ranks layer-3 experts by load. The bottom row shows CPU computation, GPU computation, weight transfer, and layer time for layers 3, 32, and 60. The labels `GPU:79 CPU:161`, `139/111`, `192/62`, and `222/34` come from the activation CSVs below.

- Script: `code/fig02_smoe_prefill_profiling/fig02_smoe_prefill_profiling.py`
- The script imports `plot_expert_load_layer3.py` and `plot_four_metrics.py` from the same directory. Those two filenames must stay unchanged.
- Per-expert rows for panel (a): `data/smoe_layer_timing_cpu10/q{512,1024,2048,4096}_blk_3/activation_load.csv`
- Millisecond table for panel (b): the `DATA` dict in `plot_four_metrics.py`, also exported as `data/fig02b_stage_times_ms.csv`

The expert counts in (a) and (b) come from two recordings. See the Figure 2 section of [README.md](README.md).

```bash
python code/fig02_smoe_prefill_profiling/fig02_smoe_prefill_profiling.py
```

Output is written to `figures_out/fig02/`.

## Figure 6. End-to-end prefill and decode throughput

The manuscript uses `e2e_prefill_decode_2x4.pdf`. The top row is prefill and the bottom row is decode. The four columns are Mixtral-8×22B, Qwen3-235B, Qwen3-Coder-480B, and DeepSeek-R1.

- Script: `code/fig06_e2e_prefill_decode.py`
- Plotted series: `data/fig06_e2e_throughput.csv`

```bash
python code/fig06_e2e_prefill_decode.py
```

## Figure 7. Prefill throughput under 10- and 64-thread configurations

The manuscript uses one axes with a 10-thread block on the left, a 64-thread block on the right, and a dashed divider. The legend is KTransformers, SMOE, TokSplit-T, and TokSplit-T+L. That layout is Version D, saved as `prefill_thread_ablation_D_split.pdf`. The same script also draws Version C, two side-by-side panels whose legend omits KTransformers. Version C is not the published figure.

- Script: `code/fig07_prefill_thread_ablation.py`
- Plotted series: `data/fig07_prefill_thread_ablation.csv`
- Model: Q4_K_M DeepSeek-V3, distinct from the DeepSeek-R1 bars in Figure 6

```bash
python code/fig07_prefill_thread_ablation.py
```

## Figure 8. Normalized CPU MoE operator speed of LoadGear over KT-Kernel

The manuscript y-axis is Normalized Operator Speed. The left block is Mixtral-8×22B and the right block is DeepSeek-R1. The dashed line is the KT-Kernel baseline at 1.0×. The same figure is saved under two names: `loadgear_operator_speedup.pdf` and `loadgear_operator_throughput.pdf`.

- Script: `code/fig08_loadgear_vs_ktkernel.py`
- Plotted series: `data/fig08_loadgear_latency.csv` (latency in milliseconds; the bars are the ratio of the two latencies)

```bash
python code/fig08_loadgear_vs_ktkernel.py
```

## Figure 9. Sparse kernel performance across different token counts

The manuscript has two panels. Panel (a) is a line chart of three Down-projection layouts against dense KT-Kernel. Panel (b) is a bar chart of Dual-Layout against KAIROX, with y ticks at 0, 2, 4, and 6. The script draws both variants. The published panel (b) is the bar chart, `sparse_op_vs_kt_kairox.pdf`. `sparse_op_vs_kt_kairox_lines.pdf` replaces panel (b) with a line chart and is not the published figure.

- Script: `code/fig09_sparse_kernel.py`
- Plotted series: `data/fig09_sparse_kernel.csv`
- Panel (a) is recorded in the script comment as Qwen3-Coder-480B-A35B in Q4_K_M. Panel (b) is layer 10 of DeepSeek-R1 with 20% of neurons retained.

```bash
python code/fig09_sparse_kernel.py
```

Output is written to `figures_out/fig09/`.

## Figure 10. Ablation at 32 concurrent requests

The manuscript stacks two panels: (a) prefill and (b) decode. They come from two scripts. The `32k` in the filenames is 32 concurrent requests, each with a sequence of about 1000 tokens.

Panel (a), prefill, normalized to Base:

- Script: `code/fig10a_prefill_ablation_32req.py`
- Output: `prefill_component_ablation_32k.pdf`

Panel (b), decode, normalized to Base:

- Script: `code/fig10b_decode_ablation_32req.py`
- Output: `decode_component_ablation_32k.pdf`

Raw throughput and normalized speed for both panels are in `data/fig10_ablation_32req.csv`.

```bash
python code/fig10a_prefill_ablation_32req.py
python code/fig10b_decode_ablation_32req.py
```

## Figure 11. Fixed-backend route-replay evaluation

The manuscript uses `trilane_route_replay.pdf`. Panel (a) shows measured layer latency, CPU tail, and exposed H2D stall across candidate prefix lengths `n` on layer 60 at 1K–8K tokens. Panels (b) and (c) summarize layers 3, 32, and 60.

- Script: `code/fig11_trilane_route_replay.py`
- Sweep log: `data/fig11/trilane_replay.jsonl` (12 records: 3 layers by 4 prompt lengths)
- Per-case summary: `data/fig11_case_summary.csv`
- Mean gap by policy: `data/fig11/trilane_route_replay_summary.csv`

The JSONL has no measured policy named `smoe_rule`. On load, the script derives SMOE-style from `sweep` with a threshold of 22 tokens per added expert, then plots it with Fixed-64, 2-Path (Serial), TriLane w/o Prefetch, and TriLane. The published SMOE-style points are this derivation.

The default `--jsonl` points at the JSONL in this repository, and the default output is `figures_out/fig11/`. Without `--collect-url`, the command only redraws the figure. It does not contact a server and does not modify the JSONL.

```bash
python code/fig11_trilane_route_replay.py --curve-layer 60
```

The sweep was collected with layers 3, 32, and 60, stride 8, and 3 repeats per prefix.
