# TokSplit paper measurements

This repository contains the throughput, latency, and accuracy measurements reported in *TokSplit: Workload-Aware CPU–GPU Inference for Large Quantized Mixture-of-Experts Models*, together with the scripts that draw Figures 1, 2, and 6–11.

Figures 3–5 are schematic diagrams of the execution flow, LoadGear, and DualSparse. The numbers behind Tables 1–4 and Figures 1, 2, and 6–11 are in `data/`. Plotting scripts, named by figure number, are in `code/`. [FIGURE_CODE.md](FIGURE_CODE.md) records which script produced the figure used in the manuscript.

The recorded series reproduce the speedups and figure annotations in the manuscript, including the `GPU:79 / CPU:161` labels in Figure 2 and the prefix choices and suboptimality gaps in Figure 11.

## Contents

```text
.
├── data/                          # CSV tables and the Figure 11 replay log
│   ├── table01_qwen35_uniform_sparsity.csv
│   ├── table02_model_configs.csv
│   ├── table03_deepseek_r1_sparsity.csv
│   ├── table04_qwen_accuracy.csv
│   ├── fig01_cpu_operator_throughput.csv
│   ├── fig02a_layer3_assignment_summary.csv
│   ├── fig02b_stage_times_ms.csv
│   ├── smoe_layer_timing_cpu10/   # per-expert rows for Figure 2(a)
│   ├── fig06_e2e_throughput.csv
│   ├── fig07_prefill_thread_ablation.csv
│   ├── fig08_loadgear_latency.csv
│   ├── fig09_sparse_kernel.csv
│   ├── fig10_ablation_32req.csv
│   ├── fig11_case_summary.csv
│   └── fig11/
│       ├── trilane_replay.jsonl   # 12 frozen-state prefix sweeps
│       └── trilane_route_replay_summary.csv
├── code/                          # plotting scripts, one entry per figure
└── FIGURE_CODE.md
```

| File | Manuscript item |
|---|---|
| `data/table01_qwen35_uniform_sparsity.csv` | Table 1 |
| `data/table02_model_configs.csv` | Table 2 |
| `data/table03_deepseek_r1_sparsity.csv` | Table 3 |
| `data/table04_qwen_accuracy.csv` | Table 4 |
| `data/fig01_cpu_operator_throughput.csv` | Figure 1 |
| `data/fig02a_layer3_assignment_summary.csv` | Figure 2(a), expert counts and token share |
| `data/smoe_layer_timing_cpu10/q*_blk_3/activation_load.csv` | Figure 2(a), one row per expert |
| `data/fig02b_stage_times_ms.csv` | Figure 2(b) |
| `data/fig06_e2e_throughput.csv` | Figure 6 |
| `data/fig07_prefill_thread_ablation.csv` | Figure 7 |
| `data/fig08_loadgear_latency.csv` | Figure 8 |
| `data/fig09_sparse_kernel.csv` | Figure 9 |
| `data/fig10_ablation_32req.csv` | Figure 10 |
| `data/fig11_case_summary.csv` | Figure 11, one row per case |
| `data/fig11/trilane_replay.jsonl` | Figure 11, full prefix sweep |
| `data/fig11/trilane_route_replay_summary.csv` | Figure 11, mean gap by policy |

## Requirements

- Python 3
- `numpy`
- `matplotlib`

Run the commands from the repository root.

## Reproduce the figures

```bash
python code/fig01_cpu_operator_throughput.py
python code/fig02_smoe_prefill_profiling/fig02_smoe_prefill_profiling.py
python code/fig06_e2e_prefill_decode.py
python code/fig07_prefill_thread_ablation.py
python code/fig08_loadgear_vs_ktkernel.py
python code/fig09_sparse_kernel.py
python code/fig10a_prefill_ablation_32req.py
python code/fig10b_decode_ablation_32req.py
python code/fig11_trilane_route_replay.py --curve-layer 60
```

Figures 1, 6, 7, 8, and 10 write PDF and PNG files next to their scripts. Figures 2, 9, and 11 write to `figures_out/`. The Figure 11 command only redraws the stored sweep. It does not contact a server.

Figure 7 also writes a two-panel variant, `prefill_thread_ablation_C_1x2.pdf`. The manuscript uses the single-axes Version D, `prefill_thread_ablation_D_split.pdf`, which includes KTransformers. Figure 9 also writes a line-chart variant of panel (b). The manuscript uses the bar chart, `sparse_op_vs_kt_kairox.pdf`.

## Experimental setup

Unless a section below says otherwise, measurements use one server:

- Two Intel Xeon Platinum 8373C CPUs, about 350 GB/s aggregate sustained memory bandwidth, 502 GB DDR4
- One NVIDIA RTX A6000 (48 GB) on PCIe 4.0 x16, 32 GB/s theoretical unidirectional bandwidth
- 64 CPU worker threads by default, split evenly across the two NUMA nodes and bound to local cores and local memory
- Q4_K_M GGUF weights
- End-to-end output length fixed at 512 tokens per request

Prefill throughput is the number of input tokens divided by prefill time, in tokens/s. For one request, prefill time is the time from request submission until the first output token (TTFT). Decode throughput is the number of tokens generated during decode divided by the wall-clock time of that phase, in tokens/s. Operator microbenchmarks use the same unit. Raw latencies are in milliseconds. A speedup is a ratio of the two recorded values.

Table 2 gives the model shapes. `T/A` is total/active parameters in billions. `E/K` is routed experts per layer and Top-K. `H/I` is the hidden size and the expert intermediate size.

| Model | T/A (B) | MoE layers | E/K | H/I |
|---|---:|---:|---:|---:|
| DeepSeek-R1 | 671/37 | 58 | 256/8 | 7168/2048 |
| DeepSeek-V3 | 671/37 | 58 | 256/8 | 7168/2048 |
| Qwen3-Coder-480B | 480/35 | 62 | 160/8 | 6144/2560 |
| Qwen3-235B | 235/22 | 94 | 128/8 | 4096/1536 |
| Mixtral-8×22B | 140.6/39.1 | 56 | 8/2 | 6144/16384 |

For the three Top-8 models (Qwen3-235B, Qwen3-Coder-480B, and DeepSeek-R1), TokSplit defers 4 experts and applies 80% neuron sparsity to those deferred experts. DeepSeek-R1 stores Down-projection weights in native-only mode. Qwen3-235B and Qwen3-Coder use dual-layout. Mixtral-8×22B is Top-2 and does not use this dual-granularity sparsity. 

## Tables 1–4

Table 1 reports Qwen3.5-35B-A3B under dense execution and under a uniform policy that retains 40% of the neurons in every expert. `delta_pp` is the change from dense execution in percentage points. Uniform sparsity lowers GSM8K by 17.31 points and MBPP by 5.80 points.

Table 3 reports DeepSeek-R1. `base_percent` is the score with expert deferral disabled. The other columns are Defer-4, which defers 4 experts per token, at neuron-sparsity levels from 0% to 90%. A sparsity of 0% still executes the deferred experts densely. Absolute scores at 40–90% come from the evaluation sheet. The 0% absolute score is Base plus the percentage-point delta printed in the paper. At 80% sparsity, every benchmark stays within 0.59 points of Base. At 90%, HumanEval drops by 2.44 points. The default setting is therefore Defer-4 with 80% neuron sparsity (IA-80%).

Table 4 applies that setting to Qwen3-235B-A22B and Qwen3-Coder, with columns for Base, Defer-4, and Defer-4+IA-80%. The largest drop from Base is GSM8K on Qwen3-235B, from 92.80% to 92.34% (0.46 points). Scores use the official `lm_eval` task definitions with at most 16 concurrent requests, so autoregressive decoding takes the DualSparse path. MMLU, HellaSwag, WinoGrande, ARC-Challenge, and ARC-Easy are likelihood multiple-choice tasks. GSM8K, HumanEval, and MBPP are generative.

## Figure 1

CPU operator throughput against the number of input tokens, on DeepSeek-R1 quantized with Q4_K_M.

- (a) KT-Kernel Top-8 MoE operator, in tokens/s. At 512 tokens the throughput is 7069.8 tokens/s, which the text reports as exceeding 7,000 tokens/s.
- (b) A single-expert FFN. The two curves are the llama.cpp dense kernel and the KAIROX kernel at 80% neuron sparsity (20% of neurons retained). The array name `kairox20` in the script is that 0.2 retention ratio.

The token axis is 1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096. The CSV stores both latency in milliseconds and tokens/s. Throughput is `tokens / latency_ms × 1000`.

The text also states that workloads below 32 input tokens sustain 280–306 GB/s, or 80–87% of a measured 350 GB/s sequential-read bandwidth, and that the Gate, Up, and Down projections together account for more than 90% of execution time. Those two statements are not stored as a separate trace. Figure 1 plots throughput.

## Figure 2

SMOE prefill scheduling in its original 10-thread configuration. Prompt lengths are 512, 1024, 2048, and 4096. Stage times cover layers 3, 32, and 60. Times are in milliseconds.

Panel (a) ranks layer-3 experts by load. The annotations on the figure match `activation_load.csv`:

| Prompt tokens | GPU experts | CPU experts | GPU share of routed tokens |
|---:|---:|---:|---:|
| 512 | 79 | 161 | 66.6% |
| 1024 | 139 | 111 | 87.1% |
| 2048 | 192 | 62 | 96.3% |
| 4096 | 222 | 34 | 98.9% |

The text summarizes the last column as rising from 67% to 99%. Per-expert rows are under `data/smoe_layer_timing_cpu10/`.

Panel (b) uses the stage times hardcoded in the plotting script: CPU computation, GPU computation, weight transfer (host pack plus H2D), and layer wall time. At 512 tokens on layer 3 these are 175.87, 42.31, 129.24, and 184.66 ms. All 12 rows are in `fig02b_stage_times_ms.csv`.

The expert counts in (a) and the milliseconds in (b) come from two recordings. The millisecond table matches the primary timing summary of that stage-time run, whose layer-3 count at 512 tokens is 82 GPU experts and 163 CPU experts. The published annotations follow the activation CSV. The stage-time curves follow `fig02b_stage_times_ms.csv`.

## Figure 6

End-to-end prefill and decode throughput for four models at batch sizes 1, 2, 4, 8, 16, and 32, in tokens/s. The systems are llama.cpp, KTransformers, and TokSplit. `fig06_e2e_throughput.csv` has all 48 rows and the TokSplit speedups against the other two systems.

Those rows give the ranges stated in the text:

- Prefill is 1.35×–3.47× over llama.cpp and 1.52×–5.16× over KTransformers. The 5.16× point is Qwen3-235B at batch 32: 971.48 / 188.27.
- On the three Top-8 models, decode is 1.68×–3.39× over llama.cpp and 1.05×–2.14× over KTransformers. The 3.39× point is Qwen3-235B at batch 32: 156.87 / 46.26. The 2.14× point is Qwen3-Coder-480B at batch 8: 43.58 / 20.41.
- Mixtral decode is 1.04×–1.46× over llama.cpp and 1.02×–1.37× over KTransformers at batch sizes 1–16. At batch 32 the three systems are close: 37.86, 37.91, and 37.90 tokens/s.

For each Top-8 model, the KTransformers decode bar is the faster of the 4-expert and 6-expert deferral settings. Mixtral decode for KTransformers does not use that sparsity.

## Figure 7

Prefill throughput of Q4_K_M DeepSeek-V3, in tokens/s. This is DeepSeek-V3, distinct from the DeepSeek-R1 bars in Figure 6. Prompt lengths are 1K, 2K, 4K, 8K, and 16K. The left block uses 10 threads and the right block uses 64 threads. The four series are KTransformers, SMOE, TokSplit-T (TriLane only), and TokSplit-T+L (TriLane and LoadGear).

The table gives the ranges in the text:

- With 10 threads, TokSplit-T and TokSplit-T+L reach 1.95× and 2.25× over SMOE.
- With 64 threads, those peaks are 3.02× and 4.30×.
- Enabling LoadGear on top of TriLane adds up to 1.17× at 10 threads and 1.66× at 64 threads.
- Against KTransformers, the speedups are 2.96×–12.32× at 10 threads and 1.87×–3.80× at 64 threads.

## Figure 8

Normalized operator speed of LoadGear over KT-Kernel. The KT-Kernel baseline is 1.0. The models are dense experts of Mixtral-8×22B and DeepSeek-R1. Each point is one MoE layer, 64 threads, 20 warm-up runs, and the mean of 50 timed runs. The token axis is the powers of two from 1 to 16384. The CSV stores latency in milliseconds. The bar height is `KT-Kernel latency / LoadGear latency`.

At 1–8 tokens, Mixtral is 1.12×–2.16× and DeepSeek-R1 is 1.16×–1.25×. The largest later speedups are 4.24× on Mixtral at 8192 tokens (1890.349 / 446.0599 ms) and 3.10× on DeepSeek-R1 at 16384 tokens (2151.462 / 693.0815 ms).

The text describes intermediate token counts as 0.98×–1.07×. On this table, DeepSeek-R1 at 16–256 tokens is 1.00×–1.07×. Mixtral at 32 tokens is 0.95× (8.885 / 9.3248 ms). The figure is drawn from these latencies.

## Figure 9

Sparse-kernel speed, 64 threads. Each panel is normalized to its own baseline of 1.0.

Panel (a) is recorded in the script as Qwen3-Coder-480B-A35B-Instruct in Q4_K_M. All three Down-projection layouts are compared with dense KT-Kernel. At 8 tokens, Transpose-Only is 2.31× (7.0796 / 3.0622 ms) and Dual-Layout is 2.32× (7.0796 / 3.0581 ms). From 1024 tokens onward, Native-Only is the fastest of the three, reaching 2.58× at 4096 tokens (554.5536 / 215.0228 ms).

Panel (b) is layer 10 of DeepSeek-R1. Dual-Layout is compared with KAIROX-Kernel. Both retain 20% of the intermediate neurons, and the exact KAIROX mask is built outside the timed region. The speedup runs from 1.72× to 6.73×, summarized in the text as about 1.7×–6.8×. The published panel (b) is the bar chart, with y ticks at 0, 2, 4, and 6.

## Figure 10

Ablation at 32 concurrent requests, each with a sequence of about 1000 tokens. The `32k` in the script filename is this concurrency, not a 32K-token sequence. The y-axis is speed normalized to Base. Base is KTransformers with Expert Deferral disabled.

Panel (a) is prefill. `T` is TriLane and `L` is LoadGear. Values are tokens/s, with the normalized speed in parentheses.

| Model | Base | Base+T | Base+L | Base+T+L |
|---|---:|---:|---:|---:|
| Qwen3-235B | 188.27 (1.00×) | 748.13 (3.97×) | 542.34 (2.88×) | 971.48 (5.16×) |
| Qwen3-Coder-480B | 113.60 (1.00×) | 477.74 (4.21×) | 345.43 (3.04×) | 582.48 (5.13×) |
| DeepSeek-R1 | 110.44 (1.00×) | 174.40 (1.58×) | 248.17 (2.25×) | 312.61 (2.83×) |

Base and Base+T+L match the batch-32 KTransformers and TokSplit prefill bars in Figure 6.

Panel (b) is decode. `ED` is Expert Deferral and `DS` is dual-granularity sparsity.

| Model | Base | Base+ED | Base+ED+DS |
|---|---:|---:|---:|
| Qwen3-235B | 54.59 (1.00×) | 100.09 (1.83×) | 156.87 (2.87×) |
| Qwen3-Coder-480B | 26.14 (1.00×) | 38.95 (1.49×) | 59.24 (2.27×) |
| DeepSeek-R1 | 28.94 (1.00×) | 31.85 (1.10×) | 37.85 (1.31×) |

Base+ED+DS matches the batch-32 TokSplit decode bars in Figure 6. The KTransformers decode bars in Figure 6 use Expert Deferral, so they sit next to Base+ED. Qwen3-235B Base+ED and the Figure 6 KTransformers bar are both 100.09 tokens/s. Qwen3-Coder is 38.95 against 39.24, and DeepSeek-R1 is 31.85 against 31.91.

## Figure 11

Fixed-backend route replay on DeepSeek-R1 with 64 threads. At layers 3, 32, and 60, and at prompt lengths 1K, 2K, 4K, and 8K, the run freezes the routing, the resident experts, and the prefetched experts that have already landed, then sweeps the streamed prefix length `n` on that same backend. There are 12 cases. Each case stores its sweep in `sweep` inside `trilane_replay.jsonl`. Each prefix is repeated 3 times.

Oracle Best is the prefix with the smallest measured layer latency `T_real(n)`. The gap is how much slower the chosen prefix is than that minimum. Panel (a) shows layer 60:

| Prompt | Oracle `n*` | Oracle layer latency | TriLane `n` | Gap |
|---:|---:|---:|---:|---:|
| 1K | 46 | 68.4 ms | 50 | +5.0% |
| 2K | 79 | 88.3 ms | 77 | +1.7% |
| 4K | 110 | 120.6 ms | 108 | +2.5% |
| 8K | 127 | 200.8 ms | 129 | +1.0% |

The text rounds those four latencies to 68, 88, 121, and 201 ms. Across all 12 cases the mean TriLane gap is 6.1%. By prompt length the means are 5.4%, 6.0%, 8.5%, and 4.5%. Disabling prefetch-ready priority raises the mean gap to 9.2%. The mean gaps of 2-Path (Serial), Fixed-64, and SMOE-style are 38.4%, 62.9%, and 92.0%.

Panel (c) averages two penalties over the 12 cases: CPU time above the Oracle CPU time, and GPU H2D wait that is not hidden. Excess CPU tail is 98.0, 57.8, 0.0, 5.2, and 0 ms for Fixed-64, 2-Path, SMOE-style, TriLane, and Oracle. Exposed H2D stall is 93.4, 22.3, and 24.8 ms for SMOE-style, TriLane, and Oracle.

SMOE-style is derived when the figure is drawn. It is not a policy key stored by the server. Along the measured sweep, the prefix keeps growing while each added expert carries at least 22 tokens on average, and it stops after passing the experts that have already landed. The same kind of per-expert threshold is described as about 20 tokens in Figure 2. Figure 11 uses 22. The derived choices are the `smoe_style_*` columns of `fig11_case_summary.csv` and the SMOE-style row of `trilane_route_replay_summary.csv`.
