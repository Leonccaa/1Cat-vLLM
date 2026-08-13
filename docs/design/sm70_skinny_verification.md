# SM70 Skinny: verification status

Companion to `sm70_skinny_execution_layer.md`. Records what has been measured on real
V100 hardware and what has not, so the remaining gates are explicit rather than
assumed.

## Verified

Real `_C` build on CT252 (`llm252.lan`, 4x V100-PCIE-32GB). The extension was
built for `compute_70/sm_70` only, inside the
`1cat-vllm:1.2.2-cu128-responses-thinking-budget-tilelang010-skinny-v1` image
with the workspace on `/mnt/llm_hfs` (CT252's root filesystem is still at 96%).
48 targets, 10m10s wall on 32 jobs.

- `_C.abi3.so` SHA-256
  `fad4b2997fe8201253b10d48331a13a4e85e2cb82c7d0434d95e18464f89880f`
  (43,960,168 bytes). Note this is an `_C`-target-only build, so it is not
  comparable to a full release artifact's hash.
- All five skinny ops register: `skinny_awq_gemm_simt`, `skinny_awq_gemm_qpn`,
  `skinny_awq_moe_gemm_simt_out`, `skinny_nvfp4_gemm_simt`,
  `skinny_nvfp4_gemm_qpn`.
- GPU operator tests: **5 passed**.
- CPU tests run against the real build: **35 passed**.

Split-K, CUDA Graph and MoE verification on the same build
(`verify_splitk.py`, N=1792 K=5120 -> 4 splits):

| check | result |
| --- | --- |
| split-K vs independent FP32 reference | max_rel 3.635e-04 |
| 20 eager repeats bit-identical | yes (fixed split order is deterministic) |
| CUDA Graph capture + 10 replays vs eager | identical |
| replay tracks an updated input | yes (graph is live, not frozen) |
| non-split shape vs FP32 reference | max_rel 4.140e-04 |
| MoE invalid expert id | zeroed, no stale buffer contents |

Standalone kernel harness on the same host, see
`benchmarks/sm70_skinny_harness/`:

- Both SIMT and QPN compile clean for `sm_70` under CUDA 12.8.
- Numerics unchanged against a host FP64 reference on every shape and every M
  in {1,2,3,4,8,16}: max relative error <= 6.7e-4 against a 3e-2 tolerance,
  identical to the pre-change kernels.
- Split-K QPN: 1.91x–2.48x where it engages, exactly 1.00x where it does not.
- Two-translation-unit link test passes (this caught `qpn_reduce_kernel` being
  a duplicate symbol, which would have broken the real build).

Host-side, on the dev box:

- 35 CPU tests pass, including new ones that validate the FP32 references
  against an explicit per-element walk of the packed bytes.
- `ruff check` / `ruff format` clean on every touched file; `clang-format`
  applied to all CUDA sources; `compileall` and `git diff --check` clean.

## Full-model A/B (Qwen3.6-27B AWQ, TP4, four V100)

Frozen model `QuantTrio/Qwen3.6-27B-AWQ` rev `9b507bdc`, 4096-token input /
256 output, `max_num_seqs=3`, no MTP, `gpu_memory_utilization=0.87`,
`max_model_len=8192`, CUDA graphs on. Prefill and decode separated by timing a
`max_tokens=1` run against the full run.

Run-to-run precision is about **0.15%**: three independent loads of the gated
arm gave 65.79 / 65.87 / 65.84 tok/s, and prefill held 2340-2348 tok/s across
every arm. Differences below ~0.3% are not meaningful; differences above ~1%
are.

| arm | kernels | residency | decode tok/s | KV tokens | overlay GiB/card |
| --- | --- | --- | ---: | ---: | ---: |
| off | shipped | none | 58.28 | 992,987 | 0 |
| oldoff | baseline | none | 58.26 | 992,987 | 0 |
| gated | shipped | MIN_ROI=1.0 | 65.83 | 946,176 | 0.93 |
| full | shipped | all shapes | **68.73** | 853,138 | 2.86 |
| oldfull | baseline | all shapes | 68.56 | 853,138 | 2.86 |
| full+ldcs | with `__ldcs` | all shapes | 67.84 | 853,138 | 2.86 |

What this establishes:

- **The overlay is worth +17.9% decode** (58.28 -> 68.73) for 2.86 GiB/card.
  That 2.86 GiB, derived from the KV-cache delta against the full arm's
  reported 17.45 GiB / 853,138 tokens, matches the independent hand accounting
  of 2.82 GiB to within 1.5%.
- **The ROI gate returns 1.93 GiB/card** (2.86 -> 0.93) for 4.2% of decode
  (68.73 -> 65.83). It keeps 72% of the overlay's benefit for 33% of its
  memory.
- **This branch's kernels are level with the baseline on this workload**:
  68.73 vs 68.56, i.e. +0.25%, inside noise. That is the expected result and
  not a disappointment: the workload is single-request with no MTP, so M is
  always 1 and the split-K QPN path never executes. Its 1.9-2.5x applies to
  MTP and multi-sequence decode, which this A/B does not exercise and which
  remains unmeasured end to end.
- **`__ldcs` was reverted.** It scored +6-21% in the standalone harness and
  measured 1.3% *slower* here (67.84 vs 68.73). See that commit; the general
  lesson is recorded in the harness README.

### Measured ROI table (L2-cold timing)

Per layer instance, per rank; this is what the loader logs. Numbers from the
corrected gate - the earlier hot-loop timing understated the base backend's
cost by up to 40% on the large shapes.

| shape | role | base us | skinny us | overlay MiB | ROI us/MiB | at 1.0 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| N=5120 K=1536 | o_proj | 23.0 | 12.1 | 4.0 | 4.626 | keep |
| N=5120 K=4352 | down | 44.2 | 23.7 | 11.3 | 2.268 | keep |
| N=4096 K=5120 | linear-attn | 52.2 | 44.0 | 10.6 | 0.771 | drop |
| N=8704 K=5120 | gate_up | 66.6 | 56.3 | 22.6 | 0.408 | drop |

The bottom two shapes hold 68% of the overlay bytes. `MIN_ROI=1.0` sits in the
gap between the two clusters; `0.5` would additionally keep the linear-attn
projection at roughly 1.6 GiB/card if more decode is wanted.

### QPN at M=8 (batched decode)

The single-request A/B never leaves M=1, so it cannot see QPN at all. Driving
8 concurrent sequences (1024 in / 128 out, `max_num_seqs=8`, prefix caching
off, `layout=both`) puts the decode batch at M=7..8, which routes to QPN.

Batched decode is far noisier than single-request - about 5% run to run against
0.15% for M=1 - so this is three interleaved repeats per arm:

| arm | decode tok/s (3 reps) | mean |
| --- | --- | ---: |
| overlay off | 158.27 / 165.75 / 161.71 | 161.91 |
| QPN resident | 149.56 / 142.18 / 140.86 | 144.20 |

**QPN costs 10.9% of batched decode throughput.** The ranges do not overlap, so
this survives the noise.

That is the opposite of what the operator-level measurement says. The same
L2-cold timing the residency gate uses reports QPN *beating* the TurboMind base
at M=8 on these shapes:

| shape | base us | QPN us | speedup |
| --- | ---: | ---: | ---: |
| N=5120 K=1536 | 42.0 | 30.7 | 1.37x |
| N=4096 K=5120 | 55.3 | 50.2 | 1.10x |

So an isolated-kernel win of 1.1-1.4x turns into a 10.9% end-to-end loss. This
is the second time in this branch that a kernel-level measurement pointed the
wrong way (the first was `__ldcs`), and it is the more important instance,
because **the ROI gate is built on exactly this kind of isolated measurement**.
The gate scores QPN at roi 2.827 and 0.482 us/MiB and would happily keep it.

Consequences:

- `VLLM_SM70_SKINNY_AWQ_LAYOUT` stays `simt` by default. QPN residency must
  remain an explicit opt-in and must not be enabled on the strength of the ROI
  table alone.
- The ROI gate is trustworthy for *which SIMT shapes to keep* - that is a
  comparison between two ways of running the same M=1 call, and the full-model
  A/B confirms the SIMT overlay is worth +17.9%. It is not trustworthy as
  evidence for turning a whole route on.

### Why split-K did not help here

Split-K fixes a launch-geometry problem that this checkpoint does not have.
`qpn_choose_k_splits` targets ~2 blocks/SM on 80 SMs, and the grid is N/32
blocks, so a shape only splits when N < 5120:

| shape | tiles = N/32 | splits chosen |
| --- | ---: | ---: |
| N=5120 K=1536 | 160 | 1 |
| N=5120 K=4352 | 160 | 1 |
| N=8704 K=5120 | 272 | 1 |
| N=4096 K=5120 | 128 | 2 |

Three of the four AWQ shapes already fill the GPU. Measured end to end at M=8,
new kernels 147.34 tok/s versus base kernels 147.76 - no difference, as the
table predicts.

The harness shape that showed 1.91-2.48x was qkv at N=1792, and this checkpoint
excludes `self_attn.q_proj/k_proj/v_proj` from quantization entirely, so that
shape does not exist here. Split-K remains correct for deployments whose AWQ
set does contain small-N projections - higher TP degree, or a checkpoint that
quantizes qkv - but it buys nothing on this model.

### Not reproduced

At `gpu_memory_utilization=0.90` with `max_model_len=131072`, both the full and
the gated arm loaded and ran without OOM, so the prefill-workspace OOM recorded
against the earlier overlay did not reproduce in this configuration (no MTP, no
multimodal, `max_num_seqs=3`). The KV gain at 0.90 was still real: 1,302,288
tokens gated versus 1,178,881 full.

## Fixed along the way

`test_awq_overlay_dispatches_simt_qpn_then_delegates_base` fed CPU tensors to
`torch.ops.vllm.sm70_skinny_awq_linear`, which `direct_register_custom_op`
registers for CUDA and Meta only. It therefore passed on a CPU-only torch and
raised `NotImplementedError` on any machine with a GPU — confirmed by running
the unmodified base commit `e9264c2f` on CT252, which fails identically. The
one test covering the M=1/8/17 routing decision had consequently never
executed on GPU hardware. It now follows the available device.

## Not yet verified

1. **Residency policy on a real model.** Needs judgement rather than a
   pass/fail — see below.
2. **Full-model paired A/B** at the chosen threshold.
3. **Concurrency.** The GPU gate is single-stream; concurrent requests sharing
   the split-K workspace allocation have not been exercised.
4. **A full release build.** Only the `_C` target was built here; `_moe_C` and
   friends were reused from the existing image.

## Calibrating the residency threshold

`VLLM_SM70_SKINNY_MIN_ROI` decides which shapes keep their overlay. The default
of 0.0 keeps any shape with a measurable win, which is conservative and will
not by itself hit a memory target.

The intended workflow is to run once and read the table the loader logs:

```
SM70 Skinny AWQ residency summary (per distinct shape, per layer instance; ...)
  N=  5120 K=  1280  roi=   0.443us/MiB  saved=    1.4us  overlay=     3.2MiB  keep
  N=  5120 K=  4352  roi=   0.352us/MiB  saved=    3.8us  overlay=    10.8MiB  keep
  N=  8704 K=  5120  roi=   0.097us/MiB  saved=    2.2us  overlay=    22.6MiB  keep
  ...
  kept X MiB/layer-set, dropped Y MiB/layer-set
```

Multiply the per-shape MiB by the layer count to get the per-card figure, pick
a threshold that lands on the memory target, and re-run. Then run the paired
A/B at that setting — the ROI numbers are a ranking signal measured per
operator at M=1, not a prediction of end-to-end decode.

Worth measuring at the same time: whether recovering the overlay memory also
allows `gpu_memory_utilization` back to 0.90. The overlay previously forced it
to 0.87 to avoid prefill workspace OOM, so the true cost was the overlay bytes
plus roughly another 0.96 GiB of surrendered headroom.

## Known-good ordering for the gates

Run the split-K and graph checks before the residency calibration: a residency
policy that silently disables the overlay would make a broken split-K path
look fine. (Done in that order above.)
