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
`max_tokens=1` run against the full run; min of 3 reps per arm.

| arm | kernels | residency | decode tok/s | e2e tok/s | KV cache tokens |
| --- | --- | --- | ---: | ---: | ---: |
| off | new | none | 58.28 | 710.83 | 992,987 |
| oldoff | base | none | 58.26 | 710.73 | 992,987 |
| roi | new | MIN_ROI=1.0 | 66.56 | 780.47 | 947,346 |
| full | new | all shapes | 67.84 | 790.45 | 853,138 |
| oldfull | base | all shapes | 68.56 | 795.89 | 853,138 |
| noldcs | new, no `__ldcs` | all shapes | 68.73 | 797.18 | 853,138 |

Prefill was 2342-2348 tok/s in every arm, confirming the overlay does not touch
it and that run-to-run precision is about 0.2%. The two `off` arms agreeing to
0.03% across separately built extensions is the control for cross-build
comparability.

What this says:

- **The overlay is worth +17.9% decode** (58.28 -> 68.73 with the shipped
  kernels), costing 2.86 GiB/card.
- **`__ldcs` cost 1.3%** and was reverted; see that commit.
- **Split-K contributed nothing here and was expected not to.** This workload
  is single-request with no MTP, so M is always 1 and QPN never runs. Its
  1.9-2.5x applies to MTP and multi-sequence decode, still unmeasured
  end to end.
- **The ROI gate returns 1.93 GiB/card for 1.9% of decode.** Dropping the two
  low-ROI shapes (68% of the overlay bytes) keeps 86.6% of the decode gain.

Overlay memory, derived from the KV-cache deltas against the full arm's
reported 17.45 GiB / 853,138 tokens:

| residency | overlay GiB/card | decode tok/s |
| --- | ---: | ---: |
| none | 0 | 58.28 |
| MIN_ROI=1.0 | 0.93 | 66.56 |
| all shapes | 2.86 | 67.84 |

The 2.86 GiB matches the independent hand accounting of 2.82 GiB to within
1.5%.

### Measured ROI table

Per layer instance, per rank. This is what the loader logs.

| shape | role | base us | skinny us | saved us | overlay MiB | ROI us/MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| N=5120 K=1536 | o_proj | 23.0 | 12.1 | 11.0 | 4.0 | 2.750 |
| N=5120 K=4352 | down | 44.2 | 23.7 | 20.5 | 11.3 | 1.814 |
| N=4096 K=5120 | linear-attn | 26.8 | 23.7 | 3.1 | 10.6 | 0.279 |
| N=8704 K=5120 | gate_up | 39.7 | 35.9 | 3.8 | 22.6 | 0.170 |

The bottom two shapes hold 68% of the overlay bytes and deliver 18% of the
saving. `MIN_ROI=1.0` sits in the gap between the clusters.

### Not reproduced

At `gpu_memory_utilization=0.90` with `max_model_len=131072`, both the full and
the gated arm loaded and ran without OOM, so the prefill-workspace OOM recorded
against the earlier overlay did not reproduce in this configuration (no MTP, no
multimodal, `max_num_seqs=3`). The KV-cache gain at 0.90 was still real:
1,302,288 tokens gated versus 1,178,881 full.

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
