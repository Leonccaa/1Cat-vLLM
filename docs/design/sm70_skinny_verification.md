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
  `460cb522f6d57ccfcf81e539f88be8673b92d1b74a8d0b9f00d44225c4b908ac`
  (43,951,976 bytes). This is the post-`__ldcs`-revert `_C`-target-only build,
  so it is not
  comparable to a full release artifact's hash.
- All five skinny ops register: `skinny_awq_gemm_simt`, `skinny_awq_gemm_qpn`,
  `skinny_awq_moe_gemm_simt_out`, `skinny_nvfp4_gemm_simt`,
  `skinny_nvfp4_gemm_qpn`.
- GPU operator tests: **5 passed**.
- CPU tests run against the real build: **43 passed**.

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

- 43 CPU tests pass, including the FP32-reference, TP-consensus, fail-closed,
  route-buffer release, and split-geometry policy checks.
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
- **This branch's M=1 kernel changes are level with the baseline on this
  workload**:
  68.73 vs 68.56, i.e. +0.25%, inside noise. That is the expected result and
  not a disappointment: the workload is single-request with no MTP, so M is
  always 1 and the split-K QPN path never executes. The separate batched gate
  below establishes that QPN does not help this checkpoint end to end either.
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

## NVFP4 + MTP3 regression after split-K closeout

The post-`__ldcs`-revert `_C` above was exercised with the real
`OptimizeLLM/Qwen3.5-122B-A10B-heretic-MTP-NVFP4` checkpoint, TP4, native
MTP3, FP8 KV, original TileLang 0.1.10 prefill, and full CUDA Graph capture.
`VLLM_SM70_QUANT_BACKEND=skinny` intentionally selected the compatibility
alias (`base=auto`, `skinny=on`), so correctness self-checks ran while the
performance residency gate was bypassed.

The loader reported the actual QPN geometry once per shape:

| shape | split-K |
| --- | ---: |
| N=512 K=3072 | 8 |
| N=3072 K=256 | 1 (inactive) |
| N=4608 K=3072 | 2 |
| N=3072 K=2048 | 2 |

All SIMT and QPN self-checks passed against the selected TurboMind Dense base;
the MoE layers independently selected Marlin. Graph capture exercised QPN at
M=4, 8 and 12 and SIMT at M=1. A fixed 4040-to-256 request completed at
3610.06 prefill tok/s and 78.03 decode tok/s, with 19.65 GiB reported model
residency per GPU, 6.95 GiB available KV cache, and 226,789 KV tokens at
`gpu_memory_utilization=0.90`. Fatal-log scan was empty. This is a regression
and integration gate, not a paired performance claim.

Evidence:
`/mnt/llm_hfs/skinny-closeout-20260813/logs/qwen35-122b-nvfp4-mtp3.log`.

## Qwen3.6-27B native ModelOpt correction (2026-08-13)

The NVIDIA checkpoint at revision `0893e160...` is mixed ModelOpt, not an
all-NVFP4 artifact: its quantization map contains 208 FP8 and 193
W4A16_NVFP4 projections. An earlier test re-quantized the FP8 projections to
NVFP4 as a loader workaround. That artifact is excluded from acceptance: its
fixed-token output mechanically repeated the input material, so its 94.44%
MTP acceptance and all derived performance comparisons were quality-invalid.

The corrected route loads the original checkpoint directly. The mixed-config
SM89 gate was lowered to the capability of its concrete SM70 layer routes, and
the ModelOpt FP8 adapter now reuses the existing TurboMind W8A16 kernel instead
of selecting an N=4120-incompatible FP8 Marlin repack. Full VLM loading,
TileLang 0.1.10 original prefill, compile/CUDA graph, encoder warmup, and the
fixed text request passed. An explicit image request remains a separate gate.

The paired contract below is TP4, FP16 activation, FP8 KV,
`max_model_len=8192`, `max_num_seqs=1`, 4096 input tokens, 256 forced output
tokens, one warmup, and three repeats. `auto` here records the pre-guard
implementation with `VLLM_SM70_SKINNY_MIN_ROI=1`; it is retained as evidence
of why NVFP4 was removed from default auto eligibility.

| Profile | Prefill tok/s | Decode tok/s | Total tok/s | Model GiB/GPU | KV GiB/GPU | MTP acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no MTP, Skinny off | 2333.65 | 53.70 | 669.16 | 5.33 | 20.89 | - |
| no MTP, pre-guard auto ROI=1 | 2344.66 | 59.72 | 723.29 | 9.87 | 16.38 | - |
| MTP4, Skinny off | 2258.66 | 67.39 | 777.49 | 5.89 | 20.32 | 36.67% |
| MTP4, pre-guard auto ROI=1 | 2260.17 | 67.54 | 776.60 | 10.43 | 15.81 | 36.08% |

Without MTP, the overlay gained 11.20% decode while adding 4.54 GiB/GPU of
model residency (+85.18%) and removing 21.55% of KV-token capacity. With
MTP4, it gained only 0.22% decode, lost 0.11% end-to-end throughput, and paid
the same 4.54 GiB/GPU. ROI=1 dropped both `lm_head` layouts, but retained SIMT
and QPN for both MLP shapes across all layers; a local ROI threshold therefore
did not enforce an acceptable global memory budget.

The direct-source output is a coherent answer rather than input echo. Three of
the four direct profiles produced the same deterministic completion; the
no-MTP base profile chose a different but coherent continuation. This is a
functional smoke, not a full model-quality acceptance. Evidence is under
`/mnt/llm_hfs/logs/sm70-nvfp4-27b-native-modelopt-roi-20260813/`.

After adding the memory guard, a final real-model `SKINNY=auto` run retained no
NVFP4 overlay: model residency returned to 5.33 GiB/GPU, available KV to
20.89 GiB/GPU, and the warmup enumerated three FP8 plus three FP4 TurboMind
base shapes. It measured 2338.93 prefill, 53.65 decode, and 669.08 total tok/s,
within +0.23%, -0.10%, and -0.01% of the off baseline. Its fatal scan was
empty. This is the accepted auto behavior until a bounded/one-copy overlay is
implemented.

## Frozen release matrix (2026-08-13)

The final closeout used one fixed serving contract on CT252: TP4, FP16 model
activations, FP8 KV, CUDA Graph enabled, `max_model_len=8192`,
`max_num_seqs=1`, a 4096-token prompt, 256 forced output tokens, one warmup,
and three measured repeats. Prefill is the prompt-token/TTFT proxy; decode
excludes TTFT. Every row used the same frozen checkpoint within its table.

The stages separate three independent changes:

- `stock`: installed 1Cat 1.2.2 image source, TileLang 0.1.9, and the issue
  #105 prefill workaround;
- `final-old`: source commit `99e8ac6be8` on that same image and workaround;
- `final-new`: the same source with the pinned TileLang/TVM-FFI 0.1.10 image
  and original FlashQLA prefill enabled.

There is no valid BF16-speed arm on V100. All rows deliberately use FP16. The
BF16-to-FP16 changes are compatibility boundaries that make the SM70 route
runnable; issue #105 was a separate TileLang header-generation failure. They
must not be presented as an isolated throughput step.

### Qwen3.6-27B AWQ

`QuantTrio/Qwen3.6-27B-AWQ` revision `9b507bdc...`; the selected base is
TurboMind. `ROI=1` means `VLLM_SM70_SKINNY_MIN_ROI=1.0`.

| Stage/profile | MTP4 | Prefill tok/s | Decode tok/s | Total tok/s | Model GiB/GPU | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stock / off | no | 2087.52 | 50.30 | 618.90 | 5.65 | - |
| final-old / off | no | 2086.26 | 50.43 | 619.92 | 5.65 | - |
| final-old / full | no | 2080.62 | 57.70 | 681.32 | 8.52 | - |
| final-old / ROI=1 | no | 2084.67 | 55.63 | 664.53 | 6.64 | - |
| final-new / off | no | 2346.40 | 50.38 | 639.32 | 5.65 | - |
| final-new / ROI=1 | no | 2347.12 | 55.61 | 687.44 | 6.64 | - |
| stock / off | yes | 1982.56 | 101.86 | 952.44 | 6.21 | 76.98% |
| final-old / off | yes | 1984.11 | 101.00 | 948.24 | 6.21 | 76.98% |
| final-old / full | yes | 1984.47 | 101.09 | 950.28 | 9.08 | 76.98% |
| final-old / ROI=1 | yes | 1981.90 | 101.90 | 952.47 | 7.20 | 76.98% |
| final-new / off | yes | 2223.87 | 101.31 | 999.46 | 6.21 | 76.98% |
| final-new / ROI=1 | yes | 2221.01 | 101.63 | 999.95 | 7.20 | 76.98% |

The source-backed off arm is transparent to stock: -0.06% prefill and +0.25%
decode. With no MTP, the full overlay adds 14.42% decode; ROI=1 adds 10.30%
for 0.99 GiB/GPU. Relative to full residency, ROI=1 returns 1.88 GiB/GPU for
3.60% decode. TileLang 0.1.10 adds 12.47% prefill without moving decode. The
final memory profile is +12.44% prefill, +10.55% decode, and +11.07% total
throughput relative to stock.

MTP4 itself doubles decode on this prompt (+102.50%). Once MTP is active, the
ROI=1 overlay changes decode by only +0.32% while retaining 0.99 GiB/GPU, so
it has no measured value in this single-request verifier profile. All 12 arms
and all three repeats produced one byte-identical completion hash.

### Qwen3.5-122B-A10B NVFP4 MoE

`OptimizeLLM/Qwen3.5-122B-A10B-heretic-MTP-NVFP4` revision `07b7c210...`.
Skinny wraps only Dense NVFP4 projections; routed experts independently remain
on Marlin. The selected Dense base is TurboMind.

| Stage/profile | MTP3 | Prefill tok/s | Decode tok/s | Total tok/s | Model GiB/GPU | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stock / off | no | 3384.68 | 36.11 | 526.16 | 18.00 | - |
| final-old / off | no | 3377.30 | 36.06 | 525.32 | 18.00 | - |
| final-old / auto | no | 3381.97 | 37.73 | 546.12 | 18.44 | - |
| final-new / off | no | 3940.69 | 35.97 | 535.38 | 18.00 | - |
| final-new / auto | no | 3946.11 | 37.81 | 559.20 | 18.44 | - |
| stock / off | yes | 2999.44 | 64.38 | 817.71 | 19.24 | 77.49% |
| final-old / off | yes | 2989.79 | 62.68 | 800.52 | 19.24 | 75.11% |
| final-old / auto | yes | 2991.23 | 64.32 | 815.85 | 19.67 | 76.50% |
| final-new / off | yes | 3425.20 | 62.53 | 826.08 | 19.24 | 75.11% |
| final-new / auto | yes | 3431.60 | 65.45 | 854.89 | 19.67 | 77.49% |

The no-MTP, same-output `final-old` pair isolates a 4.64% Dense-overlay decode
gain for 0.44 GiB/GPU. The final no-MTP combination is +16.59% prefill,
+4.70% decode, and +6.28% total throughput versus stock. With MTP3, the final
combination is +14.41% prefill and +1.65% decode versus the stock MTP3 arm at
the same acceptance and output hash. The direct final-new off/on pair is
+4.66% decode, but its acceptance also changes from 75.11% to 77.49%; that
number is an end-to-end prompt result, not a pure kernel attribution.

Each 122B arm was internally deterministic across its three repeats, but the
full matrix contains three completion hashes. The variants make the same
semantic analysis and first diverge after 356 or 447 output characters; they
are not byte-identical. This is a real numerical-path caveat, not request
noise. Kernel self-checks and fixed-length performance therefore remain
necessary but are not substitutes for task-level quality gates.

The final combination then passed a separate 262K acceptance at
`max_model_len=262144`, `gpu_memory_utilization=0.90`, and
`max_num_seqs=1`. It exposed 988,226 KV tokens, reported 19.67 GiB/GPU model
residency, used 29,590 MiB/GPU when ready, and peaked at 30,314 MiB/GPU. The
text arithmetic, required tool call, and image-count gates all passed. A
261,996-token prompt returned the exact marker with 262,024 total tokens;
TTFT was 1236.89 seconds, or 211.82 derived prefill tok/s. A post-request
smoke also passed. This final-new run used the original TileLang path and did
not carry the issue #105 workaround.

All successful runs have empty fatal scans. No final-new arm reproduced
`common.h:778` or a BF16-to-half error. Raw evidence and the generated CSV/JSON
summary are under
`/mnt/llm_hfs/logs/sm70-release-matrix-20260813/`.

### Source-backed bundle gate

One initial 122B final-source launch is intentionally excluded: a Git archive
shadowed the installed `vllm` package without carrying `_moe_C.abi3.so`, so
`_moe_C.topk_softmax` was absent. The model and Skinny dispatch were not the
cause. The repaired bundle contains `_C`, `_C_stable_libtorch`, `_moe_C`,
`cumem_allocator`, and `spinloop`; every measured source-backed launch checks
their pinned hashes and the required MoE/Skinny torch ops before loading a
model.

`benchmarks/verify_sm70_runtime_bundle.py` makes that file/hash/op gate
reusable. A source archive alone is not a release artifact; either build the
native extensions into it or copy them from the exact ABI-compatible image and
verify them inside that image.

## Fixed along the way

`test_awq_overlay_dispatches_simt_qpn_then_delegates_base` fed CPU tensors to
`torch.ops.vllm.sm70_skinny_awq_linear`, which `direct_register_custom_op`
registers for CUDA and Meta only. It therefore passed on a CPU-only torch and
raised `NotImplementedError` on any machine with a GPU — confirmed by running
the unmodified base commit `e9264c2f` on CT252, which fails identically. The
one test covering the M=1/8/17 routing decision had consequently never
executed on GPU hardware. It now follows the available device.

## Not yet verified

1. **A full release build.** Only the `_C` target was rebuilt here; `_moe_C`
   and friends were reused from the existing image.
2. **Model-general policy.** The measured AWQ residency threshold and negative
   QPN result apply to this Qwen3.6-27B checkpoint on TP4. A different TP degree
   or quantized-shape set must re-run the logged geometry and paired A/B gates.

## Calibrating the residency threshold

`VLLM_SM70_SKINNY_MIN_ROI` decides which shapes keep their overlay. The default
of 0.0 keeps any shape with a measurable win, which is conservative and will
not by itself hit a memory target.

For the frozen Qwen3.6-27B AWQ TP4 profile above, `MIN_ROI=1.0` is the measured
memory-oriented setting: 0.93 rather than 2.86 GiB/card of overlay, for 65.83
rather than 68.73 decode tok/s. It is a deployment profile, not a universal
default for other models or TP degrees.

The intended workflow is to run once and read the table the loader logs:

```text
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
