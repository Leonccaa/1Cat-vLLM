# SM70 Skinny execution overlay

## Purpose

SM70 Skinny is a small-`M` execution overlay for Volta. It is neither a new
quantization format nor a replacement for 1Cat-vLLM's linear-backend
selection. It keeps weight loading, model configuration, tensor parallelism,
attention, scheduling, and the selected base kernel in 1Cat-vLLM, then
overlays a format policy on shared SIMT and QPN execution cores.

The initial formats are:

- NVFP4 group 16, using E2M1 decode and FP8 E4M3 group scales;
- asymmetric AWQ uint4 group 128, using FP16 scales and precomputed
  `-zero * scale` biases.

## C++ boundary

`csrc/quantization/sm70_skinny/` contains three layers:

1. `common.cuh`, `simt.cuh`, and `qpn.cuh` own Volta scheduling, activation
   staging/reuse, accumulation, reduction, and output stores.
2. `formats/*.cuh` own packed-code addressing, metadata cadence, register
   decode, and fragment order.
3. Thin operator files validate tensors and choose template instantiations.

The SIMT policy contract is `Params`, `ThreadState`, `Segment`,
`make_thread_state`, `stage_pairs`, `load_segment`, and `decode_word`. The QPN
policy supplies `Params`, `ThreadState`, `make_thread_state`, and
`load_fragment`. Grouped MoE reuses the SIMT policy and adds only
`select_expert`.

This boundary is intentional: another 4-bit format should add a format policy
and prepack, not fork the scheduling core.

## Configuration, dispatch, and fallback

The two controls are deliberately orthogonal:

- `VLLM_SM70_QUANT_BACKEND=auto|turbomind|marlin` retains its existing role as
  the SM70 quantized base-backend selector;
- `VLLM_SM70_SKINNY=auto|on|off` controls only the small-`M` overlay.

Both default to `auto`. On exact SM70, `SKINNY=auto` enables only a validated,
memory-accepted Dense format/shape/operator route. Dense AWQ currently meets
that contract. Dense NVFP4 does not: a real Qwen3.6-27B TP4 run retained
4.54 GiB/card even at `MIN_ROI=1`, so NVFP4 stays on the selected base in
`auto` and requires explicit `on` for research. `on` is a fail-closed test
mode: correctness self-checks still run, but a failure aborts instead of
silently falling back and the performance residency gate is bypassed. `off`
is the one-variable rollback to the unmodified selected base. The historical
`VLLM_SM70_QUANT_BACKEND=skinny` spelling remains a compatibility alias for
base `auto` plus `VLLM_SM70_SKINNY=on`, but new launch scripts should use the
two controls above.

| Format/path | Rows | Route |
| --- | ---: | --- |
| Dense NVFP4, explicit `on` | 1-3 | Skinny SIMT |
| Dense NVFP4, explicit `on` | 4-16 | Skinny QPN |
| Dense NVFP4, `auto` or rows 17+ | any | selected base backend |
| Dense AWQ, `auto`, SIMT resident | 1-2 | Skinny SIMT |
| Dense AWQ, explicit `on`, SIMT resident | 1-3 | Skinny SIMT |
| Dense AWQ, QPN resident | 4-16 | Skinny QPN |
| Dense AWQ, `auto` | M=3 or unsupported shape/rows | selected base backend |
| Grouped AWQ MoE prototype | routed rows | Skinny grouped SIMT |

Dense AWQ accepts `VLLM_SM70_SKINNY_AWQ_LAYOUT=simt|qpn|both`:

- `simt` is the default and adds one packed-weight-sized layout for ordinary
  decode while retaining the selected base for verification and prefill;
- `qpn` keeps only the QPN overlay and targets `M=4..16` verification;
- `both` deliberately pays for both overlays and is intended for MTP research.

Each distinct dense shape and Skinny route is compared once with the
same-weight selected base result. A failed route is disabled only for that
layer state; it does not disable other shapes or remove the base fallback.
BF16 activations and checkpoint scale tensors are explicitly converted to
FP16 at the Volta kernel boundary and converted back at the public output
boundary.

The Dense overlay does not alter the MoE selector. Unsupported formats,
non-SM70 devices, MoE layers, unsupported shapes, and larger batches keep
their normal selected route.

## Pre-one-copy boundary audit (2026-08-14 UTC)

The stage-2 boundary pass found one independent graph fix and a hard limit of
load-time kernel A/B:

- Plain `max_num_seqs=3` had no exact M=3 CUDA Graph. Adding only capture size
  3 raised the Qwen3.6-27B AWQ TP4 base from 41.64 to 80.47 aggregate decode
  tok/s. With ROI=1 Skinny it reached 81.62 tok/s. This was primarily a graph
  cliff, not a Skinny gain. The finalized policy uses a logarithmic capture
  set plus the exact configured maximum: max 8 becomes `[1,2,3,4,8]`, max 16
  becomes `[1,2,3,4,8,16]`, and non-power-of-two maxima are included exactly.
  Intermediate request counts pad upward to a captured graph instead of
  silently falling into the same dynamic-path cliff. Only B=3 has a measured
  end-to-end speedup; the larger sizes are a coverage fix, not a performance
  claim.
- A same-source MTP2 causal probe measured 79.91 tok/s with Skinny off, 78.58
  with SIMT at M=1..3, and 79.96 when only M=3 fell back to the selected base.
  All three arms had the same 86.17% speculative acceptance. M=3 routing was
  therefore causal for this regression.
- A proposed automatic exact-M classifier was then tested and rejected. Its
  paired L2-cold CUDA Graph microbench marked both resident M=3 shapes as
  faster under Skinny (1.045x and 1.282x), yet the resulting full-model MTP2
  run fell further to 77.78 tok/s. More decisively, exact-graph plain B=3 was
  `+1.43%` while exact-graph MTP2 at the same M=3 was negative. The deciding
  variable is graph context, not M or shape, and graph context is unavailable
  to a load-time residency gate. The prototype was removed instead of tuning
  a threshold around one checkpoint.

Cross-session throughput drift in these arms was about 1%. The P1 prototype
and the plain M=1..3 arm had equivalent routing but differed by another 1%.
Therefore a future decision at the 1-2% scale must state its repeat count and
use at least three independent serving sessions per arm; repeated requests in
one server lifetime do not establish a cross-session result.

Consequently, `auto` still measures only per-shape layout residency and uses a
conservative static route boundary: SIMT M=1..2, M=3 selected base. Explicit
`VLLM_SM70_SKINNY=on` remains the research opt-in for the kernel's M=3
capability. Route logging includes N and K so every observed `(M,N,K)` is
auditable. A one-copy solo mode is deferred: removing the base layout would
also remove the proven M=3 escape path for any profile that can reach it. No
new classifier or kernel work is planned under this project scope. Evidence is
under `/mnt/llm_hfs/skinny-stage2-preonecopy-20260814T0324Z/`.

## Grouped MoE memory policy

`VLLM_SM70_SKINNY_AWQ_MOE=1` independently enables an experimental
replacement-layout AWQ MoE path. It converts W13 and W2 expert banks to one
N-major Skinny layout and releases the native checkpoint tensors; it does not
retain a second full base-backend expert bank. Routing, permutation, SiLU,
weighted reduction, and expert-parallel metadata remain owned by 1Cat-vLLM.

This route is default-off because one-copy residency also means there is no
large-`M` base fallback for those expert banks. It must pass model-level
prefill, decode, quality, and memory gates independently before production
use; enabling the Dense overlay never enables it.

## Validation

CPU reference and dispatch tests:

```bash
.venv/bin/python -m pytest --confcutdir=tests/quantization \
  tests/quantization/test_sm70_skinny_awq.py \
  tests/quantization/test_sm70_skinny_nvfp4.py -q
```

Exact-SM70 operator tests:

```bash
.venv/bin/python -m pytest -q \
  tests/quantization/test_sm70_skinny_awq_gpu.py \
  tests/quantization/test_sm70_skinny_nvfp4_gpu.py
```

Qwen3.6-27B TP4-shape synthetic A/B:

```bash
.venv/bin/python benchmarks/benchmark_sm70_skinny_awq.py \
  --layout both --json-out /path/to/result.json
```

The benchmark's GB/s field is logical packed-code plus metadata bytes divided
by CUDA-event time. It is an effective weight-consumption metric, not a DRAM
hardware-counter measurement.

## V100 acceptance snapshot

The 2026-08-12 CT252 gate used four PCIe V100 32 GB GPUs with TP4. Evidence is
under
`/mnt/llm_hfs/logs/sm70-skinny-core/20260812T235000Z`.

### Dense AWQ

The real-model gate used `QuantTrio/Qwen3.6-27B-AWQ` at frozen revision
`9b507bdc...`, asymmetric group-128 AWQ, CUDA Graph enabled, and the production
TurboMind base. Three cold fixed-token 4096-to-256 requests produced these
medians:

| Profile | Prefill tok/s | Decode tok/s | End-to-end tok/s | Model load / GPU |
| --- | ---: | ---: | ---: | ---: |
| TurboMind, Skinny off | 1,997.11 | 58.15 | 678.02 | 5.65 GiB |
| TurboMind + Skinny SIMT | 1,980.13 | 68.21 | 752.14 | 8.50 GiB |

This is a 17.3% decode increase with a 0.85% prefill decrease. The overlay run
used `gpu_memory_utilization=0.87`; `0.90` left insufficient prefill workspace.
`max_num_seqs=3` passed for the no-MTP profile, including arithmetic and image
requests.

With native MTP4, 4096-to-256 decode was effectively flat: 93.17 tok/s for
TurboMind versus 92.60 tok/s for the `both` SIMT+QPN overlay. The verifier did
use QPN at runtime, but its extra layout increased model residency from 6.69
to 12.35 GiB per GPU without an end-to-end gain. Therefore AWQ defaults to the
SIMT layout; `qpn` and `both` remain research opt-ins.

The current 1Cat dynamic-vocabulary GPU-LRU MTP path requires
`max_num_seqs=1`. The validated MTP launch also explicitly selected
probabilistic drafting. `max_num_seqs=3` is only a no-MTP profile until that
independent scheduler constraint changes.

The real-model Marlin gate selected `MarlinLinearKernel`, passed all four
Qwen3.6-27B shape self-checks, captured the graph, routed M=1/2 through SIMT,
and routed M=206/3920/8192 through Marlin. A 4096-to-128 request returned HTTP
200. This is a fallback-integration gate, not a paired Marlin performance
claim.

### Dense NVFP4

Both TurboMind and Marlin hybrid custom ops were exercised through
`torch.compile(dynamic=True, fullgraph=True)` on a V100. Each compiled function
saw M=1 SIMT, M=8 QPN, and M=17 selected-base fallback at runtime, so a graph
traced during decode cannot specialize a later prefill into a Skinny-only op.

The earlier full-checkpoint gate for
`OptimizeLLM/Qwen3.5-122B-A10B-heretic-MTP-NVFP4` remains the model-level
NVFP4 acceptance: Dense SIMT/QPN routes, native MTP3, image, video, tool use,
and a 262,016-token request passed. Its routed experts remained Marlin; that
mixed-MoE result must not be attributed entirely to the Dense kernel.

After the split-K changes, the same real checkpoint passed an additional TP4
MTP3 regression with full graph capture. QPN executed at M=4/8/12, SIMT at
M=1, and the selected TurboMind base handled large M; all four Dense shapes
passed both route self-checks. A fixed 4040-to-256 request measured 3610.06
prefill tok/s and 78.03 decode tok/s. This closes functional compatibility of
the retained split-K path, but is not a paired speed comparison.

### Grouped AWQ MoE prototype

The default-off replacement-layout prototype was tested with a two-layer,
four-expert selective-AWQ Qwen3.5 fixture. Its full-stack request hit the
grouped-SIMT expert route and returned HTTP 200. A Skinny-off run of the same
fixture used the existing TurboMind MoE path and produced the identical 24
deterministic completion tokens. Both fatal scans were clean. This is an eager
functional gate only; graph, real-model quality, memory, and performance gates
remain required before the MoE opt-in can be promoted.

### Operational guard

The accepted Qwen3.5/Qwen3.6 measurements above used the older 1Cat 1.2.2
runtime image, whose installed TileLang was 0.1.9, and therefore explicitly
set `VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0` as the temporary workaround for
[1Cat issue #105](https://github.com/1CatAI/1Cat-vLLM/issues/105). Omitting it
once reproduced the `common.h:778` BF16-to-FP16 compilation failure; that
failed launch is excluded from performance evidence.

This is dependency drift, not a decision to abandon TileLang. The source tree
already pins `tilelang==0.1.10` and `apache-tvm-ffi==0.1.10`, where the SM70
fallback is fixed. The intended production profile keeps original TileLang
prefill enabled by default and requires the runtime image to match those pins.
VLK remains a diagnostic A/B route, not the default fix for issue #105.

That dependency closure was repeated on the real
`OptimizeLLM/Qwen3.5-122B-A10B-heretic-MTP-NVFP4` checkpoint with image
`1cat-vllm:1.2.2-cu128-responses-thinking-budget-tilelang010-skinny-v1`
(`sha256:30cfc7a3c794...`). Both packages reported `0.1.10`; neither original
prefill override was set. The server selected FlashQLA-SM70, logged the
original TileLang path, compiled all four TileLang kernels on TP4, completed
the compile/CUDA-graph gate, and returned HTTP 200 for a 6,323-input-token
request. No `common.h:778`, BF16-to-half conversion, engine-init, OOM, or CUDA
failure occurred. The one-shot request took 2.525 seconds and the interval
logger reported 632.3 prompt tok/s; this is a functional closure, not a new
paired prefill benchmark.

Production images therefore keep an exact, tested TileLang/TVM-FFI pair rather
than following `latest` at startup. New TileLang releases should first enter an
isolated image and pass the SM70 header compile, original-prefill real-model,
CUDA-graph, long-context, correctness, and paired-performance gates before the
pin is deliberately advanced.

### Selector safety invariants

The overlay switch does not change the capability of the selected base:
`VLLM_SM70_SKINNY=off` still permits an available SM70 TurboMind or Marlin
route. Weight-only W4A16 NVFP4 also keeps its established Marlin route away
from exact SM70; it must not fall through the generic W4A4 selector, whose
activation-quantized kernels require state that a W4A16 scheme does not own.
