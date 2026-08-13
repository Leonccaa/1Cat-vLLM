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
