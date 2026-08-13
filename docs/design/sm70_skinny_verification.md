# SM70 Skinny: what is verified, and what still needs a V100 build

Companion to `sm70_skinny_execution_layer.md`. Records the state of the
optimization/robustness pass so the remaining gates are explicit rather than
assumed.

## Verified

Standalone kernel harness on CT252 (`llm252.lan`, 4x V100-PCIE-32GB), see
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

## Not yet verified — needs a real extension build

The dev box has no `nvcc` and no GPU, and CT252's root filesystem is at 96%
(~5.5 GiB free), which is not enough for a full `_C` build. Everything below
is therefore untested end to end:

1. **Build the extension** for `compute_70/sm_70` on CUDA 12.8 and record the
   new `_C.abi3.so` SHA-256. Free disk on CT252 first.
2. **GPU operator tests**: `tests/quantization/test_sm70_skinny_nvfp4_gpu.py`
   and `test_sm70_skinny_awq_gpu.py` (5 test functions).
3. **Split-K under CUDA Graph.** The `k_splits > 1` path allocates an FP32
   workspace per call and runs a second reduction kernel. Both should be fine
   under capture — the allocation comes from the graph pool — but this has only
   been exercised outside vLLM. Verify capture and repeated replay, and
   interleave M=1/8/17 in one compiled callable as the existing gate does.
4. **Determinism of the split-K reduction.** The reduction sums splits in fixed
   order and should be bit-reproducible; confirm run-to-run equality rather
   than assuming it.
5. **MoE zero-fill.** The invalid-expert path now writes zeros instead of
   leaving the previous step's values in the reused output buffer. Covered by
   no test; the MoE path is default-off.
6. **Residency policy on a real model.** This is the one that needs judgement,
   not just a pass/fail — see below.

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

Run 3 and 4 before 6: a residency policy that silently disables the overlay
would make a broken split-K path look fine.
