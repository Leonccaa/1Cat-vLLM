# SM70 Skinny NVFP4 dense overlay

## Scope

Skinny is a small-batch decorator for dense W4A16 NVFP4 linear layers on exact
SM70 GPUs. It is model-agnostic: selection depends on the quantization layout
and GEMM shape, not on a Qwen model class. The wrapped base kernel remains
responsible for larger batches and unsupported shapes.

The preferred controls are:

```bash
VLLM_SM70_QUANT_BACKEND=auto \
VLLM_SM70_SKINNY=auto \
vllm serve ... --dtype half
```

`VLLM_SM70_QUANT_BACKEND` continues to select `auto`, `turbomind`, or
`marlin`; `VLLM_SM70_SKINNY` independently selects `auto`, `on`, or `off`.
The old `VLLM_SM70_QUANT_BACKEND=skinny` value is accepted only as a
compatibility alias for base `auto` plus overlay `on`. NVFP4 uses this runtime
frontier:

| Condition | Route |
| --- | --- |
| FP16, `M <= 3`, `K % 128 == 0`, `N % 8 == 0` | skinny SIMT |
| FP16, `4 <= M <= 16`, QPN-eligible shape | skinny QPN |
| Larger M or unsupported shape | selected base kernel |
| BF16 activation | explicit FP16 conversion, selected route, BF16 output |

The decorator runs one eager comparison against the selected base for each
route on every prepared layer state. A non-finite result, exception, or
relative error above `3e-2` disables only that route for that state; the
selected base remains available because it is prepared unconditionally.

Dispatch and fallback are contained in one opaque hybrid custom op for each
supported base (`Skinny+TurboMind` and `Skinny+Marlin`). The op sees the real
runtime row count, so `torch.compile` or CUDA Graph tracing at M=1 cannot bind a
later prefill to the SIMT-only route. V100 tests compile one dynamic full graph
and then verify M=1 SIMT, M=8 QPN, and M=17 base fallback through that same
compiled callable.

## Checkpoint contract

The backend consumes the standard native NVFP4 W4A16 representation:

- packed E2M1 codes: `uint8 [N, K / 2]`;
- FP8-E4M3 scales with group size 16: `[N, K / 16]`;
- one multiplicative global weight scale.

Both compressed-tensors and ModelOpt NVFP4 linear adapters can use the backend.
W4A4 checkpoints use it as a weight-only W4A16 fallback on SM70; their
activation scales are not consumed on Volta. A mixed ModelOpt checkpoint may
still have a higher device-capability requirement because its non-NVFP4 layers
need their own SM70 conversion or fallback. Converting such a checkpoint to an
SM70-compatible compressed-tensors artifact remains a separate checkpoint-build
step.

## Memory and fallback

For dense models the load-time representation contains:

1. an overlay-owned checkpoint-native codes/scales copy for SIMT;
2. one same-size fragment-order QPN copy when that layout is enabled;
3. the selected base kernel's own prepared state.

The overlay owns its copy because a base kernel may replace or repack the
checkpoint tensors in place. This profile is appropriate for dense 27B-class
models under tensor parallelism, but it must not be applied wholesale to a
large MoE expert bank.

MoE is a separate selector. The experimental AWQ expert adapter uses one
replacement-layout SIMT bank and remains default-off; it does not change this
Dense NVFP4 decorator or the selected MoE backend. A future NVFP4 expert path
would still need a bounded/lazy or replacement-layout policy rather than an
unbounded duplicate of every expert.

## Provenance

The CUDA implementation is the production SIMT and QPN subset of the
MIT-licensed `Leonccaa/v100-skinny` project at commit
`f8194f7c3c9269fa74ee70b5029d53c20098f4c8`. Its license and pinned-source note
live beside the source in `csrc/quantization/skinny_nvfp4/`.
