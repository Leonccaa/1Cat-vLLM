# SM70 block-128 FP8 QPN8 overlay

This directory contains an SM70 small-M CUDA dataflow derived from the
MIT-licensed [`v100-skinny`](https://github.com/Leonccaa/v100-skinny) QPN8
kernel. The integration spike is based on `v100-skinny` commit
`5b589c0dc81223e0ba65bcb3e755874723f8b515`; the block-128 scale recurrence
was developed and gated in the local evaluation commit
`5b59adcca052a7ba819679dcb332a7d065555a31`.

The operator consumes checkpoint-faithful E4M3 bytes plus 128x128 block
scales. Python performs a one-time QPN fragment-order permutation and builds
FP32 adjacent-scale ratios. The kernel serves FP16 M=1..8. Unsupported rows,
dtypes, bias, formats, shapes, or devices remain on the selected base linear
kernel.

This is currently an explicit dual-copy integration gate:

- `VLLM_SM70_SKINNY=on` enables eligible block-128 FP8 dense layers.
- `auto` and `off` retain the selected base backend.
- It is not a linear backend and does not change MoE routing.
- Default-auto requires full-model validation and a bounded one-copy policy.
