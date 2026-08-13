# SM70 Skinny standalone kernel harness

Compiles the SIMT/QPN cores and format policies against plain CUDA — no torch,
no vLLM build — so the kernels can be numerically validated against a host FP64
reference and timed on a real V100 in about a minute instead of an hour.

This exists because the full extension build is a poor iteration loop for
kernel work, and because a wrong benchmark is worse than no benchmark: see
"L2 residency" below.

## Running

Needs only a CUDA devel image and a Volta card. On CT252 (`llm252.lan`,
4x V100-PCIE-32GB):

```bash
cd ~/skinny-harness && docker run --rm --security-opt apparmor=unconfined --gpus '"device=0"' -v $PWD:/w -w /w nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -lc 'nvcc -O3 -std=c++17 -arch=sm_70 -Wno-deprecated-gpu-targets -diag-suppress 177 -I new harness.cu -o harness && ./harness'
```

Lay the tree out as `<dir>/{harness.cu, new/sm70_skinny/...}` where
`new/sm70_skinny` is a copy of `csrc/quantization/sm70_skinny`. To A/B against
another revision, put that revision's headers in a second directory and build
with `-I <other>`; `-DHARNESS_BASELINE=1` selects the pre-split-K kernel
signatures, and `-DHARNESS_OLD_SIMT=1` selects the old SIMT signature with the
new QPN one.

`--security-opt apparmor=unconfined` is required inside the LXC.

Useful switches:

- `SKINNY_SPLITS=<n>` forces the QPN K-split factor instead of the policy.
- `-DSM70_SKINNY_STAGE_ONCE_SMEM=<bytes>` if a stage-once variant is
  reintroduced.
- `./harness <M>` restricts to a single M.

## What it checks

Shapes are the per-rank projections of a Qwen-class 27B under TP4, which is
what decode actually runs. For each shape and each M in {1,2,3,4,8,16} it
compares against a host FP64 dequant + GEMM over the first and last 256 output
columns — both windows, so a tile-index error and an intra-tile permutation
error are distinguishable — then reports max relative error, milliseconds, and
effective GB/s.

## L2 residency

The harness rotates over enough distinct weight buffers to exceed the 6 MB L2.
This is not incidental. Real decode walks every layer before returning to this
one, so a layer's weights are never still cached on the next step. An earlier
version of this harness hammered a single buffer, which gave any shape under
6 MB free L2 hits and produced two confident, entirely fictitious results: a
1.46x "win" on `o_proj` QPN and a 0.85x "regression" on `qkv` SIMT, both of
which were really just measuring whether `__ldcs` defeated an L2 reuse pattern
that does not exist in production. Do not remove the rotation.

## Results that shaped the current kernels

Measured on V100-PCIE-32GB, min of 3 interleaved rounds, application clocks not
locked (the LXC cannot set them), so treat sub-3% differences as noise.

- QPN split-K: 1.91x–2.48x on `qkv` (N=1792, 4 splits); exactly 1.00x on shapes
  that already fill the GPU. This is an isolated-kernel capability, not an
  end-to-end claim: Qwen3.6-27B AWQ does not quantize that qkv shape, and its
  three-repeat M=8 gate measured QPN 10.9% slower than the default route.
- `__ldcs` on streaming weight codes: +6% to +21% on three of four shapes, −6%
  on the one sitting near the L2 boundary.
- Rejected after measuring: merging the SIMT full-chunk and tail loops into one
  runtime-bounded loop (−5% to −15%, loses the compile-time trip count), and
  factoring the accumulation body into a `__forceinline__` helper taking the
  accumulator by reference (−10% to −14%, the array stops living in registers).
  Both are recorded in `simt.cuh` so they are not retried blindly.
- Stage-once activation staging measured net-negative and was dropped.
