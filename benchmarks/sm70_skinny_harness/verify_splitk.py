# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the split-K QPN path: correctness, determinism, CUDA Graph replay.

The k_splits > 1 path allocates an FP32 workspace per call and runs a second
reduction kernel. Neither behaviour existed before, so both need checking
inside a graph capture as well as in eager mode.
"""

import torch

from vllm.model_executor.layers.quantization import sm70_skinny

torch.manual_seed(0)
DEV = "cuda"


def build_awq(n, k, group=128):
    """Native-layout AWQ tensors plus the Skinny state built from them."""
    logical = torch.randint(0, 16, (k, n), dtype=torch.uint8)
    order = torch.tensor(sm70_skinny._AWQ_REVERSE_PACK_ORDER)
    inverse = torch.argsort(order)

    def pack(rows):
        p = rows.view(rows.shape[0], -1, 8).index_select(-1, inverse)
        b = p[..., 0::2] | (p[..., 1::2] << 4)
        return b.contiguous().view(rows.shape[0], -1).view(torch.int32)

    zeros = torch.randint(0, 16, (k // group, n), dtype=torch.uint8)
    scales = (torch.rand(k // group, n) * 0.02 + 0.004).to(torch.float16)
    return (
        pack(logical).to(DEV),
        scales.to(DEV),
        pack(zeros).to(DEV),
    )


def main():
    # N=1792, K=5120 is the per-rank qkv shape; qpn_choose_k_splits gives 4.
    n, k = 1792, 5120
    qweight, scales, qzeros = build_awq(n, k)
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)

    import os

    os.environ["VLLM_SM70_SKINNY_AWQ_LAYOUT"] = "both"
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    assert state.has_qpn, "QPN layout not resident; set layout=both"

    x = ((torch.rand(8, k, device=DEV) - 0.5) * 0.05).to(torch.float16)

    def run():
        return torch.ops._C.skinny_awq_gemm_qpn(
            x, state.qpn_codes, state.qpn_scales, state.qpn_biases, 128, n
        )

    out = run()
    torch.accelerator.synchronize()

    # ---- 1. correctness against the independent FP32 reference ----
    truth = sm70_skinny.awq_fp32_reference(state.codes, state.scales, state.biases, x)
    err = (out.float() - truth).abs().max() / truth.abs().max().clamp(min=1e-6)
    print(f"[1] split-K vs FP32 ground truth: max_rel={err.item():.3e}")
    assert err.item() < 3e-2, "split-K result does not match FP32 reference"

    # ---- 2. determinism: fixed split order must be bit-reproducible ----
    first = run().clone()
    torch.accelerator.synchronize()
    identical = True
    for _ in range(20):
        if not torch.equal(run(), first):
            identical = False
            break
    print(f"[2] 20 eager repeats bit-identical: {identical}")
    assert identical, "split-K reduction is not deterministic"

    # ---- 3. CUDA Graph capture + replay ----
    static_x = x.clone()

    def graph_run():
        return torch.ops._C.skinny_awq_gemm_qpn(
            static_x, state.qpn_codes, state.qpn_scales, state.qpn_biases, 128, n
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            graph_run()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        graph_out = graph_run()
    torch.accelerator.synchronize()

    replay_ok = True
    for _ in range(10):
        graph.replay()
        torch.accelerator.synchronize()
        if not torch.equal(graph_out, first):
            replay_ok = False
            break
    print(f"[3] CUDA Graph capture + 10 replays match eager: {replay_ok}")
    assert replay_ok, "split-K under CUDA Graph disagrees with eager"

    # ---- 4. changing the input between replays must change the output ----
    static_x.copy_(((torch.rand(8, k, device=DEV) - 0.5) * 0.05).to(torch.float16))
    graph.replay()
    torch.accelerator.synchronize()
    changed = not torch.equal(graph_out, first)
    print(f"[4] replay tracks new input (graph is live, not frozen): {changed}")
    assert changed, "graph replay ignored the updated input"

    # ---- 5. a shape that does NOT split must still be exact ----
    n2, k2 = 5120, 4352  # down_proj: qpn_choose_k_splits -> 1
    qw2, sc2, qz2 = build_awq(n2, k2)
    state2 = sm70_skinny.prepare_awq_state(qw2, sc2, qz2, 128)
    x2 = ((torch.rand(8, k2, device=DEV) - 0.5) * 0.05).to(torch.float16)
    out2 = torch.ops._C.skinny_awq_gemm_qpn(
        x2, state2.qpn_codes, state2.qpn_scales, state2.qpn_biases, 128, n2
    )
    truth2 = sm70_skinny.awq_fp32_reference(
        state2.codes, state2.scales, state2.biases, x2
    )
    err2 = (out2.float() - truth2).abs().max() / truth2.abs().max().clamp(min=1e-6)
    print(f"[5] non-split shape vs FP32 ground truth: max_rel={err2.item():.3e}")
    assert err2.item() < 3e-2

    # ---- 6. MoE invalid expert id must write zeros, not stale values ----
    ne, nm, km = 4, 64, 256
    codes = torch.randint(0, 256, (ne, nm, km // 2), dtype=torch.uint8, device=DEV)
    msc = (torch.rand(ne, nm, km // 128, device=DEV) * 0.02).to(torch.float16)
    mbi = (-torch.rand(ne, nm, km // 128, device=DEV) * 0.1).to(torch.float16)
    xm = ((torch.rand(3, km, device=DEV) - 0.5) * 0.05).to(torch.float16)
    outm = torch.full((3, nm), 123.0, dtype=torch.float16, device=DEV)
    ids = torch.tensor([0, 99, -1], dtype=torch.int32, device=DEV)
    torch.ops._C.skinny_awq_moe_gemm_simt_out(outm, xm, ids, codes, msc, mbi, 128)
    torch.accelerator.synchronize()
    zeroed = bool((outm[1] == 0).all() and (outm[2] == 0).all())
    print(f"[6] invalid expert ids zeroed (no stale 123.0 left): {zeroed}")
    assert zeroed, "invalid expert leaked the previous buffer contents"

    print("\nALL SPLIT-K CHECKS PASSED")


if __name__ == "__main__":
    main()
