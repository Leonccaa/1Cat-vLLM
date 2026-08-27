# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    _qsa_indexer_cublas_shape_supported,
    _qsa_sparse_launch_profile,
    _repair_qsa_boundary_ties_sm70,
)


def test_sm70_qsa_prefill_uses_narrow_tiles_and_four_warps():
    assert _qsa_sparse_launch_profile(511, 8, True) == (64, 4, 4)
    assert _qsa_sparse_launch_profile(512, 8, True) == (32, 4, 4)
    assert _qsa_sparse_launch_profile(8192, 8, True) == (32, 1, 4)


def test_non_sm70_qsa_prefill_keeps_gb300_profile():
    assert _qsa_sparse_launch_profile(512, 8, False) == (64, 4, 2)
    assert _qsa_sparse_launch_profile(8192, 8, False) == (64, 1, 2)


def test_qsa_indexer_cublas_accepts_only_exact_single_request_shape():
    query = torch.empty(8, 4, 128, dtype=torch.float16)
    cache = torch.empty(2, 400, 1, 128, dtype=torch.float16)
    page_table = torch.empty(1, 2, dtype=torch.int32)

    assert _qsa_indexer_cublas_shape_supported(query, cache, page_table)
    assert not _qsa_indexer_cublas_shape_supported(
        query.to(torch.bfloat16), cache, page_table
    )
    assert not _qsa_indexer_cublas_shape_supported(
        query, cache, page_table.expand(2, -1)
    )
    assert not _qsa_indexer_cublas_shape_supported(query[:, :3], cache, page_table)


def test_qsa_indexer_cublas_does_not_capture_decode_rows(monkeypatch):
    cache = torch.empty(2, 400, 1, 128, dtype=torch.float16)
    page_table = torch.empty(1, 2, dtype=torch.int32)
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_CUBLAS", True)
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_CUBLAS_MIN_ROWS", 256)
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )

    assert qsa_ops._use_sm70_qsa_indexer_cublas(
        torch.empty(256, 4, 128, dtype=torch.float16), cache, page_table
    )
    assert not qsa_ops._use_sm70_qsa_indexer_cublas(
        torch.empty(255, 4, 128, dtype=torch.float16), cache, page_table
    )


def test_qsa_indexer_cublas_requires_enough_score_work(monkeypatch):
    monkeypatch.setattr(
        qsa_ops,
        "_SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS",
        1024**2,
    )

    assert not qsa_ops._qsa_indexer_cublas_work_supported(1024, 512)
    assert qsa_ops._qsa_indexer_cublas_work_supported(2048, 512)


@pytest.mark.skip_global_cleanup
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm70_qsa_boundary_tie_repair_prefers_lowest_block_indices():
    rows = 3
    columns = 2048
    block_topk = 512
    visible = torch.tensor([1024, 700, 400], dtype=torch.int32, device="cuda")
    source = torch.zeros((rows, columns), dtype=torch.float32, device="cuda")
    source[0, :100] = 2.0
    source[1, 650:700] = 2.0
    source[2, :400] = torch.arange(400, dtype=torch.float32, device="cuda")

    expected = torch.full(
        (rows, block_topk),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    expected[0] = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    expected[1, :462] = torch.arange(462, dtype=torch.int32, device="cuda")
    expected[1, 462:] = torch.arange(650, 700, dtype=torch.int32, device="cuda")
    expected[2, :400] = torch.arange(400, dtype=torch.int32, device="cuda")

    outputs = []
    for _ in range(3):
        logits = source.clone()
        blocks = torch.empty(
            (rows, block_topk),
            dtype=torch.int32,
            device="cuda",
        )
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
        torch.ops._C.persistent_topk(
            logits,
            visible,
            blocks,
            workspace,
            block_topk,
            columns,
        )
        _repair_qsa_boundary_ties_sm70(
            logits,
            visible,
            blocks,
            workspace,
            columns,
        )
        torch.accelerator.synchronize()
        outputs.append(blocks.clone())

    assert all(torch.equal(output, expected) for output in outputs)
