# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _start_decode_request(scheduler) -> None:
    request = create_requests(
        num_requests=1,
        num_tokens=8,
        max_tokens=8,
        req_ids=["decode"],
    )[0]
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["decode"],
            req_id_to_index={"decode": 0},
            sampled_token_ids=[[1]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_prefill_budget_is_inactive_without_decode():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
    )
    request = create_requests(
        num_requests=1,
        num_tokens=200,
        max_tokens=4,
        req_ids=["prefill"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"prefill": 96}


def test_prefill_budget_caps_mixed_step_and_preserves_decode():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
    )
    _start_decode_request(scheduler)
    requests = create_requests(
        num_requests=2,
        num_tokens=200,
        max_tokens=4,
        req_ids=["prefill-0", "prefill-1"],
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["decode"] == 1
    assert output.num_scheduled_tokens["prefill-0"] == 32
    assert "prefill-1" not in output.num_scheduled_tokens
    assert sum(output.num_scheduled_tokens.values()) == 33

    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["decode", "prefill-0"],
            req_id_to_index={"decode": 0, "prefill-0": 1},
            sampled_token_ids=[[2], []],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    next_output = scheduler.schedule()

    assert next_output.num_scheduled_tokens["decode"] == 1
    assert next_output.num_scheduled_tokens["prefill-0"] == 32
    assert "prefill-1" not in next_output.num_scheduled_tokens


def test_prefill_budget_is_aggregate_across_requests():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
    )
    _start_decode_request(scheduler)
    requests = create_requests(
        num_requests=2,
        num_tokens=20,
        max_tokens=4,
        req_ids=["prefill-0", "prefill-1"],
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["decode"] == 1
    assert output.num_scheduled_tokens["prefill-0"] == 20
    assert output.num_scheduled_tokens["prefill-1"] == 12
    assert (
        sum(
            tokens
            for request_id, tokens in output.num_scheduled_tokens.items()
            if request_id != "decode"
        )
        == 32
    )


def test_prefill_pressure_threshold_keeps_low_pressure_path_unchanged():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        prefill_pressure_threshold=2,
        max_model_len=512,
    )
    _start_decode_request(scheduler)
    request = create_requests(
        num_requests=1,
        num_tokens=200,
        max_tokens=4,
        req_ids=["prefill"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["decode"] == 1
    assert output.num_scheduled_tokens["prefill"] == 95


def test_prefill_pressure_budget_requires_chunked_prefill():
    with pytest.raises(ValueError, match="Chunked prefill must be enabled"):
        create_scheduler(
            max_num_batched_tokens=512,
            enable_chunked_prefill=False,
            prefill_pressure_token_budget=32,
            max_model_len=512,
        )
