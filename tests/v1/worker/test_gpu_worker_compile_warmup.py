# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.utils import Range
from vllm.v1.worker.gpu_worker import _missing_compile_range_warmup_sizes


@pytest.mark.parametrize(
    ("compile_range", "warmup_sizes", "capture_sizes", "expected"),
    [
        (Range(1, 256), [], [1], [256]),
        (Range(1, 256), [], [1, 32], []),
        (Range(1, 256), [64], [1], []),
        (Range(1, 1), [], [1], []),
        (Range(1, 1), [], [], [1]),
    ],
)
@pytest.mark.skip_global_cleanup
def test_compile_range_warmup_requires_non_specializing_size(
    compile_range: Range,
    warmup_sizes: list[int],
    capture_sizes: list[int],
    expected: list[int],
) -> None:
    assert (
        _missing_compile_range_warmup_sizes(
            [compile_range],
            warmup_sizes,
            capture_sizes,
        )
        == expected
    )
