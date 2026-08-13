# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from benchmarks.verify_sm70_runtime_bundle import (
    parse_expected_hash,
    sha256_file,
    verify_native_files,
)


def test_verify_native_files_accepts_matching_bundle(tmp_path: Path) -> None:
    extension = tmp_path / "_C.abi3.so"
    extension.write_bytes(b"sm70-test-extension")
    digest = sha256_file(extension)

    records = verify_native_files(
        tmp_path,
        [extension.name],
        {extension.name: digest},
    )

    assert records == [
        {
            "name": extension.name,
            "path": str(extension.resolve()),
            "bytes": len(b"sm70-test-extension"),
            "sha256": digest,
            "expected_sha256": digest,
        }
    ]


@pytest.mark.parametrize("contents", [None, b""])
def test_verify_native_files_rejects_missing_or_empty_bundle(
    tmp_path: Path,
    contents: bytes | None,
) -> None:
    extension = tmp_path / "_moe_C.abi3.so"
    if contents is not None:
        extension.write_bytes(contents)

    with pytest.raises(RuntimeError, match="native extension"):
        verify_native_files(tmp_path, [extension.name], {})


def test_verify_native_files_rejects_hash_mismatch(tmp_path: Path) -> None:
    extension = tmp_path / "_C.abi3.so"
    extension.write_bytes(b"unexpected")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_native_files(tmp_path, [extension.name], {extension.name: "0" * 64})


def test_parse_expected_hash_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="must not contain a path"):
        parse_expected_hash("../_C.abi3.so=" + "0" * 64)
