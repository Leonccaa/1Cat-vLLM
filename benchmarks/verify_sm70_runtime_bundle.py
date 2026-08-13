# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed checks for a source-backed SM70 runtime bundle.

A Git source archive does not contain vLLM's generated native extensions. If
that archive shadows the installed package through ``PYTHONPATH``, imports can
succeed while operations such as ``_moe_C.topk_softmax`` are absent. Run this
tool inside the target runtime image before starting a real-model gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

DEFAULT_NATIVE_FILES = (
    "_C.abi3.so",
    "_C_stable_libtorch.abi3.so",
    "_moe_C.abi3.so",
    "cumem_allocator.abi3.so",
    "spinloop.abi3.so",
)

DEFAULT_IMPORTS = ("vllm._custom_ops", "vllm._sm70_ops")

DEFAULT_TORCH_OPS = (
    "_moe_C.topk_softmax",
    "_C.skinny_nvfp4_gemm_simt",
    "_C.skinny_awq_gemm_simt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expected_hash(raw: str) -> tuple[str, str]:
    name, separator, digest = raw.partition("=")
    if not separator or not name or len(digest) != 64:
        raise ValueError(f"expected NAME=SHA256, got {raw!r}")
    if Path(name).name != name:
        raise ValueError(f"native filename must not contain a path: {name!r}")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"invalid SHA256 digest in {raw!r}") from error
    return name, digest.lower()


def verify_native_files(
    package_root: Path,
    names: list[str],
    expected_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for name in names:
        if Path(name).name != name:
            failures.append(f"native filename must not contain a path: {name!r}")
            continue
        path = package_root / name
        if not path.is_file():
            failures.append(f"missing native extension: {path}")
            continue
        size = path.stat().st_size
        if size == 0:
            failures.append(f"empty native extension: {path}")
            continue
        digest = sha256_file(path)
        expected = expected_hashes.get(name)
        if expected is not None and digest != expected:
            failures.append(
                f"SHA256 mismatch for {path}: expected={expected} actual={digest}"
            )
        records.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "bytes": size,
                "sha256": digest,
                "expected_sha256": expected,
            }
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    return records


def verify_torch_ops(imports: list[str], ops: list[str]) -> None:
    for module in imports:
        importlib.import_module(module)

    import torch

    missing: list[str] = []
    for raw in ops:
        namespace, separator, op_name = raw.partition(".")
        if not separator or not namespace or not op_name:
            raise ValueError(f"expected NAMESPACE.OP, got {raw!r}")
        namespace_object = getattr(torch.ops, namespace)
        if not hasattr(namespace_object, op_name):
            missing.append(raw)
    if missing:
        raise RuntimeError(f"missing registered torch ops: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vllm",
    )
    parser.add_argument(
        "--native",
        action="append",
        dest="native_files",
        help="Required native filename. Defaults to the five vLLM runtime extensions.",
    )
    parser.add_argument(
        "--expect-sha256",
        action="append",
        default=[],
        metavar="NAME=SHA256",
    )
    parser.add_argument(
        "--runtime-op-gate",
        action="store_true",
        help="Import vLLM and require the MoE and Skinny torch ops.",
    )
    parser.add_argument("--import-module", action="append", default=[])
    parser.add_argument("--require-torch-op", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    native_files = args.native_files or list(DEFAULT_NATIVE_FILES)
    expected_hashes = dict(parse_expected_hash(raw) for raw in args.expect_sha256)
    unknown_hashes = sorted(set(expected_hashes) - set(native_files))
    if unknown_hashes:
        parser.error(
            "--expect-sha256 names must also be required by --native: "
            + ", ".join(unknown_hashes)
        )

    try:
        records = verify_native_files(args.package_root, native_files, expected_hashes)
        if args.runtime_op_gate or args.require_torch_op:
            imports = args.import_module or list(DEFAULT_IMPORTS)
            ops = args.require_torch_op or list(DEFAULT_TORCH_OPS)
            verify_torch_ops(imports, ops)
        result = {
            "status": "pass",
            "package_root": str(args.package_root.resolve()),
            "native_files": records,
            "runtime_op_gate": args.runtime_op_gate or bool(args.require_torch_op),
            "required_torch_ops": (
                args.require_torch_op or list(DEFAULT_TORCH_OPS)
                if args.runtime_op_gate or args.require_torch_op
                else []
            ),
        }
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        result = {
            "status": "fail",
            "package_root": str(args.package_root.resolve()),
            "error": str(error),
        }
        if args.output:
            args.output.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(result, indent=2))
        raise SystemExit(1) from error

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
