#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build disjoint long-context QSA calibration shards with overlap gating."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from array import array
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_documents(path: Path, source_kind: str) -> Iterator[tuple[str, str]]:
    if source_kind == "code-json-gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                content = row.get("content")
                if isinstance(content, str) and content.strip():
                    identity = (
                        f"{row.get('repo_name', '')}:{row.get('path', line_number)}"
                    )
                    yield identity, content
        return

    try:
        from pyarrow import parquet
    except ImportError as error:
        raise RuntimeError(
            "Parquet long sources require pyarrow (install it in the isolated "
            "corpus-build environment)"
        ) from error
    parquet_file = parquet.ParquetFile(path)
    columns = parquet_file.schema_arrow.names
    text_column = "text" if "text" in columns else "content"
    if text_column not in columns:
        raise ValueError(f"No text/content column in {path}: {columns}")
    row_number = 0
    for batch in parquet_file.iter_batches(batch_size=1024, columns=[text_column]):
        for content in batch.column(0).to_pylist():
            row_number += 1
            if isinstance(content, str) and content.strip():
                yield f"row-{row_number}", content


def rolling_shingles(token_ids: list[int], width: int, stride: int) -> set[int]:
    if len(token_ids) < width:
        return set()
    mask = (1 << 64) - 1
    base = 1_000_003
    high_power = pow(base, width - 1, 1 << 64)
    value = 0
    for token in token_ids[:width]:
        value = (value * base + token + 1) & mask
    output = {value}
    for index in range(width, len(token_ids)):
        outgoing = token_ids[index - width] + 1
        value = (value - outgoing * high_power) & mask
        value = (value * base + token_ids[index] + 1) & mask
        if (index - width + 1) % stride == 0:
            output.add(value)
    return output


def load_quality_shingles(
    paths: list[Path], tokenizer: Any, width: int, stride: int
) -> tuple[set[int], list[dict[str, Any]]]:
    shingles: set[int] = set()
    provenance = []
    for path in paths:
        records = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                prompt = row.get("prompt")
                if not isinstance(prompt, str):
                    raise TypeError(f"Missing quality prompt at {path}:{line_number}")
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                shingles.update(rolling_shingles(token_ids, width, stride))
                records += 1
        provenance.append(
            {"path": str(path), "sha256": sha256(path), "records": records}
        )
    return shingles, provenance


def token_ids_sha256(token_ids: list[int]) -> str:
    values = array("q", token_ids)
    if values.itemsize != 8:
        raise RuntimeError("Expected signed 64-bit token audit representation")
    return hashlib.sha256(values.tobytes()).hexdigest()


def build(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    source = Path(args.input).resolve()
    if sha256(source) != args.input_sha256.lower():
        raise ValueError("Long source SHA-256 does not match --input-sha256")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Long calibration output already exists: {output_dir}")
    targets = sorted(set(args.target_tokens))
    if not targets or targets[0] <= 0 or args.samples_per_target <= 0:
        raise ValueError("Targets and samples-per-target must be positive")

    model = str(Path(args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, local_files_only=True
    )
    quality_paths = [Path(path).resolve() for path in args.quality_manifest_jsonl]
    if not quality_paths:
        raise ValueError(
            "At least one --quality-manifest-jsonl is required; refusing to "
            "assume long-source separation"
        )
    forbidden, quality_provenance = load_quality_shingles(
        quality_paths, tokenizer, args.shingle_width, 1
    )

    documents = iter(iter_documents(source, args.source_kind))
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    pending: list[int] = []
    pending_identity = ""
    source_documents: list[dict[str, Any]] = []
    outputs: dict[int, list[dict[str, Any]]] = {target: [] for target in targets}
    for target in targets:
        for sample_index in range(args.samples_per_target):
            token_ids = []
            document_spans = []
            bos_token_id = tokenizer.bos_token_id
            if bos_token_id is not None:
                token_ids.append(int(bos_token_id))
            while len(token_ids) < target:
                if not pending:
                    try:
                        identity, content = next(documents)
                    except StopIteration as error:
                        raise ValueError(
                            "Long source exhausted before all shards"
                        ) from error
                    pending = tokenizer.encode(content, add_special_tokens=False)
                    pending_identity = identity
                    source_documents.append(
                        {
                            "identity": identity,
                            "content_sha256": hashlib.sha256(
                                content.encode("utf-8")
                            ).hexdigest(),
                            "tokens": len(pending),
                        }
                    )
                    if token_ids and separator:
                        pending = separator + pending
                take = min(target - len(token_ids), len(pending))
                span_start = len(token_ids)
                token_ids.extend(pending[:take])
                pending = pending[take:]
                document_spans.append(
                    {
                        "identity": pending_identity,
                        "start": span_start,
                        "end": len(token_ids),
                    }
                )
            overlaps = (
                rolling_shingles(token_ids, args.shingle_width, args.shingle_stride)
                & forbidden
            )
            if overlaps:
                raise ValueError(
                    f"Quality-set token-shingle overlap detected in {args.category} "
                    f"target={target} sample={sample_index}: {len(overlaps)} matches"
                )
            outputs[target].append(
                {
                    "prompt_token_ids": token_ids,
                    "calibration_metadata": {
                        "category": args.category,
                        "target_tokens": target,
                        "sample_index": sample_index,
                        "document_spans": document_spans,
                        "prompt_token_ids_sha256": token_ids_sha256(token_ids),
                    },
                }
            )

    output_dir.mkdir(parents=True)
    output_files = []
    for target, rows in outputs.items():
        output = output_dir / f"{args.category}-{target}.jsonl"
        with output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        output_files.append(
            {
                "path": str(output),
                "sha256": sha256(output),
                "target_tokens": target,
                "records": len(rows),
            }
        )
    manifest = {
        "schema_version": 1,
        "model": model,
        "source_kind": args.source_kind,
        "source": str(source),
        "source_sha256": sha256(source),
        "category": args.category,
        "targets": targets,
        "samples_per_target": args.samples_per_target,
        "packing": "sequential-disjoint-raw-token-stream",
        "quality_overlap_gate": {
            "algorithm": "rolling uint64 polynomial token shingles",
            "width": args.shingle_width,
            "candidate_stride": args.shingle_stride,
            "quality_stride": 1,
            "quality_manifests": quality_provenance,
            "matching_shingles": 0,
        },
        "source_documents": source_documents,
        "output_files": output_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source-kind", choices=["parquet", "code-json-gz"], required=True
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-tokens", type=int, nargs="+", default=[16384, 65536, 128000]
    )
    parser.add_argument("--samples-per-target", type=int, default=2)
    parser.add_argument("--quality-manifest-jsonl", action="append", default=[])
    parser.add_argument("--shingle-width", type=int, default=64)
    parser.add_argument("--shingle-stride", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
