#!/usr/bin/env python3
"""Validate the Qwen3.8-Flash-Next per-expert AWQ schema on TP4.

The fixture contains layer 0, experts 0 and 1 only.  It is intentionally a
loader gate, not a production quantized checkpoint or an inference benchmark.
Run it on four V100s with ``torchrun --nproc-per-node=4``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

os.environ.setdefault("VLLM_SM70_AWQ_TURBOMIND", "1")


CHECKPOINT_PREFIX = "model.language_model.layers.0.mlp.experts."
WEIGHT_FILE = "model-00001-of-00001.safetensors"
EXPECTED_LOADED_PARAMS = {
    "w13_qweight",
    "w13_qzeros",
    "w13_scales",
    "w2_qweight",
    "w2_qzeros",
    "w2_scales",
}


def _read_fixture(fixture_dir: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    with (fixture_dir / "config.json").open() as stream:
        config = json.load(stream)
    quant_config = config["quantization_config"]
    expected_quant_config = {
        "quant_method": "awq",
        "bits": 4,
        "zero_point": True,
        "version": "gemm",
    }
    for key, expected in expected_quant_config.items():
        actual = quant_config.get(key)
        if actual != expected:
            raise ValueError(
                f"quantization_config.{key}={actual!r}, expected {expected!r}"
            )
    if quant_config.get("group_size") not in {32, 128}:
        raise ValueError(
            "fixture group_size must be 32 or 128, got "
            f"{quant_config.get('group_size')!r}"
        )

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(fixture_dir / WEIGHT_FILE, framework="pt", device="cpu") as handle:
        # ``safe_open`` exposes keys() but is not itself iterable.
        for name in handle.keys():  # noqa: SIM118
            if not name.startswith(CHECKPOINT_PREFIX):
                raise ValueError(f"unexpected fixture tensor: {name}")
            tensors[name] = handle.get_tensor(name)
    if len(tensors) != 18:
        raise ValueError(f"expected 18 fixture tensors, found {len(tensors)}")
    return config, tensors


def _expected_tp_slice(
    source: torch.Tensor,
    projection: str,
    suffix: str,
    rank: int,
    group_repeat_factor: int,
) -> torch.Tensor:
    if suffix in {"qzeros", "scales"} and group_repeat_factor > 1:
        source = source.repeat_interleave(group_repeat_factor, dim=0)
    if projection in {"gate_proj", "up_proj"}:
        shard_size = source.shape[1] // 4
        return source[:, rank * shard_size : (rank + 1) * shard_size]
    shard_size = source.shape[0] // 4
    return source[rank * shard_size : (rank + 1) * shard_size]


def _assert_loaded_tensor(
    layer: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    expert_id: int,
    projection: str,
    suffix: str,
    rank: int,
    group_repeat_factor: int,
) -> None:
    source_name = f"{CHECKPOINT_PREFIX}{expert_id}.{projection}.{suffix}"
    expected = _expected_tp_slice(
        tensors[source_name],
        projection,
        suffix,
        rank,
        group_repeat_factor,
    )
    if projection == "down_proj":
        param_name = f"w2_{suffix}"
        actual = getattr(layer, param_name)[expert_id]
    else:
        param_name = f"w13_{suffix}"
        actual = getattr(layer, param_name)[expert_id]
        half = actual.shape[-1] // 2
        actual = actual[..., :half] if projection == "gate_proj" else actual[..., half:]
    expected = expected.to(actual.dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4:
        raise ValueError(f"this fixture requires TP4, got WORLD_SIZE={world_size}")

    config_json, tensors = _read_fixture(args.fixture)
    text_config = config_json["text_config"]
    checkpoint_group_size = config_json["quantization_config"]["group_size"]

    torch.cuda.set_device(local_rank)
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.quantization.awq import AWQConfig

    parallel_config = ParallelConfig(tensor_parallel_size=world_size)
    vllm_config = VllmConfig(parallel_config=parallel_config)
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
                backend="nccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=world_size)

            layer_name = "model.layers.0.mlp.experts"
            with torch.device(f"cuda:{local_rank}"):
                layer = FusedMoE(
                    num_experts=text_config["num_experts"],
                    top_k=text_config["num_experts_per_tok"],
                    hidden_size=text_config["hidden_size"],
                    intermediate_size=text_config["moe_intermediate_size"],
                    params_dtype=torch.float16,
                    quant_config=AWQConfig(4, checkpoint_group_size, True),
                    tp_size=world_size,
                    ep_size=1,
                    dp_size=1,
                    pcp_size=1,
                    prefix=layer_name,
                )

            holder = torch.nn.Module()
            holder.add_module("experts", layer)
            layer.expert_mapping = layer.make_expert_params_mapping(
                holder,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=text_config["num_experts"],
            )

            for name, param in layer.named_parameters(recurse=False):
                if name.endswith(("qweight", "qzeros")):
                    param.data.fill_(torch.iinfo(torch.int32).min)
                elif name.endswith("scales"):
                    param.data.fill_(torch.nan)

            relative_tensors = [
                (name.removeprefix(CHECKPOINT_PREFIX), tensor)
                for name, tensor in sorted(tensors.items())
            ]
            loaded = set(layer.load_weights(relative_tensors))
            if loaded != EXPECTED_LOADED_PARAMS:
                raise AssertionError(
                    f"loaded params {sorted(loaded)}, expected "
                    f"{sorted(EXPECTED_LOADED_PARAMS)}"
                )

            for expert_id in (0, 1):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("qweight", "qzeros", "scales"):
                        _assert_loaded_tensor(
                            layer,
                            tensors,
                            expert_id,
                            projection,
                            suffix,
                            rank,
                            layer.quant_method.group_size_div_factor,
                        )

            sentinel = torch.iinfo(torch.int32).min
            for name, param in layer.named_parameters(recurse=False):
                expert_two = param[2]
                if name.endswith(("qweight", "qzeros")):
                    if not torch.all(expert_two == sentinel):
                        raise AssertionError(f"unlisted expert changed in {name}")
                elif name.endswith("scales") and not torch.isnan(expert_two).all():
                    raise AssertionError(f"unlisted expert changed in {name}")

            success = torch.ones((), dtype=torch.int32, device="cuda")
            torch.distributed.all_reduce(success)
            if rank == 0:
                method = layer.quant_method
                print(
                    json.dumps(
                        {
                            "status": "AWQ_LOADER_FIXTURE_OK",
                            "tp_size": world_size,
                            "checkpoint_group_size": method.checkpoint_group_size,
                            "runtime_group_size": method.group_size,
                            "group_repeat_factor": method.group_size_div_factor,
                            "loaded_params": sorted(loaded),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        if torch.distributed.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
