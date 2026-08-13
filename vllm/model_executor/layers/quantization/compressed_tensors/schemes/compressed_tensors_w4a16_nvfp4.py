# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

__all__ = ["CompressedTensorsW4A16Fp4"]


class CompressedTensorsW4A16Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process parameters for marlin repacking

        # Rename weight_packed to weight that marlin expects
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        # CT has two scale conventions under the same schema. Exact SM70
        # classifies them at the format boundary and fails closed on invalid or
        # mixed metadata; other platforms retain the existing reciprocal path.
        if sm70_tm.is_exact_sm70_cuda_platform():
            sm70_tm.normalize_nvfp4_ct_global_scale_for_sm70(layer)
        else:
            layer.weight_global_scale = Parameter(
                1.0 / layer.weight_global_scale.max().to(torch.float32),
                requires_grad=False,
            )

        # This on-disk scheme is weight-only W4A16. Preserve its existing
        # Marlin behavior away from V100: the generic NVFP4 registry also
        # contains W4A4 kernels that require activation-scale state this
        # scheme intentionally does not create. Exact SM70 alone uses the
        # base selector plus optional Skinny decorator.
        if sm70_tm.is_exact_sm70_cuda_platform():
            self.nvfp4_kernel = init_nvfp4_linear_kernel()
        else:
            self.nvfp4_kernel = MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())
        self.nvfp4_kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.nvfp4_kernel.apply_weights(layer, x, bias)
