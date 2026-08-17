# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class TurboMindNvFp4LinearKernel(NvFp4LinearKernel):
    """Exact-SM70 NVFP4 W4A16 base backend."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            if (
                not current_platform.is_cuda()
                or not current_platform.is_device_capability((7, 0))
            ):
                return False, "requires exact CUDA capability 7.0"
        elif compute_capability != 70:
            return False, "requires exact CUDA capability 7.0"
        required = ("nvfp4_sm70_prepare", "nvfp4_gemm_sm70_out")
        missing = [name for name in required if not hasattr(torch.ops._C, name)]
        if missing:
            return False, "missing SM70 extension ops: " + ", ".join(missing)
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        del config
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        sm70_tm.prepare_nvfp4_linear(layer)
        layer.weight = Parameter(
            torch.empty(0, dtype=torch.uint8, device=layer.weight.device),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            torch.empty(
                0, dtype=layer.weight_scale.dtype, device=layer.weight_scale.device
            ),
            requires_grad=False,
        )
        logger.info_once("SM70 NVFP4 TurboMind base backend enabled.")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            return sm70_tm.apply_prepared_linear(layer, x, bias)
        out = sm70_tm.apply_prepared_linear(layer, x.to(torch.float16), bias)
        return out.to(torch.bfloat16)
