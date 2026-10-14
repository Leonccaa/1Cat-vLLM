# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the SM70 FP16 skinny GEMM admission, plans and routing.

No GPU arithmetic: the native launchers are replaced by an FP64 emulation so
the env switches, row range, plan table/overrides, the opaque op, the FP16
lane fallbacks (GEMV, GDN input), the linear/LM-head methods and the loader
marking can be checked on a GPU-less host. Numerical evidence lives in
tests/kernels/test_sm70_fp16_skinny_gemm.py.
"""

import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm import envs
from vllm.model_executor.kernels.linear import sm70_fp16_skinny as skinny

pytestmark = pytest.mark.skip_global_cleanup

ENV_NAMES = (
    "VLLM_SM70_FP16_SKINNY",
    "VLLM_SM70_FP16_SKINNY_MIN_M",
    "VLLM_SM70_FP16_SKINNY_MAX_M",
    "VLLM_SM70_FP16_SKINNY_CONFIG",
    "VLLM_SM70_FP16_SKINNY_LM_HEAD",
)
INVENTORY = [
    (4120, 2560),
    (4096, 2560),
    (24, 2560),
    (3584, 2560),
    (2560, 1536),
    (640, 2560),
    (512, 2560),
    (320, 2560),
    (2560, 160),
    (1, 2560),
    (336, 10240),
    (320, 10240),
    (10240, 320),
    (10240, 2560),
    (2560, 2560),
    (62080, 2560),
]


@pytest.fixture
def env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)

    def set_env(**values):
        for key, value in values.items():
            name = "VLLM_SM70_FP16_SKINNY" + ("" if key == "ON" else f"_{key}")
            monkeypatch.setenv(name, str(value))

    return set_env


@pytest.fixture
def fake_native(monkeypatch):
    """Admit CPU tensors and emulate both native ops in FP64."""
    calls = []

    def launch(out, x2d, weight, bias, workspace, config):
        m, n = out.shape
        assert x2d.stride(-1) == 1 and x2d.data_ptr() % 16 == 0
        assert workspace.dtype == torch.float32
        assert workspace.numel() == (config.splits * m * n if config.splits > 1 else 0)
        assert skinny.config_is_valid(config, n, weight.shape[1], m)
        ref = x2d.double() @ weight.double().t()
        if bias is not None:
            ref = ref + bias.double()
        out.copy_(ref.half())
        calls.append(("single", tuple(weight.shape), m, config))

    def launch2(outs, x2d, w0, w1, workspace, config):
        m = x2d.shape[0]
        y0 = (x2d.double() @ w0.double().t()).half()
        y1 = (x2d.double() @ w1.double().t()).half()
        s0, s1 = outs[0].shape[1], outs[2].shape[1]
        for out, src in zip(outs, (y0[:, :s0], y0[:, s0:], y1[:, :s1], y1[:, s1:])):
            assert out.is_contiguous() and out.shape == src.shape
            out.copy_(src)
        n = w0.shape[0] + w1.shape[0]
        assert workspace.numel() == (config.splits * m * n if config.splits > 1 else 0)
        calls.append(("dual", (tuple(w0.shape), tuple(w1.shape)), m, config))

    monkeypatch.setattr(skinny, "_device_ok", lambda t: True)
    monkeypatch.setattr(skinny, "_is_sm70", lambda: True)
    monkeypatch.setattr(skinny, "native_available", lambda: True)
    monkeypatch.setattr(skinny, "_launch", launch)
    monkeypatch.setattr(skinny, "_launch2", launch2)
    return calls


def _half(*shape, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).half()


def test_env_defaults(env):
    assert envs.VLLM_SM70_FP16_SKINNY is False
    assert envs.VLLM_SM70_FP16_SKINNY_MIN_M == 17
    assert envs.VLLM_SM70_FP16_SKINNY_MAX_M == 48
    assert envs.VLLM_SM70_FP16_SKINNY_CONFIG is None
    assert envs.VLLM_SM70_FP16_SKINNY_LM_HEAD is False


def test_row_range(env):
    assert not any(skinny.rows_admitted(m) for m in range(0, 60))
    env(ON=1)
    assert [m for m in range(0, 60) if skinny.rows_admitted(m)] == list(range(17, 49))
    env(MIN_M=2, MAX_M=100)
    assert skinny.row_range() == (2, 48)
    env(MIN_M=0, MAX_M=20)
    assert [m for m in range(0, 60) if skinny.rows_admitted(m)] == list(range(1, 21))
    env(ON=0)
    assert not skinny.rows_admitted(20)


@pytest.mark.parametrize("shape", INVENTORY, ids=lambda s: f"{s[0]}x{s[1]}")
def test_default_plans_valid_and_batch_invariant(env, shape):
    n, k = shape
    partitions = set()
    for m in range(1, 49):
        config = skinny.select_config(n, k, m)
        assert config is not None and skinny.config_is_valid(config, n, k, m), m
        partitions.add((config.warps_k, config.splits))
    # One K partition for every M: the FP32 association is M independent.
    assert len(partitions) == 1


def test_heuristic_plans_valid(env):
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.choice([1, 2, 7, 8, 9, 31, 100, 640, 1000, 4096, 70000])
        k = 16 * rng.randint(1, 1280)
        m = rng.randint(1, 48)
        config = skinny._heuristic_config(n, k)
        assert skinny.config_is_valid(config, n, k, m), (n, k, m, config)


def test_overrides(env):
    env(CONFIG="512x2560=2,2,2,4; 512x2560@17-24=1,1,8,1 ;336x10240=off")
    # First matching entry wins (the full-range entry precedes the band).
    assert skinny.select_config(512, 2560, 20) == skinny.SkinnyConfig(2, 2, 2, 4)
    assert skinny.select_config(336, 10240, 40) is None
    assert skinny.select_config(2560, 1536, 40) == skinny._DEFAULT_PLANS[(2560, 1536)]
    env(CONFIG="512x2560@17-24=1,1,8,1;512x2560=2,2,2,4")
    assert skinny.select_config(512, 2560, 20) == skinny.SkinnyConfig(1, 1, 8, 1)
    assert skinny.select_config(512, 2560, 40) == skinny.SkinnyConfig(2, 2, 2, 4)
    for bad in ("512x2560", "512x2560=1,2,3", "x=1,1,1,1", "512x2560@a-b=1,1,1,1"):
        env(CONFIG=bad)
        with pytest.raises(ValueError):
            skinny.select_config(512, 2560, 20)
    env(CONFIG="512x2560=4,1,1,1")  # ng=4 is limited to 24 rows
    assert skinny.select_config(512, 2560, 24) == skinny.SkinnyConfig(4, 1, 1, 1)
    with pytest.raises(ValueError):
        skinny.select_config(512, 2560, 25)


def test_skinny_linear_routing(env, fake_native, monkeypatch):
    w = _half(2560, 1536, seed=1, scale=0.02)
    x = _half(40, 1536, seed=2)
    assert skinny.skinny_linear(x, w) is None
    assert not fake_native
    env(ON=1)
    out = skinny.skinny_linear(x, w)
    assert out is not None and out.shape == (40, 2560)
    assert torch.equal(out, (x.double() @ w.double().t()).half())
    assert fake_native[-1] == (
        "single",
        (2560, 1536),
        40,
        skinny._DEFAULT_PLANS[(2560, 1536)],
    )
    assert skinny.skinny_linear(x[:16], w) is None
    assert skinny.skinny_linear(torch.cat((x, x[:9])), w) is None  # 49 rows
    # Bias (FP16 [N]) is passed through; any other bias keeps the fallback.
    bias = _half(2560, seed=3)
    out = skinny.skinny_linear(x, w, bias)
    assert torch.equal(out, (x.double() @ w.double().t() + bias.double()).half())
    assert skinny.skinny_linear(x, w, bias.float()) is None
    # Strided / misaligned rows are copied into an aligned buffer.
    big = _half(40, 1536 + 24, seed=4)
    strided = big[:, 4 : 4 + 1536]
    out = skinny.skinny_linear(strided, w)
    assert torch.equal(out, (strided.double() @ w.double().t()).half())
    # 3D inputs are flattened over the leading dimensions.
    x3 = _half(5, 4, 1536, seed=5)
    out3 = skinny.skinny_linear(x3, w)
    assert out3.shape == (5, 4, 2560)
    # Unsupported dtypes, K, weights and policies fall back.
    assert skinny.skinny_linear(x.float(), w.float()) is None
    assert skinny.skinny_linear(x[:, :1528], w[:, :1528].contiguous()) is None
    assert skinny.skinny_linear(x, w.t().contiguous().t()) is None
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", True)
    assert skinny.skinny_linear(x, w) is None
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert skinny.skinny_linear(x, w) is None
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    env(CONFIG="2560x1536=off")
    assert skinny.skinny_linear(x, w) is None
    env(CONFIG="2560x1536=1,2,4,3")
    skinny.skinny_linear(x, w)
    assert fake_native[-1][3] == skinny.SkinnyConfig(1, 2, 4, 3)


def test_or_fallback_and_opaque_op(env, fake_native):
    w = _half(512, 2560, seed=6, scale=0.02)
    for m in (1, 16, 17, 48):
        x = _half(m, 2560, seed=m)
        assert torch.equal(skinny.skinny_linear_or_fallback(x, w), F.linear(x, w))
        assert torch.equal(
            torch.ops.vllm.sm70_fp16_skinny_linear(x, w, None), F.linear(x, w)
        )
    assert not fake_native
    env(ON=1)
    x = _half(20, 2560, seed=7)
    assert torch.equal(
        torch.ops.vllm.sm70_fp16_skinny_linear(x, w, None),
        (x.double() @ w.double().t()).half(),
    )
    fake = skinny._sm70_fp16_skinny_linear_fake(x.view(4, 5, 2560), w, None)
    assert fake.shape == (4, 5, 512)


def test_dual_routing(env, fake_native):
    wq = _half(4096, 2560, seed=8, scale=0.02)
    wb = _half(24, 2560, seed=9, scale=0.02)
    x = _half(20, 2560, seed=10)
    assert skinny.skinny_dual_linear(x, wq, wb, 2560, 12) is None
    env(ON=1)
    qkv, z, b, a = skinny.skinny_dual_linear(x, wq, wb, 2560, 12)
    assert [t.shape for t in (qkv, z, b, a)] == [
        (20, 2560),
        (20, 1536),
        (20, 12),
        (20, 12),
    ]
    assert fake_native[-1][3] == skinny._DEFAULT_PLANS[(4120, 2560)]
    assert torch.equal(torch.cat((qkv, z), 1), (x.double() @ wq.double().t()).half())
    assert skinny.skinny_dual_linear(x, wq, wb, 0, 12) is None
    assert skinny.skinny_dual_linear(x, wq, wb, 2560, 24) is None
    assert skinny.skinny_dual_linear(x[:10], wq, wb, 2560, 12) is None


def test_fp16_lane_fallbacks(env, fake_native):
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv

    w = _half(2560, 1536, seed=11, scale=0.02)
    x = _half(40, 1536, seed=12)
    default = gemv._qwen38_sm70_fp16_gemv(x, w, ".self_attn.o_proj")
    assert torch.equal(default, F.linear(x, w)) and not fake_native
    env(ON=1)
    routed = gemv._qwen38_sm70_fp16_gemv(x, w, ".self_attn.o_proj")
    assert torch.equal(routed, (x.double() @ w.double().t()).half())
    assert fake_native[-1][:3] == ("single", (2560, 1536), 40)
    # Rows below the range keep the existing route.
    small = _half(8, 1536, seed=13)
    assert torch.equal(
        gemv._qwen38_sm70_fp16_gemv(small, w, ".self_attn.o_proj"), F.linear(small, w)
    )
    wq = _half(4096, 2560, seed=14, scale=0.02)
    wb = _half(24, 2560, seed=15, scale=0.02)
    xg = _half(40, 2560, seed=16)
    qkv, z, b, a = gemv._qwen38_sm70_fp16_gdn_input(xg, wq, wb)
    assert fake_native[-1][0] == "dual"
    assert [t.shape for t in (qkv, z, b, a)] == [
        (40, 2560),
        (40, 1536),
        (40, 12),
        (40, 12),
    ]
    assert torch.equal(b, (xg.double() @ wb[:12].double().t()).half())


def _linear(cls, prefix, shape, method=None):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.prefix = prefix
    layer.weight = nn.Parameter(
        torch.zeros(shape, dtype=torch.float16), requires_grad=False
    )
    layer.quant_method = method if method is not None else UnquantizedLinearMethod()
    return layer


def test_linear_and_lm_head_methods(env, fake_native, monkeypatch):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_skinny import (
        Sm70Fp16SkinnyLinearMethod,
    )

    w = _half(10240, 320, seed=17, scale=0.02)
    x = _half(24, 320, seed=18)
    layer = SimpleNamespace(weight=w, prefix="model.norm.input_mix_weight_up")
    method = Sm70Fp16SkinnyLinearMethod()
    assert torch.equal(method.apply(layer, x), F.linear(x, w))
    env(ON=1)
    assert torch.equal(method.apply(layer, x), (x.double() @ w.double().t()).half())
    # Prepared TurboMind / online-QPN8 layers keep the base method's route.
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    monkeypatch.setattr(
        UnquantizedLinearMethod, "apply", lambda self, layer, x, bias=None: "base"
    )
    layer._sm70_f16_prepared = True
    assert method.apply(layer, x) == "base"
    layer._sm70_f16_prepared = False
    layer._sm70_qwen4_exp_online_qpn8 = True
    assert method.apply(layer, x) == "base"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    layer._sm70_qwen4_exp_online_qpn8 = False
    assert method.apply(layer, x) == "base"
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")

    head_w = _half(640, 2560, seed=19, scale=0.02)
    xh = _half(40, 2560, seed=20)
    # cpu_linear: the CPU platform's dispatch_unquantized_gemm hook.
    head = SimpleNamespace(weight=head_w, prefix="lm_head", cpu_linear=F.linear)
    embed = UnquantizedEmbeddingMethod()
    assert torch.equal(embed.apply(head, xh), F.linear(xh, head_w))
    head._sm70_fp16_skinny_lm_head = True
    assert torch.equal(
        embed.apply(head, xh), (xh.double() @ head_w.double().t()).half()
    )


def test_enable_marks_only_plain_role_projections(env, monkeypatch):
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_skinny as wiring
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import Qwen38SM70FP16LinearMethod

    class Root(nn.Module):
        pass

    root = Root()
    root.mixer_down = _linear(
        ReplicatedLinear, "model.norm.input_mix_weight_down", (320, 10240)
    )
    root.mixer_up = _linear(
        ReplicatedLinear, "model.norm.input_mix_weight_up", (10240, 320)
    )
    root.shared_down = _linear(
        RowParallelLinear, "model.layers.0.mlp.shared_expert.down_proj", (2560, 160)
    )
    root.gate = _linear(
        ReplicatedLinear, "model.layers.0.mlp.shared_expert_gate", (1, 2560)
    )
    root.ple_key = _linear(
        ReplicatedLinear, "model.layers.3.ple.key_proj", (10240, 2560)
    )
    root.fc = _linear(MergedColumnParallelLinear, "mtp.fc_hidden", (640, 2560))
    root.router = _linear(
        ReplicatedLinear,
        "model.layers.0.mlp.gate",
        (512, 2560),
        Qwen38SM70FP16LinearMethod(),
    )
    root.vision = _linear(ReplicatedLinear, "visual.blocks.0.mlp.fc1", (1024, 1024))
    root.odd_k = _linear(
        ReplicatedLinear, "model.layers.0.mlp.shared_expert_gate", (1, 2568)
    )
    root.head = ParallelLMHead.__new__(ParallelLMHead)
    nn.Module.__init__(root.head)

    assert wiring.enable_sm70_fp16_skinny(root, torch.float16) == 0
    assert all(
        type(layer.quant_method) is not wiring.Sm70Fp16SkinnyLinearMethod
        for layer in root.children()
        if hasattr(layer, "quant_method")
    )
    env(ON=1)
    # Not SM70: rejected, nothing replaced.
    monkeypatch.setattr(
        wiring.current_platform, "is_device_capability", lambda *a, **k: False
    )
    assert wiring.enable_sm70_fp16_skinny(root, torch.float16) == 0
    monkeypatch.setattr(
        wiring.current_platform, "is_device_capability", lambda *a, **k: True
    )
    assert wiring.enable_sm70_fp16_skinny(root, torch.bfloat16) == 0
    assert wiring.enable_sm70_fp16_skinny(root, torch.float16) == 6
    for name in ("mixer_down", "mixer_up", "shared_down", "gate", "ple_key", "fc"):
        assert (
            type(getattr(root, name).quant_method) is wiring.Sm70Fp16SkinnyLinearMethod
        )
    assert type(root.router.quant_method) is Qwen38SM70FP16LinearMethod
    assert type(root.vision.quant_method) is UnquantizedLinearMethod
    assert type(root.odd_k.quant_method) is UnquantizedLinearMethod
    assert not getattr(root.head, "_sm70_fp16_skinny_lm_head", False)
    env(LM_HEAD=1)
    assert wiring.enable_sm70_fp16_skinny(root, torch.float16) == 0  # already marked
    assert root.head._sm70_fp16_skinny_lm_head is True


def test_fused_hc_accepts_skinny_method(monkeypatch):
    """Marking HC linears for the skinny method keeps the fused HC route."""
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_skinny import (
        Sm70Fp16SkinnyLinearMethod,
    )

    monkeypatch.setattr(hc, "use_sm70_decode_graph_semantics", lambda: True)
    calls = []

    def record(*args):
        calls.append(args)
        return args[0], args[0]

    monkeypatch.setattr(torch.ops.vllm, "qwen38_sm70_fp16_fused_hc", record)
    down = _linear(
        LinearBase,
        "model.layers.0.attn_hc.input_mix_weight_down_block_inject",
        (336, 10240),
        Sm70Fp16SkinnyLinearMethod(),
    )
    up = _linear(
        LinearBase,
        "model.layers.0.attn_hc.input_mix_weight_up",
        (10240, 320),
        Sm70Fp16SkinnyLinearMethod(),
    )
    x = torch.empty(1, 10240, dtype=torch.float16)
    assert hc.maybe_apply_qwen38_sm70_fp16_fused_hc(down, up, x, True) is not None
    assert len(calls) == 1
