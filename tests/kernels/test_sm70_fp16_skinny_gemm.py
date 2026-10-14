# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 small-M FP16 skinny GEMM (fp16_skinny_gemm_sm70_out / _gemm2_).

GPU checks (skipped without SM70 and the native op):
* every Qwen3.8 TP4 per-rank dense shape x M in {1, 2, 5, 8, 9, 15, 16, 17,
  20, 24, 25, 32, 33, 40, 41, 48} with the default plan against an FP64
  reference (max abs error <= 2e-3 of the reference max: one FP16 rounding
  of an FP32 sum) and against F.linear (FP32 reduction policy);
* row/batch invariance: every row of an M=40 call is bitwise equal to the
  same row computed in 8-row chunks, alone (M=1, row block position 0) and
  inside an M=48 call;
* ng / warps_n change only the work assignment: bitwise equal results for a
  fixed K partition (warps_k, splits); other partitions stay within tolerance;
* determinism over repeated calls, CUDA-graph capture/replay with changed
  inputs (single op with CTA split-K, dual op), bias, tail N, strided x,
  the dual (GDN qkvz + ba) op against two single calls, invalid inputs;
* the Python routes (env admission, opaque op, FP16 GEMV / GDN / HC / shared
  fallbacks, LM head) against the native op and the unchanged default path.

Run on a V100: pytest -s -v tests/kernels/test_sm70_fp16_skinny_gemm.py
(-s prints SKINNY_DIFF lines; SM70_SKINNY_REPORT=<file> appends JSON lines).
"""

import itertools
import json
import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70 and its native extension",
)

# (N, K) per-rank shapes: GDN qkvz/ba/out, QSA qkv/o, indexer + MTP fc,
# router, shared gate_up/down/gate, HC down/up, final mixer down, PLE key/value.
SHAPES = [
    (4096, 2560),
    (24, 2560),
    (2560, 1536),
    (3584, 2560),
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
]
M_LIST = [1, 2, 5, 8, 9, 15, 16, 17, 20, 24, 25, 32, 33, 40, 41, 48]
REPORT = os.environ.get("SM70_SKINNY_REPORT")
REL_TOL = 2e-3


def _report(**fields):
    print("SKINNY_DIFF " + " ".join(f"{k}={v}" for k, v in fields.items()))
    if REPORT:
        with open(REPORT, "a") as handle:
            handle.write(json.dumps(fields) + "\n")


def _skinny():
    from vllm import _sm70_ops  # noqa: F401  (loads vllm._C, registers fakes)
    from vllm.model_executor.kernels.linear import sm70_fp16_skinny as skinny

    if not skinny.native_available():
        pytest.skip("build lacks fp16_skinny_gemm_sm70_out")
    return skinny


@pytest.fixture(autouse=True)
def _fp32_policy():
    matmul = torch.backends.cuda.matmul
    saved = (
        matmul.allow_fp16_reduced_precision_reduction,
        getattr(matmul, "allow_fp16_accumulation", False),
    )
    matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(matmul, "allow_fp16_accumulation"):
        matmul.allow_fp16_accumulation = False
    yield
    matmul.allow_fp16_reduced_precision_reduction = saved[0]
    if hasattr(matmul, "allow_fp16_accumulation"):
        matmul.allow_fp16_accumulation = saved[1]


_WEIGHTS: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _weight(n, k):
    if (n, k) not in _WEIGHTS:
        g = torch.Generator(device="cuda").manual_seed(n * 7919 + k)
        w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).half()
        _WEIGHTS[(n, k)] = (w, w.double())
    return _WEIGHTS[(n, k)]


def _input(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, k, device="cuda", generator=g).half()


def _run(x, w, config, bias=None):
    m, n = x.shape[0], w.shape[0]
    out = torch.full((m, n), float("nan"), dtype=torch.float16, device="cuda")
    ws = torch.full(
        (max(config.splits, 1) * m * n if config.splits > 1 else 0,),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    torch.ops._C.fp16_skinny_gemm_sm70_out(out, x, w, bias, ws, *config)
    return out


def _check_ref(out, x, w64, bias=None):
    ref = x.double() @ w64.t()
    if bias is not None:
        ref = ref + bias.double()
    diff = (out.double() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert diff <= REL_TOL * scale + 2**-24, (diff, scale)
    return diff, scale


@pytest.mark.parametrize("m", M_LIST)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_against_fp64_and_linear(shape, m):
    skinny = _skinny()
    n, k = shape
    w, w64 = _weight(n, k)
    x = _input(m, k, seed=m * 131 + n)
    config = skinny.select_config(n, k, m)
    assert config is not None and skinny.config_is_valid(config, n, k, m)
    out = _run(x, w, config)
    torch.cuda.synchronize()
    diff, scale = _check_ref(out, x, w64)
    lin = F.linear(x, w)
    lin_diff = (out.float() - lin.float()).abs().max().item()
    _report(
        test="reference",
        n=n,
        k=k,
        m=m,
        plan=",".join(map(str, config)),
        max_abs=diff,
        ref_max=scale,
        vs_linear_max_abs=lin_diff,
        bitwise_linear=bool(torch.equal(out, lin)),
    )
    assert lin_diff <= REL_TOL * scale + 2**-24


@pytest.mark.parametrize("m", [17, 40, 48])
def test_lm_head_shape(m):
    skinny = _skinny()
    n, k = 62080, 2560
    g = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).half()
    x = _input(m, k, seed=m)
    config = skinny.select_config(n, k, m)
    out = _run(x, w, config)
    ref = torch.cat(
        [x.double() @ w[i : i + 8192].double().t() for i in range(0, n, 8192)], 1
    )
    diff = (out.double() - ref).abs().max().item()
    scale = ref.abs().max().item()
    _report(test="lm_head", m=m, max_abs=diff, ref_max=scale)
    assert diff <= REL_TOL * scale + 2**-24


INVARIANCE_SHAPES = [
    (4096, 2560),
    (2560, 1536),
    (336, 10240),
    (10240, 320),
    (2560, 160),
    (1, 2560),
    (512, 2560),
]


@pytest.mark.parametrize("shape", INVARIANCE_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_row_invariance(shape):
    skinny = _skinny()
    n, k = shape
    w, _ = _weight(n, k)
    x48 = _input(48, k, seed=4242)
    config = skinny.select_config(n, k, 40)
    # The default plans use one K partition for every M.
    for m in (1, 8, 17, 48):
        other = skinny.select_config(n, k, m)
        assert (other.warps_k, other.splits) == (config.warps_k, config.splits)
    full = _run(x48[:40], w, config)
    for j in range(5):
        chunk = _run(x48[8 * j : 8 * j + 8], w, config)
        assert torch.equal(chunk, full[8 * j : 8 * j + 8]), j
    for i in (0, 3, 7, 8, 17, 30, 39):
        row = _run(x48[i : i + 1].contiguous(), w, config)
        assert torch.equal(row[0], full[i]), i
    wide = _run(x48, w, config)
    assert torch.equal(wide[:40], full)
    # Rows are independent of their neighbours (different co-batched rows).
    mixed = x48[:40].clone()
    mixed[1::2] = _input(20, k, seed=99)
    mixed_out = _run(mixed, w, config)
    assert torch.equal(mixed_out[0::2], full[0::2])


def _valid_configs(skinny, n, k, m):
    for ng, wn, wk, s in itertools.product(
        (1, 2, 4), (1, 2, 4), (1, 2, 4, 8), (1, 2, 3, 6)
    ):
        config = skinny.SkinnyConfig(ng, wn, wk, s)
        if skinny.config_is_valid(config, n, k, m):
            yield config


@pytest.mark.parametrize(
    "shape,m",
    [
        ((4096, 2560), 40),
        ((336, 10240), 20),
        ((640, 2560), 24),
        ((24, 2560), 17),
        ((10240, 320), 48),
        ((2560, 160), 9),
    ],
    ids=lambda v: f"{v[0]}x{v[1]}" if isinstance(v, tuple) else str(v),
)
def test_work_assignment_invariance(shape, m):
    skinny = _skinny()
    n, k = shape
    w, w64 = _weight(n, k)
    x = _input(m, k, seed=n + m)
    by_partition: dict[tuple[int, int], torch.Tensor] = {}
    count = 0
    for config in _valid_configs(skinny, n, k, m):
        out = _run(x, w, config)
        _check_ref(out, x, w64)
        key = (config.warps_k, config.splits)
        if key in by_partition:
            assert torch.equal(out, by_partition[key]), config
        else:
            by_partition[key] = out
        count += 1
    _report(
        test="work_assignment",
        n=n,
        k=k,
        m=m,
        configs=count,
        partitions=len(by_partition),
    )
    assert count > 0


@pytest.mark.parametrize(
    "shape,config",
    [
        ((336, 10240), (1, 1, 4, 6)),
        ((4096, 2560), (1, 1, 2, 1)),
        ((512, 2560), (2, 2, 2, 4)),
    ],
    ids=["hc_down_split6", "qkvz", "router_split4"],
)
def test_determinism_and_graph_replay(shape, config):
    skinny = _skinny()
    n, k = shape
    w, _ = _weight(n, k)
    config = skinny.SkinnyConfig(*config)
    m = 40
    x = _input(m, k, seed=1)
    eager = [_run(x, w, config) for _ in range(5)]
    for out in eager[1:]:
        assert torch.equal(out, eager[0])

    static_x = x.clone()
    out = torch.empty(m, n, dtype=torch.float16, device="cuda")
    ws = torch.empty(config.splits * m * n, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C.fp16_skinny_gemm_sm70_out(out, static_x, w, None, ws, *config)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C.fp16_skinny_gemm_sm70_out(out, static_x, w, None, ws, *config)
    for seed in (2, 3, 4):
        new_x = _input(m, k, seed=seed)
        static_x.copy_(new_x)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _run(new_x, w, config)), seed


@pytest.mark.parametrize("m", [17, 20, 33, 40, 48])
def test_dual_gdn_op(m):
    skinny = _skinny()
    wq, wq64 = _weight(4096, 2560)
    wb, wb64 = _weight(24, 2560)
    x = _input(m, 2560, seed=m + 5)
    config = skinny.select_config(4120, 2560, m)
    for cfg in (config, skinny.SkinnyConfig(1, 1, 4, 2)):
        qkv, z, b, a = (
            torch.full((m, c), float("nan"), dtype=torch.float16, device="cuda")
            for c in (2560, 1536, 12, 12)
        )
        ws = torch.empty(cfg.splits * m * 4120, dtype=torch.float32, device="cuda")
        torch.ops._C.fp16_skinny_gemm2_sm70_out(qkv, z, b, a, x, wq, wb, ws, *cfg)
        single_q = _run(x, wq, cfg)
        single_b = _run(x, wb, cfg)
        # Same K partition and per-column arithmetic as two single calls.
        assert torch.equal(torch.cat((qkv, z), 1), single_q)
        assert torch.equal(torch.cat((b, a), 1), single_b)
        _check_ref(torch.cat((qkv, z), 1), x, wq64)
        _check_ref(torch.cat((b, a), 1), x, wb64)


def test_dual_graph_replay():
    _skinny()
    wq, _ = _weight(4096, 2560)
    wb, _ = _weight(24, 2560)
    m = 40
    static_x = _input(m, 2560, seed=10)
    outs = tuple(
        torch.empty((m, c), dtype=torch.float16, device="cuda")
        for c in (2560, 1536, 12, 12)
    )
    cfg = (1, 1, 4, 2)
    ws = torch.empty(2 * m * 4120, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C.fp16_skinny_gemm2_sm70_out(*outs, static_x, wq, wb, ws, *cfg)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C.fp16_skinny_gemm2_sm70_out(*outs, static_x, wq, wb, ws, *cfg)
    for seed in (11, 12):
        new_x = _input(m, 2560, seed=seed)
        static_x.copy_(new_x)
        graph.replay()
        torch.cuda.synchronize()
        expected = tuple(torch.empty_like(o) for o in outs)
        ws2 = torch.empty_like(ws)
        torch.ops._C.fp16_skinny_gemm2_sm70_out(*expected, new_x, wq, wb, ws2, *cfg)
        for got, want in zip(outs, expected):
            assert torch.equal(got, want)


@pytest.mark.parametrize("splits", [1, 3])
@pytest.mark.parametrize("m", [5, 33, 48])
def test_bias(m, splits):
    _skinny()
    n, k = 2560, 1536
    w, w64 = _weight(n, k)
    x = _input(m, k, seed=m)
    bias = (torch.randn(n, device="cuda") * 0.5).half()
    from vllm.model_executor.kernels.linear.sm70_fp16_skinny import SkinnyConfig

    out = _run(x, w, SkinnyConfig(1, 1, 2, splits), bias=bias)
    _, scale = _check_ref(out, x, w64, bias)
    lin = F.linear(x, w, bias)
    assert (out.float() - lin.float()).abs().max().item() <= REL_TOL * scale + 1e-6


@pytest.mark.parametrize("n", [1, 3, 8, 9, 17, 100, 333, 1000])
@pytest.mark.parametrize("k", [160, 2560])
def test_tail_n(n, k):
    skinny = _skinny()
    g = torch.Generator(device="cuda").manual_seed(n + k)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).half()
    for m in (5, 24, 41):
        x = _input(m, k, seed=n * m)
        for config in (
            skinny.SkinnyConfig(1, 1, 1, 1),
            skinny.SkinnyConfig(2, 2, 2, 1),
            skinny.SkinnyConfig(4, 1, 2, 2),
            skinny.SkinnyConfig(1, 4, 2, 3),
        ):
            if skinny.config_is_valid(config, n, k, m):
                _check_ref(_run(x, w, config), x, w.double())


def test_strided_input_rows():
    skinny = _skinny()
    n, k, m = 3584, 2560, 40
    w, _ = _weight(n, k)
    big = _input(m, k + 64, seed=3)
    x = big[:, 8 : 8 + k]  # row stride k + 64, 16-byte aligned offset
    assert not x.is_contiguous()
    config = skinny.select_config(n, k, m)
    assert torch.equal(_run(x, w, config), _run(x.contiguous(), w, config))


def test_invalid_inputs():
    _skinny()
    w, _ = _weight(512, 2560)
    x = _input(20, 2560, seed=0)
    x25 = _input(25, 2560, seed=1)
    out = torch.empty(20, 512, dtype=torch.float16, device="cuda")
    out25 = torch.empty(25, 512, dtype=torch.float16, device="cuda")
    ws = torch.empty(4 * 25 * 512, dtype=torch.float32, device="cuda")
    misaligned = _input(20, 2560 + 16, seed=2)[:, 1 : 1 + 2560]
    op = torch.ops._C.fp16_skinny_gemm_sm70_out
    ok = (1, 1, 4, 1)
    op(out, x, w, None, ws, *ok)
    op(out, x, w, None, ws, 4, 1, 4, 1)  # ng=4 is legal up to 24 rows
    bad_calls = {
        "m0": lambda: op(out[:0], x[:0], w, None, ws, *ok),
        "m49": lambda: op(
            torch.empty(49, 512, dtype=torch.float16, device="cuda"),
            _input(49, 2560, seed=3),
            w,
            None,
            ws,
            *ok,
        ),
        "k_not_16": lambda: op(
            out, x[:, :2552], w[:, :2552].contiguous(), None, ws, *ok
        ),
        "dtype": lambda: op(out, x.float(), w, None, ws, *ok),
        "weight_layout": lambda: op(out, x, w.t().contiguous().t(), None, ws, *ok),
        "device": lambda: op(out, x.cpu(), w, None, ws, *ok),
        "misaligned_x": lambda: op(out, misaligned, w, None, ws, *ok),
        "ng3": lambda: op(out, x, w, None, ws, 3, 1, 4, 1),
        "ng4_m25": lambda: op(out25, x25, w, None, ws, 4, 1, 4, 1),
        "16_warps": lambda: op(out, x, w, None, ws, 1, 4, 4, 1),
        "partitions": lambda: op(out, x, w, None, ws, 1, 1, 8, 32),
        "workspace": lambda: op(out, x, w, None, ws[:10], 1, 1, 4, 4),
        "out_shape": lambda: op(out[:, :500], x, w, None, ws, *ok),
        "bias_shape": lambda: op(
            out,
            x,
            w,
            torch.zeros(511, dtype=torch.float16, device="cuda"),
            ws,
            *ok,
        ),
    }
    accepted = []
    for name, call in bad_calls.items():
        try:
            call()
        except RuntimeError:
            continue
        accepted.append(name)
    assert not accepted, accepted


# ---------------------------------------------------------------- routes


@pytest.fixture
def skinny_env(monkeypatch):
    for name in (
        "VLLM_SM70_FP16_SKINNY",
        "VLLM_SM70_FP16_SKINNY_MIN_M",
        "VLLM_SM70_FP16_SKINNY_MAX_M",
        "VLLM_SM70_FP16_SKINNY_CONFIG",
        "VLLM_SM70_FP16_SKINNY_LM_HEAD",
    ):
        monkeypatch.delenv(name, raising=False)

    def enable(value="1", **extra):
        monkeypatch.setenv("VLLM_SM70_FP16_SKINNY", value)
        for key, val in extra.items():
            monkeypatch.setenv(f"VLLM_SM70_FP16_SKINNY_{key}", str(val))

    return enable


def test_route_admission_and_opaque_op(skinny_env):
    skinny = _skinny()
    n, k = 2560, 1536
    w, _ = _weight(n, k)
    for m in (1, 16, 17, 40, 48):
        x = _input(m, k, seed=m)
        assert skinny.skinny_linear(x, w) is None  # umbrella off
        assert torch.equal(
            torch.ops.vllm.sm70_fp16_skinny_linear(x, w, None), F.linear(x, w)
        )
    skinny_env()
    for m in (1, 16, 17, 40, 48):
        x = _input(m, k, seed=m)
        routed = skinny.skinny_linear(x, w)
        op_out = torch.ops.vllm.sm70_fp16_skinny_linear(x, w, None)
        if 17 <= m <= 48:
            expected = _run(x, w, skinny.select_config(n, k, m))
            assert routed is not None and torch.equal(routed, expected)
            assert torch.equal(op_out, expected)
        else:
            assert routed is None
            assert torch.equal(op_out, F.linear(x, w))
    skinny_env(MIN_M=2, MAX_M=20)
    x = _input(5, k, seed=5)
    assert skinny.skinny_linear(x, w) is not None
    assert skinny.skinny_linear(_input(21, k, seed=21), w) is None
    skinny_env(CONFIG=f"{n}x{k}=off")
    assert skinny.skinny_linear(x, w) is None
    # 3D input (MTP fc_hidden layout) is flattened to rows.
    skinny_env(CONFIG="", MIN_M=17, MAX_M=48)
    x3 = _input(20, 4 * k, seed=8).view(5, 4, k)
    out3 = skinny.skinny_linear(x3, w)
    assert out3.shape == (5, 4, n)
    assert torch.equal(
        out3.reshape(20, n), _run(x3.reshape(20, k), w, skinny.select_config(n, k, 20))
    )


def test_gemv_and_gdn_routes(skinny_env):
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv

    skinny = _skinny()
    w, _ = _weight(2560, 1536)
    wq, _ = _weight(4096, 2560)
    wb, _ = _weight(24, 2560)
    x = _input(40, 1536, seed=1)
    xg = _input(40, 2560, seed=2)
    default = gemv._qwen38_sm70_fp16_gemv(x, w, ".self_attn.o_proj")
    assert torch.equal(default, F.linear(x, w))
    default_gdn = gemv._qwen38_sm70_fp16_gdn_input(xg, wq, wb)
    skinny_env()
    routed = gemv._qwen38_sm70_fp16_gemv(x, w, ".self_attn.o_proj")
    assert torch.equal(routed, _run(x, w, skinny.select_config(2560, 1536, 40)))
    qkv, z, b, a = gemv._qwen38_sm70_fp16_gdn_input(xg, wq, wb)
    cfg = skinny.select_config(4120, 2560, 40)
    single_q = _run(xg, wq, cfg)
    single_b = _run(xg, wb, cfg)
    assert torch.equal(qkv, single_q[:, :2560]) and torch.equal(z, single_q[:, 2560:])
    assert torch.equal(b, single_b[:, :12]) and torch.equal(a, single_b[:, 12:])
    for got, base in zip((qkv, z, b, a), default_gdn):
        assert got.shape == base.shape and got.is_contiguous()
        scale = base.float().abs().max().item()
        assert (got.float() - base.float()).abs().max().item() <= 2 * REL_TOL * scale


def test_hc_and_shared_routes(skinny_env):
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc
    from vllm.models.qwen4_exp.nvidia.ops import hc as hc_ops  # noqa: F401

    _skinny()
    down, _ = _weight(336, 10240)
    up, _ = _weight(10240, 320)
    shared, _ = _weight(320, 2560)
    x = _input(40, 10240, seed=3)
    xs = _input(40, 2560, seed=4)
    base_block, base_inj = hc._qwen38_sm70_fp16_fused_hc(x, down, up)
    base_shared = gemv._qwen38_sm70_shared_up(xs, shared, None)
    skinny_env()
    block, inj = hc._qwen38_sm70_fp16_fused_hc(x, down, up)
    shared_out = gemv._qwen38_sm70_shared_up(xs, shared, None)
    for got, base in ((block, base_block), (inj, base_inj), (shared_out, base_shared)):
        assert got.shape == base.shape
        scale = base.float().abs().max().item()
        # FP16 boundaries between the GEMMs can amplify one rounding step.
        assert (got.float() - base.float()).abs().max().item() <= 1e-2 * scale + 1e-3


def test_linear_method_and_lm_head_routes(skinny_env):
    from types import SimpleNamespace

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_skinny import (
        Sm70Fp16SkinnyLinearMethod,
    )

    skinny = _skinny()
    w, _ = _weight(10240, 320)
    x = _input(40, 320, seed=6)
    layer = SimpleNamespace(weight=w, prefix="model.norm_mixer.input_mix_weight_up")
    method = Sm70Fp16SkinnyLinearMethod()
    assert torch.equal(method.apply(layer, x), F.linear(x, w))
    skinny_env()
    expected = _run(x, w, skinny.select_config(10240, 320, 40))
    assert torch.equal(method.apply(layer, x), expected)
    head_w, _ = _weight(2560, 2560)  # stand-in vocab shard
    xh = _input(40, 2560, seed=7)
    head = SimpleNamespace(weight=head_w, prefix="lm_head")
    embed = UnquantizedEmbeddingMethod()
    assert torch.equal(embed.apply(head, xh), F.linear(xh, head_w))
    head._sm70_fp16_skinny_lm_head = True
    assert torch.equal(
        embed.apply(head, xh), _run(xh, head_w, skinny.select_config(2560, 2560, 40))
    )
