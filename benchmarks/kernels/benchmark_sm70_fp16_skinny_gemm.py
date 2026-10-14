# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark: SM70 FP16 skinny GEMM vs F.linear (cuBLAS) per shape and M.

Times every Qwen3.8 TP4 per-rank dense FP16 projection on ONE V100. Each arm
runs inside a CUDA graph of --calls calls (median over --replays replays);
the weight rotates over enough copies that every call misses L2 (--l2-mb), as
in a decode step, like v100-decode-analysis exp/dense_gemm_bench.py. Reports
us/call, effective weight GB/s and the speedup over F.linear, plus the max
abs error of the skinny output vs an FP64 reference (relative to ref max).

Arms:
  cublas       F.linear, FP32 reduction policy (no FP16 reduced-precision)
  skinny       fp16_skinny_gemm_sm70_out with the built-in plan (or the
               VLLM_SM70_FP16_SKINNY_CONFIG override)
  sweep        (--sweep) every valid (ng, warps_n, warps_k, splits) of a grid
               at --sweep-m; prints/stores the best plan per shape and a
               suggested VLLM_SM70_FP16_SKINNY_CONFIG string (one K partition
               per shape, minimum summed time over the swept M)
  lanes        (M <= 16) the existing qwen38 lanes for the same shapes:
               dense_batch (512x2560, 2560x1536), router_batch (packed 512x2560),
               gdn_input_batch (packed / row-major) for the fused GDN row
Composite "gdn_dual": qkvz 4096x2560 + ba 24x2560 -> qkv/z/b/a; cublas arm =
two F.linear + the production split copy; skinny arm = fp16_skinny_gemm2.

Example (V100, nothing else on the GPU):
  CUDA_VISIBLE_DEVICES=0 python benchmarks/kernels/benchmark_sm70_fp16_skinny_gemm.py \\
      --json /tmp/skinny.json --sweep --sweep-m 20 40 48
"""

import argparse
import itertools
import json
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F

SHAPES = [
    ("gdn_qkvz", 4096, 2560),
    ("gdn_ba", 24, 2560),
    ("out_proj", 2560, 1536),
    ("qsa_qkv", 3584, 2560),
    ("idx_fc", 640, 2560),
    ("router", 512, 2560),
    ("shared_gate_up", 320, 2560),
    ("shared_down", 2560, 160),
    ("shared_gate", 1, 2560),
    ("hc_down", 336, 10240),
    ("mixer_down", 320, 10240),
    ("hc_up", 10240, 320),
    ("ple_key", 10240, 2560),
    ("ple_value", 2560, 2560),
    ("lm_head", 62080, 2560),
]
DEFAULT_M = [5, 10, 15, 17, 20, 24, 25, 30, 32, 35, 40, 45, 48]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class Bench:
    def __init__(self, args):
        from vllm import _sm70_ops  # noqa: F401  (loads vllm._C)
        from vllm.model_executor.kernels.linear import sm70_fp16_skinny as skinny

        self.args = args
        self.skinny = skinny
        if not skinny.native_available():
            sys.exit("vllm._C lacks fp16_skinny_gemm_sm70_out")
        self.stream = torch.cuda.Stream()
        self.results = []

    def time_graph(self, fn):
        a = self.args
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                fn(i)
        s.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            for i in range(a.calls):
                fn(i)
        torch.cuda.synchronize()
        for _ in range(a.warmup):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(a.replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0 / a.calls)
        del graph
        return statistics.median(times), min(times)

    def copies(self, nbytes):
        return max(1, min(self.args.calls, math.ceil(self.args.l2_mb * 2**20 / nbytes)))

    def record(self, **kw):
        if kw.get("us") and kw.get("weight_bytes"):
            kw["gbps"] = kw["weight_bytes"] / (kw["us"] * 1e3)
        self.results.append(kw)
        log(
            f"  {kw['role']:15s} M={kw['m']:3d} {kw['impl']:24s} "
            f"{kw.get('us', float('nan')):8.2f} us {kw.get('gbps', 0):6.0f} GB/s"
            + (f"  err={kw['rel_err']:.2e}" if "rel_err" in kw else "")
            + (f"  {kw['error']}" if kw.get("error") else "")
        )

    def skinny_call(self, x, w, config, out, ws):
        torch.ops._C.fp16_skinny_gemm_sm70_out(out, x, w, None, ws, *config)

    def sweep_grid(self, n, k, m):
        a = self.args
        for ng, wn, wk, s in itertools.product(
            a.grid_ng, a.grid_wn, a.grid_wk, a.grid_s
        ):
            config = self.skinny.SkinnyConfig(ng, wn, wk, s)
            if self.skinny.config_is_valid(config, n, k, m):
                yield config

    def bench_shape(self, role, n, k, mlist):
        a = self.args
        skinny = self.skinny
        nbytes = n * k * 2
        nc = self.copies(nbytes)
        g = torch.Generator(device="cuda").manual_seed(n * 31 + k)
        ws_list = [
            (torch.randn(n, k, device="cuda", generator=g) * 0.02).half()
            for _ in range(nc)
        ]
        packed_router = None
        if (n, k) == (512, 2560) and hasattr(
            torch.ops._C, "qwen38_router_batch_sm70_out"
        ):
            packed_router = [
                w.reshape(64, 8, 4, 40, 2, 8).permute(0, 3, 4, 2, 1, 5).contiguous()
                for w in ws_list
            ]
        for m in mlist:
            x = torch.randn(m, k, device="cuda").half()
            base = dict(role=role, n=n, k=k, m=m, copies=nc, weight_bytes=nbytes)
            cub_us, _ = self.time_graph(
                lambda i, x=x, ws=ws_list: F.linear(x, ws[i % nc])
            )
            self.record(**base, impl="cublas", us=cub_us)
            config = skinny.select_config(n, k, m)
            if config is None:
                self.record(**base, impl="skinny", error="plan off")
                continue
            out = torch.empty(m, n, device="cuda", dtype=torch.half)
            wsp = torch.empty(
                config.splits * m * n if config.splits > 1 else 0,
                device="cuda",
                dtype=torch.float32,
            )
            self.skinny_call(x, ws_list[0], config, out, wsp)
            if n * k <= 64 * 2**20:
                ref = x.double() @ ws_list[0].double().t()
                rel = ((out.double() - ref).abs().max() / ref.abs().max()).item()
            else:
                rel = float("nan")
            us, _ = self.time_graph(
                lambda i, x=x, ws=ws_list, c=config, o=out, wsp=wsp: self.skinny_call(
                    x, ws[i % nc], c, o, wsp
                )
            )
            self.record(
                **base,
                impl="skinny",
                plan=list(config),
                us=us,
                speedup=cub_us / us,
                rel_err=rel,
            )
            if m <= 16:
                self.bench_lanes(base, x, ws_list, packed_router, nc)
            if a.sweep and m in a.sweep_m:
                for cfg in self.sweep_grid(n, k, m):
                    wsp2 = torch.empty(
                        cfg.splits * m * n if cfg.splits > 1 else 0,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    us2, _ = self.time_graph(
                        lambda i, x=x, ws=ws_list, cfg=cfg, o=out, wsp2=wsp2: (
                            self.skinny_call(x, ws[i % nc], cfg, o, wsp2)
                        )
                    )
                    self.results.append(
                        dict(
                            **base,
                            impl="sweep",
                            plan=list(cfg),
                            us=us2,
                            gbps=nbytes / (us2 * 1e3),
                        )
                    )
                best = min(
                    (
                        r
                        for r in self.results
                        if r["impl"] == "sweep" and r["role"] == role and r["m"] == m
                    ),
                    key=lambda r: r["us"],
                )
                log(
                    f"  {role:15s} M={m:3d} sweep best {best['plan']} "
                    f"{best['us']:.2f} us {best['gbps']:.0f} GB/s"
                )
        torch.cuda.empty_cache()

    def bench_lanes(self, base, x, ws_list, packed_router, nc):
        m = x.shape[0]
        n, k = base["n"], base["k"]
        if m < 2:
            return
        if (n, k) in ((512, 2560), (2560, 1536)) and hasattr(
            torch.ops._C, "qwen38_dense_batch_sm70_out"
        ):
            o = torch.empty(m, n, device="cuda", dtype=torch.half)
            us, _ = self.time_graph(
                lambda i, o=o: torch.ops._C.qwen38_dense_batch_sm70_out(
                    o, x, ws_list[i % nc]
                )
            )
            self.record(**base, impl="lane_dense_batch", us=us)
        if packed_router is not None:
            o = torch.empty(m, 512, device="cuda", dtype=torch.half)
            us, _ = self.time_graph(
                lambda i, o=o: torch.ops._C.qwen38_router_batch_sm70_out(
                    o, x, packed_router[i % nc]
                )
            )
            self.record(**base, impl="lane_router_batch", us=us)

    def bench_gdn_dual(self, mlist):
        skinny = self.skinny
        k = 2560
        nbytes = (4096 + 24) * k * 2
        nc = self.copies(nbytes)
        g = torch.Generator(device="cuda").manual_seed(4120)
        qs = [
            (torch.randn(4096, k, device="cuda", generator=g) * 0.02).half()
            for _ in range(nc)
        ]
        bs = [
            (torch.randn(24, k, device="cuda", generator=g) * 0.02).half()
            for _ in range(nc)
        ]
        have_lane = hasattr(torch.ops._C, "qwen38_gdn_input_batch_sm70_out")
        if have_lane:
            pq = [
                q.reshape(-1, 32, 160, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
                for q in qs
            ]
            pb = [
                torch.cat((b, b.new_zeros(8, k)))
                .reshape(-1, 32, 160, 2, 8)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
                for b in bs
            ]
        from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
            _split_gdn_projection_outputs,
        )

        for m in mlist:
            x = torch.randn(m, k, device="cuda").half()
            base = dict(
                role="gdn_dual", n=4120, k=k, m=m, copies=nc, weight_bytes=nbytes
            )

            def cub(i, x=x):
                _split_gdn_projection_outputs(
                    F.linear(x, qs[i % nc]), F.linear(x, bs[i % nc])
                )

            cub_us, _ = self.time_graph(cub)
            self.record(**base, impl="cublas+split", us=cub_us)
            config = skinny.select_config(4120, k, m)
            outs = tuple(
                torch.empty(m, c, device="cuda", dtype=torch.half)
                for c in (2560, 1536, 12, 12)
            )
            wsp = torch.empty(
                config.splits * m * 4120 if config.splits > 1 else 0,
                device="cuda",
                dtype=torch.float32,
            )

            def dual(i, x=x, outs=outs, wsp=wsp, config=config):
                torch.ops._C.fp16_skinny_gemm2_sm70_out(
                    *outs, x, qs[i % nc], bs[i % nc], wsp, *config
                )

            us, _ = self.time_graph(dual)
            self.record(
                **base,
                impl="skinny_dual",
                plan=list(config),
                us=us,
                speedup=cub_us / us,
            )
            if have_lane and 2 <= m <= 16:
                lane_out = tuple(
                    torch.empty(m, c, device="cuda", dtype=torch.half)
                    for c in (2560, 1536, 12, 12)
                )
                for name, wq, wb in (
                    ("lane_gdn_packed", pq, pb),
                    ("lane_gdn_rowmajor", qs, bs),
                ):

                    def lane(i, x=x, wq=wq, wb=wb, lane_out=lane_out):
                        torch.ops._C.qwen38_gdn_input_batch_sm70_out(
                            *lane_out, x, wq[i % nc], wb[i % nc]
                        )

                    us, _ = self.time_graph(lane)
                    self.record(**base, impl=name, us=us)
        torch.cuda.empty_cache()


def suggest(results, sweep_m):
    """Per shape: the plan with the least summed time over the swept M."""
    by_shape = {}
    for r in results:
        if r["impl"] != "sweep" or r["m"] not in sweep_m:
            continue
        key = (r["n"], r["k"])
        by_shape.setdefault(key, {}).setdefault(tuple(r["plan"]), {})[r["m"]] = r["us"]
    entries, table = [], {}
    for (n, k), plans in sorted(by_shape.items()):
        ms = set().union(*(set(v) for v in plans.values()))
        complete = {p: sum(v.values()) for p, v in plans.items() if set(v) == ms}
        if not complete:
            continue
        best = min(complete, key=complete.get)
        table[f"{n}x{k}"] = dict(
            plan=list(best),
            sum_us=complete[best],
            per_m={str(mm): plans[best][mm] for mm in sorted(ms)},
        )
        entries.append(f"{n}x{k}={','.join(map(str, best))}")
    return ";".join(entries), table


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--m", type=int, nargs="+", default=DEFAULT_M)
    p.add_argument(
        "--roles", nargs="*", default=[], help="subset of roles (+ gdn_dual)"
    )
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--l2-mb", type=float, default=64.0)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--sweep-m", type=int, nargs="+", default=[20, 40, 48])
    p.add_argument("--grid-ng", type=int, nargs="+", default=[1, 2])
    p.add_argument("--grid-wn", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--grid-wk", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--grid-s", type=int, nargs="+", default=[1, 2, 4, 6])
    p.add_argument("--json", default=None)
    a = p.parse_args()
    if a.sweep:
        a.m = sorted(set(a.m) | set(a.sweep_m))
    torch.cuda.set_device(0)
    matmul = torch.backends.cuda.matmul
    matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(matmul, "allow_fp16_accumulation"):
        matmul.allow_fp16_accumulation = False
    b = Bench(a)
    t0 = time.time()
    for role, n, k in SHAPES:
        if a.roles and role not in a.roles:
            continue
        log(f"== {role} N={n} K={k}")
        b.bench_shape(role, n, k, a.m)
    if not a.roles or "gdn_dual" in a.roles:
        log("== gdn_dual")
        b.bench_gdn_dual(a.m)
    log("\nrole            M   cublas_us skinny_us  GB/s  speedup")
    rows = {}
    for r in b.results:
        rows.setdefault((r["role"], r["m"]), {})[r["impl"]] = r
    for (role, m), arms in sorted(rows.items()):
        cub = arms.get("cublas") or arms.get("cublas+split")
        sk = arms.get("skinny") or arms.get("skinny_dual")
        if cub and sk and sk.get("us"):
            log(
                f"{role:15s} {m:3d} {cub['us']:9.2f} {sk['us']:9.2f} "
                f"{sk.get('gbps', 0):5.0f} {cub['us'] / sk['us']:7.2f}"
            )
    meta = dict(
        torch=torch.__version__,
        device=torch.cuda.get_device_name(0),
        calls=a.calls,
        replays=a.replays,
        l2_mb=a.l2_mb,
        m=a.m,
        elapsed_s=time.time() - t0,
    )
    out = dict(meta=meta, results=b.results)
    if a.sweep:
        spec, table = suggest(b.results, a.sweep_m)
        out["suggested_config"] = spec
        out["suggested_table"] = table
        log(f"\nsuggested VLLM_SM70_FP16_SKINNY_CONFIG={spec}")
    if a.json:
        with open(a.json, "w") as handle:
            json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
