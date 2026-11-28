#!/usr/bin/env python3
"""Compile-and-time probe for fusion resource cliffs (exploratory).

Not part of the 72-case corpus. Each (variant, shape, role) runs in its own
process with its own cache/trace directory, mirroring
``strategies.compile_variant``: same pinned Inductor knobs, same evidence
extraction. ``role=split`` uses ``strategies.split_context`` (lowering
``realize`` + scheduler veto); ``role=fused`` compiles the same entry without
the guard.

Variants
--------
``expand``    ``t = sum(x, 1); y = t[:, None] * x`` — the consumer re-reads the
              producer input, so a fused kernel keeps more state live.
``expand2``   ``t = sum(x, 1); y = t[:, None] * x + t[:, None] * t[:, None]``.
``chain``     ``t = sum_b(x * a_b + c_b); y = relu(t)`` with ``b = --branches``
              independent branches (parallel live values, unlike a serial chain).
``multi``     ``t = stack([x[:, i::m].sum(1)], 1); y = (t * w).sum(1)`` with
              ``m = --branches`` accumulators feeding a second reduction.

The probe writes one JSON per run with kernel resources (block, warps, regs,
spills, occupancy, shared) and single-role timing; paired timing is a separate
step once a resource difference exists.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True,
                   choices=("expand", "expand2", "chain", "multi", "wide"))
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--dtype", required=True, choices=("float32", "float16"))
    p.add_argument("--branches", type=int, default=8,
                   help="chain branches or multi accumulators")
    p.add_argument("--role", required=True, choices=("fused", "split"))
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--per-block", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    tag = f"{args.variant}-M{args.M}-K{args.K}-{args.dtype}-b{args.branches}"
    cache = out / "cache" / tag / args.role
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)

    import torch

    from fusionboundary import forensics, templates
    from fusionboundary import strategies as S

    dtype = getattr(torch, args.dtype)
    M, K, B = args.M, args.K, args.branches

    if args.variant in ("expand", "expand2"):
        def fn(x):
            t = torch.sum(x, dim=1, dtype=torch.float32)
            y = t[:, None] * x
            if args.variant == "expand2":
                y = y + t[:, None] * t[:, None]
            return y
        spec = templates.EdgeSpec(template="EXP", producer_ops=("sum",),
                                  consumer_ops=("mul",), t_shape=(M,),
                                  t_dtype="float32")
    elif args.variant == "chain":
        a = [1.0 + 0.01 * i for i in range(B)]
        c = [0.001 * i for i in range(B)]

        def fn(x):
            t = x * a[0] + c[0]
            for i in range(1, B):
                t = t + (x * a[i] + c[i])
            return torch.relu(t)
        spec = templates.EdgeSpec(template="CHAIN", producer_ops=("add",),
                                  consumer_ops=("relu",), t_shape=(M, K),
                                  t_dtype=args.dtype)
    elif args.variant == "wide":
        # Each roll is a distinct load expression: num_load + num_reduction
        # crosses 10, which makes _reduction_configs cap RBLOCK/num_warps
        # (register_intensive) in the fused kernel only.
        shifts = list(range(1, B + 1))

        def fn(x):
            t = torch.sum(x, dim=1, dtype=torch.float32)
            acc = x
            for s in shifts:
                acc = acc + torch.roll(x, s, dims=1)
            return t[:, None] * acc
        spec = templates.EdgeSpec(template="WIDE", producer_ops=("sum",),
                                  consumer_ops=("mul",), t_shape=(M,),
                                  t_dtype="float32")
    else:  # multi
        w = torch.arange(1, B + 1, dtype=torch.float32, device="cuda")

        def fn(x):
            t = torch.stack([x[:, i::B].sum(dim=1, dtype=torch.float32)
                             for i in range(B)], dim=1)
            return (t * w).sum(dim=1)
        spec = templates.EdgeSpec(template="MULTI", producer_ops=("sum",),
                                  consumer_ops=("sum_1", "sum"), t_shape=(M, B),
                                  t_dtype="float32")

    case = SimpleNamespace(case_id=tag, template=spec.template,
                           t_shape=spec.t_shape, t_dtype=spec.t_dtype)
    original_edge_spec = templates.edge_spec
    templates.edge_spec = lambda _case: spec  # split_guard resolves it at call time

    x = torch.empty((M, K), dtype=dtype, device="cuda").uniform_(-1, 1)
    with torch.no_grad():
        ref = fn(x)
    torch.cuda.synchronize()

    trace_dir = out / "trace" / tag / args.role
    trace_dir.mkdir(parents=True, exist_ok=True)

    def _compile():
        with S.inductor_environment(str(trace_dir)):
            compiled = torch.compile(fn, backend="inductor", dynamic=False,
                                     fullgraph=True)
            with torch.no_grad():
                compiled(x)
            torch.cuda.synchronize()
        return compiled

    def _collect() -> dict:
        trace = forensics.collect_trace_artifacts(str(trace_dir))
        code = trace.get("output_code")
        if not code:
            raise RuntimeError(f"no output_code.py under {trace_dir}")
        frozen = out / f"{tag}__{args.role}_output_code.py"
        shutil.copyfile(code, frozen)
        parsed = forensics.parse_output_code(str(frozen))
        names = [k.name for k in parsed.kernels]
        return {"output_code": str(frozen),
                "kernels": [{"name": k.name, "original_aten": k.original_aten}
                            for k in parsed.kernels],
                "resources": S.extract_kernel_resources(names),
                "parse_notes": parsed.parse_notes}

    def _bench(compiled) -> dict:
        with torch.no_grad():
            for _ in range(args.warmup):
                compiled(x)
            torch.cuda.synchronize()
            per = []
            for _ in range(args.blocks):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(args.per_block):
                    compiled(x)
                e.record()
                torch.cuda.synchronize()
                per.append(s.elapsed_time(e) * 1e3 / args.per_block)
        return {"us_mean": statistics.fmean(per), "us_median": statistics.median(per),
                "us_min": min(per), "blocks": args.blocks,
                "per_block": args.per_block}

    result: dict = {"role": args.role, "variant": args.variant, "M": M, "K": K,
                    "dtype": args.dtype, "branches": B,
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__, "triton": __import__("triton").__version__,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        if args.role == "split":
            with S.split_context(case) as (guard, log):
                compiled = _compile()
                result["guard_blocks"] = len(guard.log)
                result["realize_log"] = log
        else:
            compiled = _compile()
        with torch.no_grad():
            got = compiled(x)
        torch.cuda.synchronize()
        result["graphs"] = [_collect()]
        result["numerics"] = {
            "max_abs_err": (got.float() - ref.float()).abs().max().item(),
            "ok": bool(torch.allclose(got.float(), ref.float(), atol=2e-3, rtol=2e-3)),
        }
        result["timing"] = _bench(compiled)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        templates.edge_spec = original_edge_spec

    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = out / "json" / f"{tag}__{args.role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: result[k] for k in
                      ("variant", "role", "M", "K", "dtype", "branches",
                       "timing", "numerics", "error") if k in result}))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
