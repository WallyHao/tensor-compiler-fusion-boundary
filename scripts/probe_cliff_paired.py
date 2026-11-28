#!/usr/bin/env python3
"""Paired A/B timing for two archived Inductor wrappers (cliff probe).

Loads the ``output_code.py`` produced by ``probe_cliff.py`` for the fused and
split variants of the same entry and times them in one process, block by block,
alternating order. Compilation happens in the probe processes; this harness
only loads the archived wrappers, so no Dynamo/Inductor cache key can leak
between the two sides (the same reason ``runtime.CandidateStore`` loads bytes).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fused-code", required=True)
    p.add_argument("--split-code", required=True)
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--dtype", required=True, choices=("float32", "float16"))
    p.add_argument("--branches", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--blocks", type=int, default=30)
    p.add_argument("--per-block", type=int, default=10)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--profile", action="store_true",
                   help="record per-kernel CUDA self time for both sides")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Kernel compilation (Triton) is shared with the probe caches: point the
    # environment at the parent run's cache so loading does not recompile.
    tag = f"wide-M{args.M}-K{args.K}-{args.dtype}-b{args.branches}"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(out / "cache" / tag / "paired"))

    import torch
    from torch._inductor.codecache import PyCodeCache

    from fusionboundary import strategies as S

    x = torch.empty((args.M, args.K), dtype=getattr(torch, args.dtype),
                    device="cuda").uniform_(-1, 1)
    fused_src = Path(args.fused_code).read_text()
    split_src = Path(args.split_code).read_text()

    with S.inductor_environment(None):
        fused_mod = PyCodeCache.load(fused_src)
        split_mod = PyCodeCache.load(split_src)

    def fused():
        return fused_mod.call([x])[0]

    def split():
        return split_mod.call([x])[0]

    with torch.no_grad():
        y_f = fused()
        y_s = split()
    torch.cuda.synchronize()
    same = bool(torch.allclose(y_f.float(), y_s.float(), atol=2e-3, rtol=2e-3))

    def block_time(fn) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.per_block):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / args.per_block

    with torch.no_grad():
        for _ in range(args.warmup):
            fused()
            split()
    torch.cuda.synchronize()

    pairs = []
    for block in range(args.blocks):
        if block % 2 == 0:
            s = block_time(split)
            f = block_time(fused)
        else:
            f = block_time(fused)
            s = block_time(split)
        pairs.append((s, f))

    profile_times = None
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        def kernel_times(fn):
            with torch.no_grad():
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        fn()
                    torch.cuda.synchronize()
            out = {}
            for evt in prof.key_averages():
                if evt.self_device_time_total > 0 and "triton" in evt.key:
                    out[evt.key] = evt.self_device_time_total / 5.0
            return out

        profile_times = {"fused": kernel_times(fused), "split": kernel_times(split)}

    b_us = statistics.fmean(s - f for s, f in pairs)
    split_us = statistics.fmean(s for s, _ in pairs)
    fused_us = statistics.fmean(f for _, f in pairs)
    rng = random.Random(2026)
    boot = []
    for _ in range(10000):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        boot.append(statistics.fmean(s - f for s, f in sample))
    boot.sort()
    lo, hi = boot[250], boot[9749]
    result = {
        "fused_code": args.fused_code, "split_code": args.split_code,
        "M": args.M, "K": args.K, "dtype": args.dtype,
        "blocks": args.blocks, "per_block": args.per_block,
        "warmup": args.warmup,
        "T_fused_us": fused_us, "T_split_us": split_us,
        "B_us": b_us, "relative_benefit": b_us / split_us,
        "B_ci95_us": [lo, hi],
        "B_ci95_relative": [lo / split_us, hi / split_us],
        "outputs_match": same,
        "pairs": pairs,
        "profile_us_per_call": profile_times,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = out / "json" / f"paired-M{args.M}-K{args.K}-{args.dtype}-b{args.branches}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: result[k] for k in
                      ("B_us", "B_ci95_us", "T_fused_us", "T_split_us",
                       "relative_benefit", "outputs_match")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
