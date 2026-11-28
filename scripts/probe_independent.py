#!/usr/bin/env python3
"""Compile-only gate for an independently scheduled split candidate.

The probe answers one question: when the producer half is compiled as its own
graph, does TorchInductor choose a different reduction tile than the producer
kernel inside the ``fused`` variant?  It is not a strategy and writes no
measurement rows.  A role runs in its own process with its own
``TORCHINDUCTOR_CACHE_DIR`` and trace directory, mirroring
``strategies.compile_variant`` (same pinned Inductor knobs, same evidence
extraction).

Roles
-----
``producer``      ``t = producer(x)`` compiled alone.
``consumer``      ``y = consumer(t)`` compiled alone, ``t`` from eager producer.
``fused``         the single-graph template entry (reference time and config).
``independent``   producer and consumer compiled as two graphs, then called in
                  sequence inside the timed region: the candidate upper bound
                  (it pays two Python entries).
``paired``        ``independent`` against ``fused`` in one process, 30 blocks
                  with balanced SF/FS order and a block bootstrap: the gate's
                  timing evidence.

Gate (see README ``审阅发现``): the producer kernel of ``independent`` must
differ from the ``fused`` producer kernel in ``block``/``num_warps``; otherwise
the candidate is byte-for-byte the existing same-graph split and must not be
measured or implemented.

The script derives ``cache/<case>/<role>`` and ``trace/<case>/<tag>`` from
``--out``; never share a trace directory across cases (``os.walk`` order is not
creation order and the collector would pick another case's ``output_code.py``).

Example
-------
    PYTHON=/path/to/venv/bin/python bash scripts/run_probe.sh      # 3 cases
    PYTHON=/path/to/venv/bin/python JOBS=6 bash scripts/run_probe_all.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="corpus case_id")
    parser.add_argument("--role", required=True,
                        choices=("producer", "consumer", "fused", "independent", "paired"))
    parser.add_argument("--out", required=True, help="result directory")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--per-block", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Per-case directories: a shared trace dir lets collect_trace_artifacts pick
    # another case's output_code.py (os.walk order is not creation order).
    cache_dir = out / "cache" / args.case / args.role
    cache_dir.mkdir(parents=True, exist_ok=True)
    # The cache directory must be fixed before torch initialises Inductor.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    import torch

    from fusionboundary import corpus, forensics, templates
    from fusionboundary import strategies as S
    from fusionboundary.validate import iter_inputs

    case = corpus.get_case(args.case)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("probe needs a CUDA device")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_info = {
        "name": props.name,
        "uuid": str(getattr(props, "uuid", "")),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "python": sys.version.split()[0],
    }

    _, x = next(iter(iter_inputs(case, device)))
    producer_fn, consumer_fn = templates.PRODUCER_CONSUMER[case.template]
    fused_fn = templates.TEMPLATE_FNS[case.template]
    reference = templates.reference(case, x)

    def _compile(fn, tag: str):
        trace_dir = out / "trace" / args.case / tag
        trace_dir.mkdir(parents=True, exist_ok=True)
        with S.inductor_environment(str(trace_dir)):
            compiled = torch.compile(fn, backend="inductor", dynamic=False, fullgraph=True)
            with torch.no_grad():
                compiled(x) if fn is not consumer_fn else compiled(_eager_t(x))
            torch.cuda.synchronize()
        return compiled, trace_dir

    def _eager_t(arg):
        with torch.no_grad():
            return producer_fn(arg)

    def _collect(tag: str, trace_dir: Path) -> dict:
        trace = forensics.collect_trace_artifacts(str(trace_dir))
        code = trace.get("output_code")
        if not code:
            raise RuntimeError(f"no output_code.py in trace dir {trace_dir}")
        frozen = out / f"{args.case}__{tag}_output_code.py"
        shutil.copyfile(code, frozen)
        parsed = forensics.parse_output_code(str(frozen))
        names = [k.name for k in parsed.kernels]
        resources = S.extract_kernel_resources(names)
        return {
            "tag": tag,
            "trace_dir": str(trace_dir),
            "output_code": str(frozen),
            "kernels": [{"name": k.name, "original_aten": k.original_aten,
                         "grid": k.grid} for k in parsed.kernels],
            "kernel_resources": resources,
            "parse_notes": parsed.parse_notes,
        }

    def _bench(step) -> dict:
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        per_call = []
        for _ in range(args.blocks):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.per_block):
                step()
            end.record()
            torch.cuda.synchronize()
            per_call.append(start.elapsed_time(end) * 1e3 / args.per_block)
        return {
            "us_mean": statistics.fmean(per_call),
            "us_median": statistics.median(per_call),
            "us_min": min(per_call),
            "us_max": max(per_call),
            "blocks": args.blocks,
            "per_block": args.per_block,
            "warmup": args.warmup,
        }

    def _numerics(got, want) -> dict:
        err = (got.float() - want.float()).abs().max().item()
        scale = want.float().abs().max().item()
        ok = bool(torch.allclose(got.float(), want.float(),
                                 atol=case.atol, rtol=case.rtol))
        return {"max_abs_err": err, "max_abs_ref": scale,
                "atol": case.atol, "rtol": case.rtol, "ok": ok}

    result: dict = {
        "case_id": case.case_id,
        "template": case.template,
        "shape": list(case.shape),
        "dtype": case.dtype,
        "layout": case.layout,
        "role": args.role,
        "device": device_info,
        "cache_dir": str(cache_dir),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "graphs": [],
        "notes": [],
    }

    try:
        if args.role == "producer":
            compiled, trace = _compile(producer_fn, "producer")
            got = compiled(x)
            torch.cuda.synchronize()
            result["graphs"].append(_collect("producer", trace))
            result["numerics"] = _numerics(got, _eager_t(x))
            result["timing"] = _bench(lambda: compiled(x))
        elif args.role == "consumer":
            t_in = _eager_t(x)
            compiled, trace = _compile(consumer_fn, "consumer")
            got = compiled(t_in)
            torch.cuda.synchronize()
            result["graphs"].append(_collect("consumer", trace))
            with torch.no_grad():
                want = consumer_fn(t_in)
            result["numerics"] = _numerics(got, want)
            result["timing"] = _bench(lambda: compiled(t_in))
        elif args.role == "fused":
            compiled, trace = _compile(fused_fn, "fused")
            got = compiled(x)
            torch.cuda.synchronize()
            result["graphs"].append(_collect("fused", trace))
            result["numerics"] = _numerics(got, reference)
            result["timing"] = _bench(lambda: compiled(x))
        elif args.role == "independent":
            prod, prod_trace = _compile(producer_fn, "producer")
            cons, cons_trace = _compile(consumer_fn, "consumer")
            got = cons(prod(x))
            torch.cuda.synchronize()
            result["graphs"].append(_collect("producer", prod_trace))
            result["graphs"].append(_collect("consumer", cons_trace))
            result["numerics"] = _numerics(got, reference)

            def step():
                return cons(prod(x))

            result["timing"] = _bench(step)
            result["notes"].append(
                "independent pays two Python entries; timing is an upper bound "
                "on the eventual candidate's cost")
        else:  # paired: the candidate against fused, interleaved per block
            fused, fused_trace = _compile(fused_fn, "fused")
            prod, prod_trace = _compile(producer_fn, "producer")
            cons, cons_trace = _compile(consumer_fn, "consumer")

            def split_step():
                return cons(prod(x))

            def fused_step():
                return fused(x)

            got = split_step()
            torch.cuda.synchronize()
            result["graphs"].append(_collect("fused", fused_trace))
            result["graphs"].append(_collect("producer", prod_trace))
            result["graphs"].append(_collect("consumer", cons_trace))
            result["numerics"] = _numerics(got, reference)

            def block_time(fn) -> float:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.per_block):
                    fn()
                end.record()
                torch.cuda.synchronize()
                return start.elapsed_time(end) * 1e3 / args.per_block

            for _ in range(args.warmup):
                split_step()
                fused_step()
            torch.cuda.synchronize()

            pairs = []
            for block in range(args.blocks):
                # "SF" even, "FS" odd: balanced order, as in fb's pairing.
                if block % 2 == 0:
                    s = block_time(split_step)
                    f = block_time(fused_step)
                else:
                    f = block_time(fused_step)
                    s = block_time(split_step)
                pairs.append((s, f))

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
            rel = b_us / split_us if split_us else float("nan")
            result["timing"] = {
                "blocks": args.blocks,
                "per_block": args.per_block,
                "warmup": args.warmup,
                "T_split_independent_us": split_us,
                "T_fused_us": fused_us,
                "B_us": b_us,
                "relative_benefit": rel,
                "B_ci95_us": [lo, hi],
                "B_ci95_relative": [lo / split_us, hi / split_us] if split_us else None,
                "pairs": pairs,
            }
            result["notes"].append(
                "split side is producer+consumer compiled as two graphs and called "
                "through two Python entries; it is an upper bound on the eventual "
                "candidate's cost")
    except Exception as exc:  # keep the failure in the archive, do not guess
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = out / "json" / f"{case.case_id}__{args.role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
