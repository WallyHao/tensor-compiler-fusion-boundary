"""Synthetic paired sample for the analysis owner (C).

Everything in the produced run directory is fabricated by FakeBackend and is
marked ``"synthetic": true`` at row level and in run.json. It exists to pin the
table schema, pairing rules and verification queries before real data arrives.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .backend import FakeBackend, PreparedVariant
from .corpus import build_catalog
from .protocol import Protocol
from .rundir import RunContext
from .runner import Runner
from .util import (
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    linear_fit,
    median,
    sha256_json,
    utcnow,
    write_jsonl_atomic,
)
from .verify import print_verification, verify_run

SYNTH_CASES = ("P1-n65536-f32-contig-t", "R1-m256-k512-f32-contig-t")
SYNTH_COSTS = {
    ("P1-n65536-f32-contig-t", "split"): 3.42,
    ("P1-n65536-f32-contig-t", "fused"): 2.63,
    ("R1-m256-k512-f32-contig-t", "split"): 4.61,
    ("R1-m256-k512-f32-contig-t", "fused"): 4.08,
}


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, seed: int = 17):
        self.seed = seed
        self.cases = [c for c in build_catalog() if c["case_id"] in SYNTH_CASES]

    def catalog(self):
        cases = []
        for c in self.cases:
            row = dict(c)
            if c["case_id"] == "P1-n65536-f32-contig-t":
                row["source"] = {"default_maps_to": "distinct",
                                 "variant_status": {"split": "valid", "fused": "valid"},
                                 "candidate_run": "synthetic"}
            cases.append(row)
        return cases

    def prepare(self, case: dict, strategy: str) -> PreparedVariant:
        cid = case["case_id"]
        meta = {
            "provider": "synthetic",
            "strategy": strategy,
            "artifact_sha256": sha256_json({"case": cid, "strategy": strategy,
                                            "purpose": "synthetic-fixture"}),
            "artifacts": [f"synthetic/{cid}/{strategy}/output_code.py"],
            "artifact_dir": f"synthetic/{cid}/{strategy}",
            "n_kernels": 2 if strategy == "split" else 1,
            "preallocated": True,
            "compile_ms": 12.5,
            "submit_mode": "direct",
            "default_equivalent": "split" if strategy == "split" else "split",
            "default_partition": "same",
            "notes": "FABRICATED fixture; not a measurement",
            "kernel_list": [
                {"index": 0, "name": f"{strategy}_kernel_0", "grid": 256,
                 "block": 256, "registers": 24, "shared_memory_bytes": 0}
            ] + (
                [{"index": 1, "name": "split_kernel_1", "grid": 256, "block": 256,
                  "registers": 24, "shared_memory_bytes": 0}]
                if strategy == "split" else []
            ),
        }

        def check(input_id: str) -> dict:
            return {
                "reference": "eager",
                "tolerance": {"atol": 1e-6, "rtol": 1e-5},
                "output_shape": case["t_shape"],
                "output_dtype": case["t_dtype"],
                "nonfinite_count": 0,
                "max_abs_err": 0.0,
                "max_rel_err": 0.0,
                "fp64_reference": "not_applicable",
                "pass": True,
                "reason": None,
                "synthetic": True,
            }

        return PreparedVariant(
            case_id=cid, strategy=strategy, meta=meta, _call=lambda: None,
            _check=check,
        )

    def close(self):
        pass


def _synth_calibration(ctx: RunContext, protocol: Protocol) -> dict:
    rng = random.Random(7)
    rows: list[dict] = []
    direct = {"slope": 3.1, "intercept": 1.9}
    replay = {"slope": 1.5, "intercept": 1.05}
    for mode, model in (("direct", direct), ("replay", replay)):
        for k in protocol.launch_k:
            for block in range(protocol.calibration_blocks):
                segment_us = (model["slope"] * k + model["intercept"])
                segment_us *= 1.0 + rng.uniform(-0.01, 0.01)
                us = segment_us / k
                n = 500
                rows.append({
                    "run_id": ctx.run_id, "kind": "launch", "mode": mode,
                    "probe": "L", "size_or_k": k, "k": k, "block": block,
                    "repeats": n, "calls": k * n,
                    "elapsed_ms": segment_us * n / 1000.0, "us_per_call": us,
                    "segment_us": segment_us,
                    "derived": {"us_per_launch": us, "us_per_segment": segment_us,
                                "units": "us"},
                    "order": block % len(protocol.launch_k),
                    "launcher": "triton noop_write grid=(1,)" if mode == "direct" else "graph.replay()",
                    "stream": "default", "gpu_uuid": "GPU-SYNTHETIC-0000",
                    "calibration_version": "fb-calib-v1", "synthetic": True,
                    "ts_utc": utcnow(),
                })
    l2 = 72 * 1024 * 1024
    sizes = [1 << b for b in range(8, 30)]
    for s in sizes:
        for block in range(5):
            us = 3.5 + 2.0 * s / 1250.0
            us *= 1.0 + rng.uniform(-0.015, 0.015)
            rows.append({
                "run_id": ctx.run_id, "kind": "bandwidth", "mode": "direct",
                "probe": "BW", "size_or_k": s, "buffer_bytes": s, "block": block,
                "repeats": 20, "calls": 20, "elapsed_ms": us * 20 / 1000.0,
                "us_per_call": us,
                "derived": {"bytes_moved": 2 * s, "bytes_per_us": 2.0 * s / us,
                            "gbs": 2.0 * s / us * 1e-3,
                            "verified_equal": True, "units": "bytes/µs"},
                "order": block, "launcher": "triton copy_bytes BLOCK=1024",
                "stream": "default", "l2_bytes": l2,
                "calibration_version": "fb-calib-v1", "synthetic": True,
                "ts_utc": utcnow(),
            })
    ctx.append("calibration", rows)
    ctx.close_writers("calibration")

    fit = {"calibration_version": "fb-calib-v1", "synthetic": True,
           "captured_utc": utcnow(),
           "device": {"gpu_name": "SYNTHETIC (4090-shaped)", "l2_cache_bytes": l2,
                      "gpu_uuid": "GPU-SYNTHETIC-0000", "synthetic": True},
           "launch_count_check": {"direct_k8": {"observed_events": 8, "synthetic": True}},
           "capture_ms": {"1": 0.4, "8": 0.5, "32": 0.6, "128": 0.9},
           "notes": ["FABRICATED calibration; not measured."]}
    for mode in ("direct", "replay"):
        med = {}
        for k in protocol.launch_k:
            vals = [r["segment_us"] for r in rows
                    if r["kind"] == "launch" and r["mode"] == mode and r["k"] == k]
            med[k] = median(vals)
        f = linear_fit(sorted(med), [med[k] for k in sorted(med)])
        f.update(valid=True, slope_units="us per kernel",
                 intercept_units="us per timed segment",
                 point_dispersion={k: {"median_segment_us": med[k],
                                       "median_per_launch_us": med[k] / k} for k in med})
        fit[f"launch_{mode}"] = f
    size_med = {}
    for s in sizes:
        vals = [r["us_per_call"] for r in rows
                if r["kind"] == "bandwidth" and r["size_or_k"] == s]
        size_med[s] = median(vals)
    above = sorted([s for s in size_med if s > l2], reverse=True)
    main_sizes = above[:3]
    bw_main = median([median([2.0 * s / r["us_per_call"] for r in rows
                              if r["kind"] == "bandwidth" and r["size_or_k"] == s])
                      for s in main_sizes])
    fit["bandwidth"] = {
        "valid": True, "bw_main_bytes_per_us": bw_main,
        "bw_main_gbs": bw_main * 1e-3, "sizes_used_bytes": main_sizes,
        "rule": protocol.bw_main_rule, "l2_bytes": l2,
        "sizes_above_l2_bytes": above,
        "per_size": {
            str(s): {
                "median_bytes_per_us": median([2.0 * s / r["us_per_call"] for r in rows
                                               if r["kind"] == "bandwidth" and r["size_or_k"] == s]),
                "gbs": median([2.0 * s / r["us_per_call"] for r in rows
                               if r["kind"] == "bandwidth" and r["size_or_k"] == s]) * 1e-3,
                "repeats": 20, "verified_equal": True, "above_l2": s > l2,
            } for s in sizes
        },
        "spec_compliant_largest3_bytes": sorted(size_med, reverse=True)[:3],
        "spec_compliant_largest3_median_bytes_per_us": None,
    }
    atomic_write_json(ctx.run_dir / "calibration_fit.json", fit)
    return fit


def _synth_telemetry(ctx: RunContext, seconds: int = 120) -> None:
    rng = random.Random(11)
    start = 1760000000.0
    rows = []
    for i in range(0, seconds, 2):
        rows.append({
            "kind": "gpu", "ts_utc": utcnow(), "ts_unix": start + i,
            "source": "synthetic", "timestamp": f"synthetic+{i}s",
            "name": "SYNTHETIC (4090-shaped)", "uuid": "GPU-SYNTHETIC-0000",
            "temperature.gpu": str(46 + rng.randint(0, 2)),
            "power.draw": str(120.0 + rng.random()), "power.limit": "450.00",
            "clocks.sm": "2520", "clocks.mem": "10501",
            "clocks_throttle_reasons.active": "0x0000000000000000",
            "utilization.gpu": "97", "memory.used": "812",
            "synthetic": True,
        })
        rows.append({
            "kind": "apps", "ts_utc": utcnow(), "ts_unix": start + i,
            "source": "synthetic", "processes": [], "synthetic": True,
        })
    ctx.append("telemetry", rows)
    ctx.close_writers("telemetry")


def export_synthetic(out_dir: str | Path, force: bool = False) -> Path:
    import shutil

    out = Path(out_dir)
    if (out / "run.json").exists():
        if force:
            shutil.rmtree(out)
        else:
            out = out.parent / (out.name + "-" + utcnow().replace(":", "").replace("-", "")[:15])
    ensure_dir(out)
    proto = Protocol()
    code = {"commit": "synthetic", "dirty": False, "diff_sha256": None}
    ctx = RunContext.create(
        out.parent, out.name, proto, "synthetic",
        argv=["fb", "synth-export"], synthetic=True,
        corpus_hash=sha256_json(SyntheticProvider().catalog()),
        owner="measurement (fixture)",
        cloud={"platform": "synthetic", "instance_spec": "synthetic", "image_id": "synthetic"},
    )
    ctx.update(
        purpose="synthetic",
        code=code,
        code_hash=sha256_json(code),
        candidates={"provider": "synthetic", "formal": False,
                    "note": "FakeBackend fixture, NOT a measurement"},
    )
    provider = SyntheticProvider()
    backend = FakeBackend(SYNTH_COSTS, jitter_us=0.04,
                          replay_launch_saving_us=0.75, seed=7)
    backend.device_info = lambda: {  # type: ignore[method-assign]
        "backend": "fake", "device": "fake", "gpu_uuid": "GPU-SYNTHETIC-0000",
        "synthetic": True,
    }
    runner = Runner(ctx, provider, backend, proto, telemetry=None,
                    formal=False, allow_unverified=True)
    summary = runner.run([c["case_id"] for c in provider.catalog()])
    for case in provider.catalog():
        for strategy in ("split", "fused"):
            d = out / "synthetic" / case["case_id"] / strategy
            d.mkdir(parents=True, exist_ok=True)
            (d / "output_code.py").write_text(
                "# FABRICATED synthetic artifact for schema fixtures\n", encoding="utf-8"
            )
    ctx.update(candidates={"provider": "synthetic", "formal": False,
                           "note": "FakeBackend fixture, NOT a measurement"})
    _synth_calibration(ctx, proto)
    _synth_telemetry(ctx)
    ctx.update(calibration={"path": "calibration_fit.json", "synthetic": True})
    ctx.finish("done")
    atomic_write_text(out / "README.md", _readme())
    results = verify_run(out)
    print_verification(results)
    from .cli import seal_run

    seal_run(out)
    return out


def _readme() -> str:
    return """# SYNTHETIC paired sample (NOT MEASUREMENTS)

Every row in this run is fabricated by `fb.backend.FakeBackend`, and every row
carries `"synthetic": true`. run.json has `"purpose": "synthetic"`.

Purpose: give the analysis owner (C) a minimal, schema-complete paired sample
to build table readers and verification queries before real 4090 data exists.

Contents:
- `cases.jsonl` - 2 cases (P1/N=65536/f32, R1/M=256/K=512/f32)
- `timings.jsonl` - 240 paired rows (2 cases x 2 modes x 30 blocks x 2 strategies)
  plus 60 unpaired `default` rows for the one case whose default is a distinct
  partition (P1), marked `paired: false`
- `variants.jsonl`, `checks.jsonl` - one row per case/strategy and per input seed
- `calibration.jsonl`, `calibration_fit.json` - fabricated launch/bandwidth rows
- `telemetry.jsonl` - fabricated 1 Hz samples
- `case_order.jsonl`, `state/`, `blocks/` - ordering and atomic block commits
- `verification.json` / `verification.md` - independent checks

Do not use any number here in the report. Real data uses a separate run_id and
`synthetic: false`.

Minimal pairing query (first complete attempt per block, both strategies, same n):

```sql
-- conceptual, one row per block attempt
SELECT case_id, mode, block, attempt, n,
       max(CASE WHEN strategy='split' THEN us_per_call END) AS split_us,
       max(CASE WHEN strategy='fused' THEN us_per_call END) AS fused_us
FROM timings
GROUP BY case_id, mode, block, attempt, n
HAVING count(DISTINCT strategy) = 2
   AND min(quality) = 'ok'
   AND count(*) = 2;
```
"""


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/synthetic-c-sample")
    ns = ap.parse_args()
    print(export_synthetic(ns.out))
