"""Separate-process profiling: per-kernel timing trace (torch.profiler).

Nsight Compute is optional; exact commands are emitted for a follow-up run.
Profiles never share a process or GPU window with formal timing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .backend import CudaBackend
from .provider import load_provider
from .rundir import RunContext
from .runner import normalize_case_filter
from .util import atomic_write_json, atomic_write_text, ensure_dir, read_json, utcnow

PROFILE_VERSION = "fb-profile-torch-v1"


def _events_to_rows(prof, case_id: str, strategy: str, iters: int,
                    trace_rel: str | None, run_id: str, synthetic: bool) -> list[dict]:
    rows: list[dict] = []
    counts: dict[str, dict] = {}
    order: list[str] = []
    for evt in prof.events():
        device_type = str(getattr(evt, "device_type", ""))
        if "CUDA" not in device_type:
            continue
        name = getattr(evt, "name", None)
        if not name:
            continue
        total = getattr(evt, "self_device_time_total", None)
        if total is None:
            total = getattr(evt, "self_cuda_time_total", None)
        if total is None:
            continue
        if name not in counts:
            counts[name] = {"count": 0, "total_us": 0.0}
            order.append(name)
        counts[name]["count"] += 1
        counts[name]["total_us"] += float(total)
    for idx, name in enumerate(order):
        c = counts[name]
        rows.append({
            "run_id": run_id,
            "case_id": case_id,
            "strategy": strategy,
            "kernel_index": idx,
            "kernel_name": name,
            "metric": "kernel_time_us_total",
            "sample": None,
            "raw_metric_name": "self_device_time_total",
            "source": "torch.profiler",
            "value": c["total_us"],
            "unit": "us",
            "scope": f"{iters} timed calls after warmup; per-call={iters}",
            "trace_path": trace_rel,
            "tool_version": "torch.profiler",
            "synthetic": synthetic,
            "ts_utc": utcnow(),
        })
        rows.append({
            "run_id": run_id,
            "case_id": case_id,
            "strategy": strategy,
            "kernel_index": idx,
            "kernel_name": name,
            "metric": "kernel_count",
            "sample": None,
            "raw_metric_name": "count",
            "source": "torch.profiler",
            "value": c["count"],
            "unit": "count",
            "scope": f"{iters} timed calls after warmup",
            "trace_path": trace_rel,
            "tool_version": "torch.profiler",
            "synthetic": synthetic,
            "ts_utc": utcnow(),
        })
    rows.append({
        "run_id": run_id,
        "case_id": case_id,
        "strategy": strategy,
        "kernel_index": None,
        "kernel_name": None,
        "metric": "distinct_kernels",
        "sample": None,
        "raw_metric_name": "len(key_averages)",
        "source": "torch.profiler",
        "value": len(order),
        "unit": "count",
        "scope": f"{iters} timed calls after warmup",
        "trace_path": trace_rel,
        "tool_version": "torch.profiler",
        "synthetic": synthetic,
        "ts_utc": utcnow(),
    })
    for metric in ("dram_read_bytes", "dram_write_bytes", "l2_bytes", "achieved_occupancy",
                   "registers_per_thread", "shared_memory_bytes", "grid", "block"):
        rows.append({
            "run_id": run_id,
            "case_id": case_id,
            "strategy": strategy,
            "kernel_index": None,
            "kernel_name": None,
            "metric": metric,
            "sample": None,
            "raw_metric_name": None,
            "source": "unavailable",
            "value": None,
            "unit": None,
            "scope": None,
            "trace_path": trace_rel,
            "reason": "torch.profiler does not expose hardware counters; run the emitted ncu command",
            "tool_version": None,
            "synthetic": synthetic,
            "ts_utc": utcnow(),
        })
    return rows


def run_profile(run_dir: str | Path, cases_spec: str, iters: int = 10,
                device: str = "cuda:0") -> dict:
    run_dir = Path(run_dir)
    ctx = RunContext.open_existing(run_dir)
    run = read_json(run_dir / "run.json")
    spec = run.get("candidates", {}).get("provider")
    if not spec:
        raise RuntimeError("run.json has no candidates.provider; cannot profile")
    provider = load_provider(spec, run.get("candidates", {}).get("options") or {})
    backend = CudaBackend(device)
    torch = backend.torch
    recorded_uuid = (
        run.get("gpu_uuid")
        or ((run.get("environment") or {}).get("gpu") or {}).get("uuid")
        or next(iter((run.get("device_check") or {}).get("observed_uuids") or []), None)
    )
    if recorded_uuid and backend.gpu_uuid and recorded_uuid != backend.gpu_uuid:
        raise RuntimeError(
            f"GPU UUID changed: run={recorded_uuid} now={backend.gpu_uuid}; start a new run"
        )
    catalog = provider.catalog()
    ids = normalize_case_filter(cases_spec, catalog)
    by_id = {c["case_id"]: c for c in catalog}
    eligible = [cid for cid in ids if by_id[cid].get("planned_status") == "planned"]
    if not eligible:
        eligible = [c["case_id"] for c in catalog
                    if c.get("planned_status") == "planned"]
    if not eligible:
        raise RuntimeError("no planned (pairable) case to profile in this run")
    ids = eligible
    outdir = ensure_dir(run_dir / "artifacts" / "profiles")
    rows: list[dict] = []
    ncu_lines = ["#!/usr/bin/env bash", "# emitted by fb profile; requires ncu on PATH"]
    for cid in ids:
        case = by_id[cid]
        for strategy in ("split", "fused"):
            v = provider.prepare(case, strategy)
            backend.warmup(v, 5)
            backend.sync()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                for _ in range(iters):
                    v.call()
                backend.sync()
            trace = outdir / f"{cid}__{strategy}.json"
            try:
                prof.export_chrome_trace(str(trace))
                trace_rel = str(trace.relative_to(run_dir))
            except Exception:
                trace_rel = None
            rows.extend(
                _events_to_rows(prof, cid, strategy, iters, trace_rel,
                                ctx.run_id, bool(run.get("synthetic")))
            )
            meta_rows = []
            for k in v.meta.get("kernel_list") or []:
                for metric, value in k.items():
                    meta_rows.append({
                        "run_id": ctx.run_id,
                        "case_id": cid,
                        "strategy": strategy,
                        "kernel_index": k.get("index"),
                        "kernel_name": k.get("name"),
                        "metric": metric,
                        "sample": None,
                        "raw_metric_name": metric,
                        "source": "compile_metadata",
                        "value": value,
                        "unit": None,
                        "scope": "compiled artifact",
                        "trace_path": trace_rel,
                        "synthetic": bool(run.get("synthetic")),
                        "ts_utc": utcnow(),
                    })
            rows.extend(meta_rows)
            ncu_lines.append(
                "ncu --target-processes all --set full --kernel-name-base demangled "
                f"-o {outdir}/{cid}__{strategy}_ncu "
                f"python -m fb.ncu_target --provider {spec!r} --case {cid} "
                f"--strategy {strategy} --iters {iters}"
            )
            v.close()
    ctx.append("profiles", rows)
    ctx.close_writers("profiles")
    atomic_write_text(outdir / "ncu-commands.sh", "\n".join(ncu_lines) + "\n")
    atomic_write_json(outdir / "profile_run.json", {
        "profile_version": PROFILE_VERSION,
        "run_id": ctx.run_id,
        "cases": ids,
        "iters": iters,
        "device": device,
        "gpu_uuid": backend.gpu_uuid,
        "gpu_uuid_matches_run": (
            None if not recorded_uuid or not backend.gpu_uuid
            else recorded_uuid == backend.gpu_uuid
        ),
        "utc": utcnow(),
        "note": "run in a separate process, after all paired timing has finished",
    })
    return {"cases": ids, "rows": len(rows), "dir": str(outdir)}
