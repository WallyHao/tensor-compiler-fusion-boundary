"""Device calibration: launch cost (direct/replay) and effective copy bandwidth."""
from __future__ import annotations

import csv
import io
import math
import random
import time
from pathlib import Path
from typing import Any

from .backend import CudaBackend
from .cuda_kernels import copy_bytes, noop_write
from .protocol import Protocol
from .rundir import RunContext
from .util import linear_fit, median, utcnow

CALIBRATION_VERSION = "fb-calib-v1"


def _block_order(items: list[Any], seed: int, tag: str) -> list[Any]:
    rng = random.Random(f"{seed}:{tag}")
    out = list(items)
    rng.shuffle(out)
    return out


def _time_calls(backend: CudaBackend, fn, calls: int) -> float:
    torch = backend.torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    backend.sync()
    start.record(backend.stream)
    for _ in range(calls):
        fn()
    end.record(backend.stream)
    end.synchronize()
    return float(start.elapsed_time(end))


def _count_kernels(backend: CudaBackend, fn, calls: int) -> dict:
    torch = backend.torch
    try:
        backend.sync()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(calls):
                fn()
            backend.sync()
        names: dict[str, int] = {}
        for evt in prof.events():
            if str(getattr(evt, "device_type", "")) == "DeviceType.CUDA" or "cuda" in str(
                getattr(evt, "device_type", "")
            ).lower():
                if evt.name and not evt.name.startswith("cuda"):
                    names[evt.name] = names.get(evt.name, 0) + 1
        return {"observed_events": sum(names.values()), "kernels": names, "calls": calls}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "calls": calls}


def run_calibration(
    ctx: RunContext,
    backend: CudaBackend,
    protocol: Protocol,
    *,
    synthetic: bool = False,
) -> dict:
    torch = backend.torch
    if noop_write is None or copy_bytes is None:
        raise RuntimeError("triton is unavailable; calibration cannot run")
    rows: list[dict] = []
    device = backend.device_info()
    l2_bytes = device.get("l2_cache_bytes")

    def emit(row: dict) -> None:
        row.update(
            run_id=ctx.run_id,
            calibration_version=CALIBRATION_VERSION,
            gpu_uuid=backend.gpu_uuid,
            ts_utc=utcnow(),
            synthetic=synthetic,
        )
        rows.append(row)
        ctx.append("calibration", row)

    # ---------------- L, direct ----------------
    out = torch.zeros(1, dtype=torch.int32, device=backend.device)
    grid = (1,)
    launch = lambda: noop_write[grid](out)
    backend.warmup_variant = None
    for _ in range(20):
        launch()
    backend.sync()
    launch_check = _count_kernels(backend, launch, 8)

    direct_points: dict[int, list[float]] = {k: [] for k in protocol.launch_k}
    order = []
    for block in range(protocol.calibration_blocks):
        shuffled = _block_order(list(protocol.launch_k), protocol.seed, "L-direct")
        for pos, k in enumerate(shuffled):
            repeats = max(
                1,
                math.ceil(
                    protocol.calibration_target_ms
                    * 1000.0
                    / (k * protocol.calibration_assumed_launch_us)
                ),
            )
            for _ in range(k):
                launch()  # warm each k once outside events (same launcher)
            backend.sync()

            def segment(_k=k):
                for _ in range(_k):
                    launch()

            elapsed_ms = _time_calls(backend, segment, repeats)
            per_launch = elapsed_ms * 1000.0 / (k * repeats)
            per_segment = elapsed_ms * 1000.0 / repeats
            direct_points[k].append(per_segment)
            emit(
                {
                    "kind": "launch",
                    "mode": "direct",
                    "probe": "L",
                    "size_or_k": k,
                    "k": k,
                    "block": block,
                    "repeats": repeats,
                    "calls": k * repeats,
                    "elapsed_ms": elapsed_ms,
                    "us_per_call": per_launch,
                    "segment_us": per_segment,
                    "derived": {"us_per_launch": per_launch,
                                "us_per_segment": per_segment, "units": "us"},
                    "order": pos,
                    "launcher": "triton noop_write grid=(1,)",
                    "stream": "default",
                }
            )

    # ---------------- L, replay ----------------
    replay_points: dict[int, list[float]] = {k: [] for k in protocol.launch_k}
    capture_info: dict[int, float] = {}
    replay_checks: dict[int, dict] = {}
    graphs: dict[int, Any] = {}
    for k in protocol.launch_k:
        side = torch.cuda.Stream(device=backend.device)
        side.wait_stream(backend.stream)
        with torch.cuda.stream(side):
            for _ in range(k):
                launch()
        backend.stream.wait_stream(side)
        backend.sync()
        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(graph):
            for _ in range(k):
                launch()
        backend.sync()
        capture_info[k] = (time.perf_counter() - t0) * 1000.0
        graphs[k] = graph
        replay_checks[k] = _count_kernels(backend, graph.replay, 4)
    for block in range(protocol.calibration_blocks):
        shuffled = _block_order(list(protocol.launch_k), protocol.seed, "L-replay")
        for pos, k in enumerate(shuffled):
            repeats = max(
                1,
                math.ceil(
                    protocol.calibration_target_ms
                    * 1000.0
                    / (k * protocol.calibration_assumed_launch_us)
                ),
            )
            elapsed_ms = _time_calls(backend, graphs[k].replay, repeats)
            per_launch = elapsed_ms * 1000.0 / (k * repeats)
            per_segment = elapsed_ms * 1000.0 / repeats
            replay_points[k].append(per_segment)
            emit(
                {
                    "kind": "launch",
                    "mode": "replay",
                    "probe": "L",
                    "size_or_k": k,
                    "k": k,
                    "block": block,
                    "repeats": repeats,
                    "calls": k * repeats,
                    "elapsed_ms": elapsed_ms,
                    "us_per_call": per_launch,
                    "segment_us": per_segment,
                    "derived": {
                        "us_per_launch": per_launch,
                        "us_per_segment": per_segment,
                        "capture_ms": capture_info.get(k),
                        "capture_path": "manual torch.cuda.CUDAGraph capture of k noop launches",
                    },
                    "order": pos,
                    "launcher": "graph.replay()",
                    "stream": "default",
                }
            )

    # ---------------- BW ----------------
    bw_points: dict[int, list[float]] = {s: [] for s in protocol.bw_sizes_bytes}
    bw_repeats: dict[int, int] = {}
    bw_verified: dict[int, bool] = {}
    bw_order = 0
    for s in protocol.bw_sizes_bytes:
        src = torch.randint(0, 256, (s,), dtype=torch.uint8, device=backend.device)
        dst = torch.zeros_like(src)
        grid_s = (max(1, (s + 1023) // 1024),)
        fn = lambda: copy_bytes[grid_s](src, dst, s, BLOCK=1024)
        for _ in range(5):
            fn()
        backend.sync()
        dst.zero_()
        fn()
        backend.sync()
        bw_verified[s] = bool(torch.equal(src, dst))
        est_us = max(1e-3, _time_calls(backend, fn, 8) * 1000.0 / 8.0)
        bw_repeats[s] = max(
            1, min(8192, math.ceil(protocol.bw_target_ms * 1000.0 / est_us))
        )
        for block in range(protocol.bw_blocks):
            elapsed_ms = _time_calls(backend, fn, bw_repeats[s])
            us_per_call = elapsed_ms * 1000.0 / bw_repeats[s]
            bytes_per_us = 2.0 * s / us_per_call
            bw_points[s].append(bytes_per_us)
            emit(
                {
                    "kind": "bandwidth",
                    "mode": "direct",
                    "probe": "BW",
                    "size_or_k": s,
                    "buffer_bytes": s,
                    "block": block,
                    "repeats": bw_repeats[s],
                    "calls": bw_repeats[s],
                    "elapsed_ms": elapsed_ms,
                    "us_per_call": us_per_call,
                    "derived": {
                        "bytes_moved": 2 * s,
                        "bytes_per_us": bytes_per_us,
                        "gbs": bytes_per_us * 1e-3,
                        "units": "bytes/µs; GB/s = bytes_per_us * 1e-3",
                        "verified_equal": bw_verified[s],
                    },
                    "order": bw_order,
                    "launcher": "triton copy_bytes BLOCK=1024",
                    "stream": "default",
                    "l2_bytes": l2_bytes,
                }
            )
            bw_order += 1
        del src, dst
        torch.cuda.empty_cache()

    # ---------------- fits ----------------
    def fit(points: dict[int, list[float]], tag: str) -> dict:
        # points hold per-segment time for k launches, so the least-squares
        # slope is the incremental us per kernel and the intercept is the
        # fixed cost of one timed segment. This is the quantity x1 needs.
        med = {k: median(v) for k, v in points.items() if v}
        xs = sorted(med)
        ys = [med[k] for k in xs]
        try:
            f = linear_fit(xs, ys)
        except Exception as exc:
            return {"valid": False, "reason": f"{type(exc).__name__}: {exc}", "points": med}
        f.update(
            valid=bool(f["slope"] > 0),
            slope_units="us per kernel",
            intercept_units="us per timed segment (k launches)",
            point_dispersion={
                k: {"median_segment_us": med[k],
                    "median_per_launch_us": med[k] / k,
                    "min_segment_us": min(points[k]),
                    "max_segment_us": max(points[k])}
                for k in xs
            },
            tag=tag,
        )
        return f

    fit_direct = fit(direct_points, "L-direct")
    fit_replay = fit(replay_points, "L-replay")

    size_med = {s: median(v) for s, v in bw_points.items() if v}
    sizes_sorted = sorted(size_med, reverse=True)
    if l2_bytes:
        above = [s for s in sizes_sorted if s > l2_bytes]
    else:
        above = []
    main_sizes = above[:3] if l2_bytes else sizes_sorted[:3]
    bw_main = median([size_med[s] for s in main_sizes]) if len(main_sizes) == 3 else None
    spec_largest3 = sizes_sorted[:3]
    bw_fit = {
        "valid": bw_main is not None and bw_main > 0,
        "bw_main_bytes_per_us": bw_main,
        "bw_main_gbs": None if bw_main is None else bw_main * 1e-3,
        "sizes_used_bytes": main_sizes,
        "rule": protocol.bw_main_rule,
        "l2_bytes": l2_bytes,
        "sizes_above_l2_bytes": above,
        "per_size": {
            str(s): {
                "median_bytes_per_us": size_med[s],
                "gbs": size_med[s] * 1e-3,
                "min": min(bw_points[s]),
                "max": max(bw_points[s]),
                "repeats": bw_repeats[s],
                "verified_equal": bw_verified[s],
                "above_l2": bool(l2_bytes and s > l2_bytes),
            }
            for s in sizes_sorted
        },
        "spec_compliant_largest3_bytes": spec_largest3,
        "spec_compliant_largest3_median_bytes_per_us": median([size_med[s] for s in spec_largest3]),
    }
    result = {
        "calibration_version": CALIBRATION_VERSION,
        "captured_utc": utcnow(),
        "device": device,
        "launch_direct": fit_direct,
        "launch_replay": fit_replay,
        "bandwidth": bw_fit,
        "launch_count_check": {"direct_k8": launch_check, "replay": replay_checks},
        "capture_ms": {str(k): v for k, v in capture_info.items()},
        "notes": [
            "L is the effective incremental cost on this submit path, not the theoretical launch latency.",
            "BW rows above L2 are used for the main constant; smaller sizes are diagnostics.",
            "Both fits can be recomputed from calibration.jsonl alone.",
        ],
    }
    if not fit_direct.get("valid"):
        result["model_status"] = "launch_direct_invalid: check event boundaries, units, device state"
    ctx.update(calibration={"path": "calibration_fit.json", "summary": {
        "launch_direct_us_per_kernel": fit_direct.get("slope"),
        "launch_replay_us_per_kernel": fit_replay.get("slope"),
        "bw_main_gbs": bw_fit["bw_main_gbs"],
        "sizes_used_bytes": main_sizes,
    }})
    from .util import atomic_write_json

    atomic_write_json(ctx.run_dir / "calibration_fit.json", result)
    return result
