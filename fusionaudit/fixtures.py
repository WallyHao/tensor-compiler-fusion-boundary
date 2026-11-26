"""Deterministic synthetic fixture run.

The fixture is deliberately small (4 blocks per case) and is marked as
synthetic at run level and in every report.  It exercises four failure/edge
conditions required before touching real 4090 data:

1. a block with only one half of the pair,
2. a failed pair that is retested as a whole on attempt 2,
3. a benefit interval inside the +/-2% practical band,
4. a supplied fold assignment that leaks one template across train/test.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl, write_text

RUN_ID = "synthetic-fixture-20261125"
BLOCKS = 4
BLOCK_DELTAS = [0.0, 0.02, -0.015, 0.01]
FUSED_DELTAS = [0.0, 0.015, -0.01, 0.005]
NEAR_ZERO_SPLIT = [100.000, 100.005, 99.997, 100.002]
NEAR_ZERO_FUSED = [99.895, 99.899, 99.893, 99.898]


def write_fixture_run(root: Path, *, force: bool = False) -> Path:
    run_dir = root / RUN_ID
    if run_dir.exists():
        if not force:
            return run_dir
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "traces").mkdir()

    _write_environment(run_dir)
    cases = _cases()
    write_jsonl(run_dir / "cases.jsonl", [{k: v for k, v in case.items() if not k.startswith("_")} for case in cases])
    variants = _write_artifacts_and_variants(run_dir, cases)
    write_jsonl(run_dir / "variants.jsonl", variants)
    write_jsonl(run_dir / "checks.jsonl", _checks(cases))
    write_jsonl(run_dir / "timings.jsonl", _timings(cases, variants))
    write_jsonl(run_dir / "calibration.jsonl", _calibration())
    write_jsonl(run_dir / "telemetry.jsonl", _telemetry(cases))
    write_jsonl(run_dir / "results.jsonl", _leaking_input_folds(cases))
    write_json(
        run_dir / "run.json",
        {
            "run_id": RUN_ID,
            "purpose": "SYNTHETIC FIXTURE: validate audit, attempt selection, pairing, classification and leak checks; not experimental data",
            "synthetic": True,
            "parent_run": None,
            "protocol_hash": "sha256:" + _digest_text("synthetic-protocol-v1"),
            "corpus_hash": "sha256:" + _digest_text("synthetic-corpus-v1"),
            "code_hash": "sha256:" + _digest_text("fusionaudit-fixture-generator"),
            "command": "python -m fusionaudit fixtures --out runs --force",
            "cloud": "local CPU",
            "instance": "none (fixture)",
            "image": "none (fixture)",
            "persistent_volume": str(root),
            "environment_lock": "environment.lock",
            "gpu_snapshot": "gpu.json",
            "started_at": "2026-11-25T00:00:00Z",
            "ended_at": "2026-11-25T00:10:00Z",
            "seed": 2026,
            "owner": "analysis",
            "status": "synthetic",
            "expected_blocks": BLOCKS,
            "expected_case_count": len(cases),
            "planned_template_counts": {"P1": 3, "P2": 2, "R1": 2, "R2": 2},
            "calibration": {
                "launch_us_per_kernel": {"direct": 4.0, "replay": 0.8},
                "bandwidth_bytes_per_us": {"direct": 500.0, "replay": 500.0},
                "l2_bytes": 100663296,
                "environment_ref": "gpu.json",
            },
        },
    )
    return run_dir


def _cases() -> list[dict[str, Any]]:
    base = [
        ("P1-n8192-f32-contig-t", "P1", {"N": 8192}, "float32", 65536, 40.0, 39.70, "split"),
        ("P1-n65536-f32-contig-t", "P1", {"N": 65536}, "float32", 524288, 80.0, 70.0, "fused"),
        ("P1-n65536-f16-contig-t", "P1", {"N": 65536}, "float16", 262144, 60.0, 57.0, "split"),
        ("P2-n16384-f32-contig-t", "P2", {"N": 16384}, "float32", 131072, 55.0, 45.0, "fused"),
        ("P2-n131072-f16-contig-t", "P2", {"N": 131072}, "float16", 524288, 100.0, 105.0, "other"),
        ("R1-m64-k512-f32-contig-t", "R1", {"M": 64, "K": 512}, "float32", 512, 100.0, 99.895, "split"),
        ("R1-m256-k2048-f32-contig-t", "R1", {"M": 256, "K": 2048}, "float32", 2048, 800.0, 760.0, "fused"),
        ("R2-m64-k512-f32-contig-t", "R2", {"M": 64, "K": 512}, "float32", 512, 90.0, 91.0, "split"),
        ("R2-m256-k1024-f16-contig-t", "R2", {"M": 256, "K": 1024}, "float16", 1024, 120.0, 115.0, "fused"),
    ]
    rows = []
    for case_id, template, shape, dtype, saved_bytes, split_base, fused_base, default in base:
        reduction = template.startswith("R")
        rows.append(
            {
                "case_id": case_id,
                "template": template,
                "shape": shape,
                "dtype": dtype,
                "layout": "contiguous",
                "edge": "t",
                "input_seeds": [17, 29, 43, 0],
                "t_shape": [shape.get("M")] if reduction else [shape.get("N")],
                "t_dtype": dtype,
                "logical_saved_bytes": saved_bytes,
                "planned_status": "valid",
                "exclusion_reason": None,
                "synthetic": True,
                "_split_base": split_base,
                "_fused_base": fused_base,
                "_default": default,
            }
        )
    return rows


def _write_artifacts_and_variants(run_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for index, case in enumerate(cases):
        case_id = case["case_id"]
        for strategy in ("split", "fused"):
            artifact = run_dir / "artifacts" / case_id / f"{strategy}.py"
            trace = run_dir / "traces" / case_id / f"{strategy}.trace.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            trace.parent.mkdir(parents=True, exist_ok=True)
            source = (
                f"# SYNTHETIC fixture artifact\n"
                f"# case_id={case_id}\n"
                f"# strategy={strategy}\n"
                f"# edge=t\n"
                f"def triton_kernel_{index}_{strategy}():\n"
                f"    pass\n"
            )
            write_text(artifact, source)
            write_json(
                trace,
                {
                    "synthetic": True,
                    "case_id": case_id,
                    "strategy": strategy,
                    "verdict": "edge-materialized" if strategy == "split" else "edge-elided",
                    "launches": 2 if strategy == "split" else 1,
                },
            )
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            fused_blocks = 8 if index % 2 == 0 else 4
            kernels = (
                [
                    {"index": 0, "name": f"producer_{index}", "region": True, "blocks_per_sm": 8, "warps_per_block": 4, "sm_max_warps": 64},
                    {"index": 1, "name": f"consumer_{index}", "region": True, "blocks_per_sm": 8, "warps_per_block": 4, "sm_max_warps": 64},
                ]
                if strategy == "split"
                else [
                    {"index": 0, "name": f"fused_{index}", "region": True, "blocks_per_sm": fused_blocks, "warps_per_block": 4, "sm_max_warps": 64},
                ]
            )
            variants.append(
                {
                    "case_id": case_id,
                    "strategy": strategy,
                    "compile_config": "inductor-synthetic-fixed",
                    "artifact_hash": "sha256:" + digest,
                    "artifact_path": str(artifact.relative_to(run_dir)),
                    "edge_mapping": "t",
                    "materialized": strategy == "split",
                    "verified_default_decision": case["_default"],
                    "compile_ms": 12.5 + index + (0.5 if strategy == "fused" else 0.0),
                    "static_launch_count": len(kernels),
                    "kernels": kernels,
                    "trace_path": str(trace.relative_to(run_dir)),
                    "trace_verdict": "edge-materialized" if strategy == "split" else "edge-elided",
                }
            )
    return variants


def _checks(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for strategy in ("split", "fused"):
            for mode in ("direct", "replay"):
                for seed in case["input_seeds"]:
                    input_id = "zero" if seed == 0 else f"seed:{seed}"
                    rows.append(
                        {
                            "case_id": case["case_id"],
                            "strategy": strategy,
                            "input_id": input_id,
                            "mode": mode,
                            "reference": "eager-cpu",
                            "tolerance": {"atol": 1e-6 if case["dtype"] == "float32" else 2e-3, "rtol": 1e-5 if case["dtype"] == "float32" else 2e-3},
                            "output_shape": case["t_shape"],
                            "output_dtype": case["dtype"],
                            "nonfinite_count": 0,
                            "max_abs_error": 0.0,
                            "max_rel_error": 0.0,
                            "pass": True,
                            "reason": None,
                            "synthetic": True,
                        }
                    )
    return rows


def _block_values(case: dict[str, Any], mode: str) -> tuple[list[float], list[float]]:
    if case["case_id"] == "R1-m64-k512-f32-contig-t":
        splits = list(NEAR_ZERO_SPLIT)
        fused = list(NEAR_ZERO_FUSED)
    else:
        splits = [case["_split_base"] + delta for delta in BLOCK_DELTAS]
        fused = [case["_fused_base"] + delta for delta in FUSED_DELTAS]
    factor = 0.90 if mode == "replay" else 1.0
    factor += 0.005 if mode == "replay" and case["template"] == "P2" else 0.0
    return [value * factor for value in splits], [value * factor for value in fused]


def _timings(cases: list[dict[str, Any]], variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hashes = {(row["case_id"], row["strategy"]): row["artifact_hash"] for row in variants}
    rows = []
    for case in cases:
        case_id = case["case_id"]
        missing_half = case_id == "P1-n65536-f16-contig-t"
        retry_case = case_id == "P2-n16384-f32-contig-t"
        for mode in ("direct", "replay"):
            splits, fused = _block_values(case, mode)
            n = 2000 if mode == "replay" else 1000
            for block in range(BLOCKS):
                attempts = [1]
                if retry_case and block in (0, 1):
                    attempts = [1, 2]
                for attempt in attempts:
                    order = "SF" if block < BLOCKS // 2 else "FS"
                    quality = {"split": "valid", "fused": "valid"}
                    reason = {"split": None, "fused": None}
                    if retry_case and attempt == 1 and block == 0:
                        quality["fused"] = "contaminated"
                        reason["fused"] = "external GPU activity in telemetry window"
                    if retry_case and attempt == 1 and block == 1:
                        quality["split"] = "contaminated"
                        reason["split"] = "external GPU activity in telemetry window"
                    for strategy, values in (("split", splits), ("fused", fused)):
                        if missing_half and strategy == "fused" and block == 2:
                            continue
                        us = values[block] + (0.25 if attempt == 2 else 0.0)
                        rows.append(
                            {
                                "case_id": case_id,
                                "mode": mode,
                                "block": block,
                                "attempt": attempt,
                                "strategy": strategy,
                                "iteration_order": order,
                                "n": n,
                                "elapsed_ms": us * n / 1000.0,
                                "us_per_call": us,
                                "window": {
                                    "start": f"2026-11-25T00:{block:02d}:00Z",
                                    "end": f"2026-11-25T00:{block:02d}:02Z",
                                },
                                "device": "synthetic-cpu",
                                "gpu_uuid": "SYNTHETIC-NO-GPU",
                                "stream": "default",
                                "artifact_hash": hashes[(case_id, strategy)],
                                "quality": quality[strategy],
                                "reason": reason[strategy],
                                "synthetic": True,
                            }
                        )
    return rows


def _calibration() -> list[dict[str, Any]]:
    rows = []
    for mode, slope, intercept in (("direct", 4.0, 2.0), ("replay", 0.8, 0.3)):
        for order, k in enumerate((1, 8, 32, 128), 1):
            us = intercept + slope * k
            repeats = 100
            rows.append(
                {
                    "mode": mode,
                    "probe": "launch",
                    "size_or_k": k,
                    "block": order - 1,
                    "repeats": repeats,
                    "elapsed_ms": us * repeats / 1000.0,
                    "order": order,
                    "derived_unit": "us_per_kernel",
                    "environment_ref": "gpu.json",
                    "synthetic": True,
                }
            )
    for mode in ("direct", "replay"):
        for order, size in enumerate((256, 1 << 20, 16 << 20, 64 << 20), 1):
            repeats = 5
            elapsed_ms = 2.0 * size * repeats / 500.0 / 1000.0
            rows.append(
                {
                    "mode": mode,
                    "probe": "copy",
                    "buffer_bytes": size,
                    "block": order - 1,
                    "repeats": repeats,
                    "elapsed_ms": elapsed_ms,
                    "order": order,
                    "derived_unit": "bytes_per_us",
                    "environment_ref": "gpu.json",
                    "synthetic": True,
                }
            )
    return rows


def _telemetry(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": "2026-11-25T00:00:00Z",
            "gpu_uuid": "SYNTHETIC-NO-GPU",
            "temperature_c": None,
            "clock_sm_mhz": None,
            "power_w": None,
            "throttle": "fixture",
            "external_activity": False,
            "source": "synthetic",
        }
    ]


def _leaking_input_folds(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Deliberately bad assignment: both R2 cases are the same template but are
    # placed on opposite sides of the same fold.
    return [
        {
            "case_id": "R2-m64-k512-f32-contig-t",
            "mode": "direct",
            "fold": "naive-row-split",
            "train_case_ids": ["R2-m256-k1024-f16-contig-t"],
            "test_case_ids": ["R2-m64-k512-f32-contig-t"],
            "synthetic": True,
        },
        {
            "case_id": "R2-m256-k1024-f16-contig-t",
            "mode": "direct",
            "fold": "naive-row-split-2",
            "train_case_ids": ["R2-m64-k512-f32-contig-t"],
            "test_case_ids": ["R2-m256-k1024-f16-contig-t"],
            "synthetic": True,
        },
    ]


def write_collection_schema_copy(src_run: Path, dst_run: Path) -> Path:
    """Rewrite the synthetic fixture using the collection harness spellings.

    This is a compatibility fixture, not a second experiment.  It exercises the
    aliases documented in docs/analysis-usage.md (shape lists, logical_save_bytes,
    seeds, artifact_dir/output_code_sha256, n_kernels, default_maps_to,
    iteration_order lists, window_start_utc, kind=launch/bandwidth).
    """
    if dst_run.exists():
        shutil.rmtree(dst_run)
    shutil.copytree(src_run, dst_run)

    run = json.loads((dst_run / "run.json").read_text(encoding="utf-8"))
    run.update(
        {
            "run_id": dst_run.name,
            "purpose": "pilot real-schema compatibility check (SYNTHETIC)",
            "synthetic": True,
            "status": "running",
            "started_utc": run.pop("started_at"),
            "finished_utc": run.pop("ended_at"),
            "command_argv": ["python", "-m", "fusionaudit", "fixtures"],
            "code": {"commit": "synthetic-compat", "dirty": False},
            "protocol": {"version": "fb-4090-p1-v1", "blocks": BLOCKS, "modes": ["direct", "replay"], "seed": 17},
            "cloud": {"platform": "local", "instance_spec": "none", "image_id": "none", "persistent_root": str(dst_run.parent)},
        }
    )
    for key in ("started_at", "ended_at", "command", "code_hash", "expected_blocks", "expected_case_count", "planned_template_counts", "calibration"):
        run.pop(key, None)
    write_json(dst_run / "run.json", run)

    cases = _read_jsonl(dst_run / "cases.jsonl")
    for case in cases:
        shape = case["shape"]
        dims = [shape["N"]] if "N" in shape else [shape["M"], shape["K"]]
        case["shape"] = dims
        case["kind"] = "elementwise" if "N" in shape else "reduction"
        case["seeds"] = case.pop("input_seeds")
        case["logical_save_bytes"] = case.pop("logical_saved_bytes")
        case["planned_status"] = "planned"
        if "N" in shape:
            case["n"] = shape["N"]
            case["grid_axes"] = {"N": shape["N"]}
        else:
            case["m"] = shape["M"]
            case["k"] = shape["K"]
            case["grid_axes"] = {"M": shape["M"], "K": shape["K"]}
    write_jsonl(dst_run / "cases.jsonl", cases)

    variants = _read_jsonl(dst_run / "variants.jsonl")
    for variant in variants:
        old = Path(variant["artifact_path"])
        artifact_dir = dst_run / old.parent
        output_code = artifact_dir / f"{variant['strategy']}_output_code.py"
        shutil.copyfile(dst_run / old, output_code)
        digest = hashlib.sha256(output_code.read_bytes()).hexdigest()
        variant.update(
            {
                "artifact_dir": str(old.parent),
                "output_code": str(output_code.relative_to(dst_run)),
                "output_code_sha256": digest,
                "artifact_sha256": digest,
                "status": "valid",
                "kernel_count": variant.pop("static_launch_count"),
                "default_maps_to": variant.pop("verified_default_decision"),
            }
        )
        for key in ("artifact_path", "artifact_hash"):
            variant.pop(key, None)
    write_jsonl(dst_run / "variants.jsonl", variants)

    checks = []
    for check in _read_jsonl(dst_run / "checks.jsonl"):
        check["out_shape"] = check.pop("output_shape")
        check["out_dtype"] = check.pop("output_dtype")
        check["max_abs_err"] = check.pop("max_abs_error")
        check["max_rel_err"] = check.pop("max_rel_error")
        check["nonfinite_out"] = check.pop("nonfinite_count")
        checks.append(check)
    write_jsonl(dst_run / "checks.jsonl", checks)

    timings = []
    for timing in _read_jsonl(dst_run / "timings.jsonl"):
        timing["run_id"] = dst_run.name
        timing["block_pattern"] = timing["iteration_order"]
        timing["iteration_order"] = ["split", "fused"] if timing["block_pattern"] == "SF" else ["fused", "split"]
        window = timing.pop("window")
        timing["window_start_utc"] = window["start"]
        timing["window_end_utc"] = window["end"]
        timing["quality"] = "ok" if timing["quality"] == "valid" else timing["quality"]
        timings.append(timing)
    write_jsonl(dst_run / "timings.jsonl", timings)

    calibration = []
    for row in _read_jsonl(dst_run / "calibration.jsonl"):
        if row.get("buffer_bytes"):
            row["kind"] = "bandwidth"
            row["probe"] = "BW"
        else:
            row["kind"] = "launch"
            row["probe"] = "L"
        calibration.append(row)
    write_jsonl(dst_run / "calibration.jsonl", calibration)

    (dst_run / "environment").mkdir(exist_ok=True)
    write_text(dst_run / "environment" / "environment.lock.json", '{"gpu": "none"}\n')
    write_json(dst_run / "environment" / "environment.json", {"gpu": {"uuid": "SYNTHETIC-NO-GPU"}})
    (dst_run / "results.jsonl").unlink(missing_ok=True)
    return dst_run


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_environment(run_dir: Path) -> None:
    write_text(
        run_dir / "environment.lock",
        "# SYNTHETIC fixture lock\npython=3.12.15\nnumpy=none\nmatplotlib=none\n",
    )
    write_json(
        run_dir / "gpu.json",
        {
            "synthetic": True,
            "gpu_name": "none (fixture)",
            "gpu_uuid": "SYNTHETIC-NO-GPU",
            "note": "This run exists only to validate the analysis entrypoint.",
        },
    )


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
