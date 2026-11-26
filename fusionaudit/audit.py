"""Data-contract audit, attempt selection, calibration and feature extraction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config
from .io import (
    hash_matches,
    issue,
    load_json,
    load_jsonl,
    normalize_hash,
    sha256_file,
    sha256_json,
    source_of,
    strip_private,
)
from .stats import mean, median


REQUIRED_RUN_FIELDS = (
    "run_id",
    "purpose",
    "protocol_hash",
    "corpus_hash",
    "code_hash",
    "command",
    "started_at",
    "ended_at",
    "seed",
    "owner",
    "status",
)

FORMAL_EXPECTED_CASE_COUNT = 72
FORMAL_TEMPLATE_COUNTS = {"P1": 12, "P2": 12, "R1": 24, "R2": 24}


def _normalize_run(run: dict[str, Any], run_dir: Path, issues: list[dict[str, Any]]) -> None:
    """Accept both the fixture/contract spelling and the collection harness spelling."""
    if not run:
        return
    if not run.get("started_at") and run.get("started_utc"):
        run["started_at"] = run.get("started_utc")
    if not run.get("ended_at") and run.get("finished_utc"):
        run["ended_at"] = run.get("finished_utc")
    if not run.get("command"):
        argv = run.get("command_argv")
        if isinstance(argv, (list, tuple)):
            run["command"] = " ".join(str(part) for part in argv)
        elif argv:
            run["command"] = str(argv)
    if not run.get("code_hash"):
        code = run.get("code")
        if isinstance(code, dict) and code:
            run["code_hash"] = "sha256:" + sha256_json(code)
        elif run.get("code_sha256"):
            run["code_hash"] = run.get("code_sha256")
    cloud = run.get("cloud")
    if isinstance(cloud, dict):
        aliases = {
            "platform": "cloud",
            "instance_spec": "instance",
            "image_id": "image",
            "persistent_root": "persistent_volume",
        }
        for source, target in aliases.items():
            if not run.get(target) and cloud.get(source):
                run[target] = cloud.get(source)
    environment = run.get("environment")
    if isinstance(environment, dict):
        path = environment.get("path")
        if path and (run_dir / str(path)).is_file():
            if not run.get("gpu_snapshot"):
                run["gpu_snapshot"] = path
            if not run.get("environment_lock"):
                lock_candidate = Path(str(path)).parent / "environment.lock.json"
                if (run_dir / lock_candidate).is_file():
                    run["environment_lock"] = str(lock_candidate)
                else:
                    run["environment_lock"] = path
    if not run.get("environment_lock"):
        for candidate in ("environment/environment.lock.json", "environment/environment.json", "environment.lock"):
            if (run_dir / candidate).is_file():
                run["environment_lock"] = candidate
                break
    if not run.get("gpu_snapshot"):
        for candidate in ("environment/environment.json", "environment/gpu.json", "gpu.json"):
            if (run_dir / candidate).is_file():
                run["gpu_snapshot"] = candidate
                break
    if run.get("environment_lock") and not (run_dir / str(run["environment_lock"])).is_file():
        issues.append(issue("warning", "missing_evidence", f"environment lock path does not exist: {run['environment_lock']}", file=str(run["environment_lock"])))


def _is_formal_run(run: dict[str, Any]) -> bool:
    purpose = str(run.get("purpose", "")).lower()
    status = str(run.get("status", "")).lower()
    if any(word in purpose for word in ("pilot", "synthetic", "fixture", "smoke")):
        return False
    if status in {"formal", "main", "production", "complete", "completed"}:
        return True
    return any(word in purpose for word in ("full", "corpus", "formal"))


REQUIRED_CASE_FIELDS = (
    "case_id",
    "template",
    "shape",
    "dtype",
    "layout",
    "edge",
    "input_seeds",
    "t_shape",
    "t_dtype",
    "logical_saved_bytes",
    "planned_status",
)


def _norm_quality(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("status")
    if value is None:
        return None
    return str(value).strip().lower()


def _norm_order(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        first = str(value[0]).strip().lower()
        if first.startswith("split"):
            return "SF"
        if first.startswith("fused"):
            return "FS"
        return None
    text = str(value).strip().lower().replace("_", "-").replace(" ", "")
    if text in {"sf", "s-f", "split-fused", "splitfirst", "split-first"}:
        return "SF"
    if text in {"fs", "f-s", "fused-split", "fusedfirst", "fused-first"}:
        return "FS"
    return str(value)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "pass", "passed"}
    return False


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _shape_summary(shape: Any, case: dict[str, Any] | None = None) -> dict[str, Any]:
    template = str((case or {}).get("template", ""))
    extras = (case or {}).get("extras") if isinstance((case or {}).get("extras"), dict) else {}
    summary: dict[str, Any] = {}
    if isinstance(shape, dict):
        summary.update(shape)
    elif isinstance(shape, (list, tuple)):
        dims = [int(value) if isinstance(value, (int, float)) else value for value in shape]
        summary["dims"] = dims
        if len(dims) == 1:
            summary["N"] = dims[0]
        elif len(dims) == 2:
            if template.startswith("R"):
                summary["M"], summary["K"] = dims
            else:
                summary["N"] = dims[0]
                summary["dims"] = dims
    else:
        summary["raw"] = shape
    # Corpus rows also carry explicit n / m / k / grid_axes aliases.
    for key in ("n", "m", "k"):
        if key in (case or {}):
            summary.setdefault(key.upper(), (case or {})[key])
    for key, value in extras.items():
        if key in ("n", "m", "k"):
            summary.setdefault(key.upper(), value)
    grid = (case or {}).get("grid_axes") or extras.get("grid_axes")
    if isinstance(grid, dict):
        for key in ("N", "M", "K"):
            if key in grid:
                summary.setdefault(key, grid[key])
    return summary


def _case_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    shape = _shape_summary(case.get("shape"), case)
    if "N" in shape:
        return (case.get("template", ""), str(case.get("dtype", "")), _as_float(shape.get("N")) or 0.0, case.get("case_id", ""))
    return (
        case.get("template", ""),
        str(case.get("dtype", "")),
        _as_float(shape.get("M")) or 0.0,
        _as_float(shape.get("K")) or 0.0,
        case.get("case_id", ""),
    )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-9, 1e-6 * max(abs(a), abs(b)))


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


class AuditResult(dict):
    """Plain dict with attribute-style access for the pipeline's core tables."""


def audit_run(run_dir: Path, *, expected_blocks: int | None = None) -> AuditResult:
    run_dir = run_dir.resolve()
    issues: list[dict[str, Any]] = []

    run = load_json(run_dir / "run.json", issues)
    cases = load_jsonl(run_dir / "cases.jsonl", issues)
    variants = load_jsonl(run_dir / "variants.jsonl", issues)
    checks = load_jsonl(run_dir / "checks.jsonl", issues)
    timings = load_jsonl(run_dir / "timings.jsonl", issues)
    calibration_rows = load_jsonl(run_dir / "calibration.jsonl", issues)
    results_input = load_jsonl(run_dir / "results.jsonl", issues, required=False)
    telemetry = load_jsonl(run_dir / "telemetry.jsonl", issues, required=False)
    profiles = load_jsonl(run_dir / "profiles.jsonl", issues, required=False)

    # ------------------------------------------------------------ run metadata
    _normalize_run(run, run_dir, issues)
    for field in REQUIRED_RUN_FIELDS:
        if run.get(field) in (None, ""):
            severity = "error" if field in {"run_id", "purpose", "protocol_hash", "status"} else "warning"
            issues.append(issue(severity, "missing_run_field", f"run.json lacks required field '{field}'", file="run.json"))

    synthetic = bool(run.get("synthetic", False)) or "synthetic" in str(run.get("run_id", "")).lower()
    if synthetic:
        run["synthetic"] = True

    protocol = run.get("protocol") if isinstance(run.get("protocol"), dict) else {}
    blocks = (
        expected_blocks
        or _as_int(run.get("expected_blocks"))
        or _as_int(protocol.get("blocks"))
        or config.DEFAULT_EXPECTED_BLOCKS
    )
    if blocks <= 0:
        issues.append(issue("error", "invalid_expected_blocks", f"expected_blocks must be positive, got {blocks}"))
        blocks = config.DEFAULT_EXPECTED_BLOCKS

    # ------------------------------------------------------------------- cases
    case_index: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("case_id")
        if not case_id:
            issues.append(issue("error", "missing_case_id", "cases.jsonl row lacks case_id", file=source_of(case)[0], line=source_of(case)[1]))
            continue
        if "logical_saved_bytes" not in case and "logical_save_bytes" in case:
            case["logical_saved_bytes"] = case.get("logical_save_bytes")
        if "input_seeds" not in case and "seeds" in case:
            case["input_seeds"] = case.get("seeds")
        if "planned_status" not in case:
            case["planned_status"] = "planned"
        case["_shape"] = _shape_summary(case.get("shape"), case)
        if case_id in case_index:
            issues.append(issue("error", "duplicate_case", f"duplicate case_id '{case_id}'", case_id=case_id, file=source_of(case)[0], line=source_of(case)[1]))
            continue
        for field in REQUIRED_CASE_FIELDS:
            if case.get(field) is None:
                issues.append(issue("error", "missing_case_field", f"case '{case_id}' lacks '{field}'", case_id=case_id, file=source_of(case)[0], line=source_of(case)[1]))
        if case.get("logical_saved_bytes") is not None and (_as_float(case.get("logical_saved_bytes")) or 0) < 0:
            issues.append(issue("error", "invalid_case", f"case '{case_id}' has negative logical_saved_bytes", case_id=case_id, file=source_of(case)[0], line=source_of(case)[1]))
        case_index[case_id] = case

    for case_id, case in case_index.items():
        status = str(case.get("planned_status", "")).lower()
        if status not in config.VALID_CASE_STATUS and not case.get("exclusion_reason"):
            issues.append(
                issue("warning", "case_not_planned", f"case '{case_id}' has status '{status}' without exclusion_reason", case_id=case_id, file=source_of(case)[0], line=source_of(case)[1])
            )

    expected_case_count = _as_int(run.get("expected_case_count"))
    planned_counts = run.get("planned_template_counts")
    if _is_formal_run(run):
        if expected_case_count is None:
            expected_case_count = FORMAL_EXPECTED_CASE_COUNT
            issues.append(issue("info", "expected_counts_applied", f"formal run: using {FORMAL_EXPECTED_CASE_COUNT} expected cases from the frozen corpus", file="run.json"))
        if not isinstance(planned_counts, dict):
            planned_counts = dict(FORMAL_TEMPLATE_COUNTS)
            issues.append(issue("info", "expected_counts_applied", "formal run: using template counts 12/12/24/24 from the frozen corpus", file="run.json"))
    if expected_case_count is not None and len(case_index) != expected_case_count:
        issues.append(issue("error", "case_count_mismatch", f"expected {expected_case_count} unique cases, found {len(case_index)}", file="run.json"))
    if isinstance(planned_counts, dict):
        actual_counts = Counter(str(case.get("template")) for case in case_index.values())
        for template, expected in planned_counts.items():
            if actual_counts.get(str(template), 0) != _as_int(expected):
                issues.append(
                    issue("error", "template_count_mismatch", f"template {template}: planned {expected}, found {actual_counts.get(str(template), 0)}", file="run.json")
                )

    # ---------------------------------------------------------------- variants
    variant_index: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_variants: set[tuple[str, str]] = set()
    for variant in variants:
        case_id = variant.get("case_id")
        strategy = variant.get("strategy")
        if case_id not in case_index:
            issues.append(issue("error", "dangling_reference", f"variant references unknown case '{case_id}'", case_id=case_id, strategy=strategy, file=source_of(variant)[0], line=source_of(variant)[1]))
            continue
        if strategy not in config.VALID_STRATEGIES:
            issues.append(issue("error", "invalid_strategy", f"case '{case_id}': invalid strategy '{strategy}'", case_id=case_id, strategy=strategy, file=source_of(variant)[0], line=source_of(variant)[1]))
            continue
        if not variant.get("artifact_hash"):
            for alias in ("artifact_sha256", "output_code_sha256"):
                if variant.get(alias):
                    variant["artifact_hash"] = variant[alias]
                    break
        if not variant.get("artifact_path"):
            for alias in ("artifact_dir", "output_code"):
                if variant.get(alias):
                    variant["artifact_path"] = variant[alias]
                    break
        if not variant.get("artifact_path"):
            artifacts = variant.get("artifacts")
            if isinstance(artifacts, list) and artifacts:
                variant["artifact_path"] = artifacts[0]
        if variant.get("static_launch_count") is None:
            for alias in ("n_kernels", "kernel_count"):
                if variant.get(alias) is not None:
                    variant["static_launch_count"] = variant[alias]
                    break
        if variant.get("kernels") is None and isinstance(variant.get("kernel_list"), list):
            variant["kernels"] = variant["kernel_list"]
        if not variant.get("verified_default_decision"):
            for alias in ("default_maps_to", "default_equivalent"):
                if variant.get(alias) is not None:
                    variant["verified_default_decision"] = variant[alias]
                    break
        if variant.get("verified_default_decision") == "distinct":
            variant["verified_default_decision"] = "other"
        if not variant.get("verified_default_decision"):
            if variant.get("partition_equal_default_fused") is True:
                variant["verified_default_decision"] = "fused"
            elif variant.get("partition_equal_default_split") is True:
                variant["verified_default_decision"] = "split"
        key = (str(case_id), str(strategy))
        if key in variant_index:
            duplicate_variants.add(key)
            issues.append(issue("error", "duplicate_variant", f"case '{case_id}': duplicate variant for strategy '{strategy}'", case_id=case_id, strategy=strategy, file=source_of(variant)[0], line=source_of(variant)[1]))
            continue
        variant_index[key] = variant

    for (case_id, strategy), variant in variant_index.items():
        artifact_path = variant.get("artifact_path")
        artifact_hash = variant.get("artifact_hash")
        variant_status = str(variant.get("status", "")).lower()
        failed_variant = bool(variant_status) and variant_status not in {"valid", "cache_only", "ok"}
        severity = "warning" if (failed_variant or synthetic) else "error"
        if not artifact_path:
            issues.append(issue(severity, "missing_artifact_path", f"case '{case_id}' strategy '{strategy}' has no artifact_path", case_id=case_id, strategy=strategy))
        else:
            path = run_dir / str(artifact_path)
            if not path.exists():
                issues.append(issue(severity, "artifact_missing", f"artifact does not exist: {artifact_path}", case_id=case_id, strategy=strategy, file=artifact_path))
            elif path.is_dir():
                artifact_files = variant.get("artifact_files")
                if isinstance(artifact_files, dict):
                    for name, expected in artifact_files.items():
                        file = path / name
                        ok, actual = hash_matches(file, expected)
                        if not ok:
                            issues.append(issue("error", "artifact_hash_mismatch",
                                                f"{case_id}/{strategy}: {name} hash mismatch ({actual})",
                                                case_id=case_id, strategy=strategy, file=str(file)))
                output_code = variant.get("output_code")
                if output_code and (run_dir / str(output_code)).is_file() and variant.get("output_code_sha256"):
                    ok, actual = hash_matches(run_dir / str(output_code), variant.get("output_code_sha256"))
                    if not ok:
                        issues.append(issue("error", "artifact_hash_mismatch", f"case '{case_id}' strategy '{strategy}': output_code hash mismatch ({actual})", case_id=case_id, strategy=strategy, file=str(output_code)))
                elif artifact_hash and not failed_variant:
                    issues.append(issue("warning", "artifact_hash_unverified", f"case '{case_id}' strategy '{strategy}': artifact_dir hash is not independently recomputed", case_id=case_id, strategy=strategy, file=artifact_path))
            elif not artifact_hash:
                issues.append(issue(severity, "missing_artifact_hash", f"case '{case_id}' strategy '{strategy}' has no artifact_hash", case_id=case_id, strategy=strategy))
            else:
                ok, actual = hash_matches(path, artifact_hash)
                if not ok:
                    issues.append(issue("error", "artifact_hash_mismatch", f"case '{case_id}' strategy '{strategy}': artifact hash {artifact_hash} != {actual}", case_id=case_id, strategy=strategy, file=artifact_path))
        trace_path = variant.get("trace_path")
        if trace_path and not (run_dir / str(trace_path)).is_file():
            issues.append(issue(severity, "trace_missing", f"trace does not exist: {trace_path}", case_id=case_id, strategy=strategy, file=str(trace_path)))
        decision = variant.get("verified_default_decision")
        if decision not in {"split", "fused", "other", None}:
            issues.append(issue("error", "invalid_default_decision", f"case '{case_id}': invalid default decision '{decision}'", case_id=case_id, strategy=strategy))

    for case_id, case in case_index.items():
        if not _case_is_analyzable(case):
            continue
        if not any((variant_index.get((case_id, strategy)) or {}).get("verified_default_decision") for strategy in ("split", "fused", "default")):
            issues.append(issue("warning", "default_decision_missing", f"case '{case_id}': no verified default decision; default coverage will be reduced", case_id=case_id, file=source_of(case)[0], line=source_of(case)[1]))

    # ---------------------------------------------------------------- checks
    check_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for check in checks:
        case_id = check.get("case_id")
        strategy = check.get("strategy")
        mode = check.get("mode")
        if case_id not in case_index:
            issues.append(issue("error", "dangling_reference", f"check references unknown case '{case_id}'", case_id=case_id, strategy=strategy, mode=mode, file=source_of(check)[0], line=source_of(check)[1]))
            continue
        if strategy not in {"split", "fused", "default"}:
            issues.append(issue("error", "invalid_check", f"case '{case_id}': check has invalid strategy '{strategy}'", case_id=case_id, strategy=strategy, mode=mode, file=source_of(check)[0], line=source_of(check)[1]))
            continue
        if mode not in config.VALID_MODES and mode not in {"load", None}:
            issues.append(issue("warning", "invalid_check", f"case '{case_id}': check has non-timing mode '{mode}'", case_id=case_id, strategy=strategy, mode=mode, file=source_of(check)[0], line=source_of(check)[1]))
        check_mode = mode if mode in config.VALID_MODES else "*"
        check["_pass"] = _is_true(check.get("pass"))
        check_index[(str(case_id), str(strategy), str(check_mode))].append(check)
        if not check["_pass"] and strategy in {"split", "fused"}:
            issues.append(
                issue("warning", "check_failed", f"case '{case_id}' {strategy}/{mode} check failed: {check.get('reason') or 'no reason'}", case_id=case_id, strategy=strategy, mode=mode, file=source_of(check)[0], line=source_of(check)[1])
            )

    # --------------------------------------------------------------- timings
    timing_index: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    timing_rows_valid = 0
    for timing in timings:
        case_id = timing.get("case_id")
        mode = timing.get("mode")
        strategy = timing.get("strategy")
        block = _as_int(timing.get("block"))
        attempt = _as_int(timing.get("attempt"))
        if case_id not in case_index:
            issues.append(issue("error", "dangling_reference", f"timing references unknown case '{case_id}'", case_id=case_id, mode=mode, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1]))
            continue
        if mode not in config.VALID_MODES:
            issues.append(issue("error", "invalid_timing", f"case '{case_id}': invalid mode '{mode}'", case_id=case_id, mode=mode, file=source_of(timing)[0], line=source_of(timing)[1]))
            continue
        if strategy not in {"split", "fused", "default"}:
            issues.append(issue("error", "invalid_timing", f"case '{case_id}': invalid timing strategy '{strategy}'", case_id=case_id, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1]))
            continue
        if block is None or block < 0:
            issues.append(issue("error", "invalid_timing", f"case '{case_id}': invalid block '{timing.get('block')}'", case_id=case_id, mode=mode, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1]))
            continue
        if attempt is None or attempt < 1:
            issues.append(issue("error", "invalid_timing", f"case '{case_id}': invalid attempt '{timing.get('attempt')}'", case_id=case_id, mode=mode, block=block, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1]))
            continue
        key = (str(case_id), str(mode), block, attempt, str(strategy))
        timing["_valid"] = True
        timing["_reason"] = None
        timing["_block"] = block
        timing["_attempt"] = attempt
        n = _as_int(timing.get("n"))
        elapsed_ms = _as_float(timing.get("elapsed_ms"))
        if n is None or n <= 0:
            timing["_valid"] = False
            timing["_reason"] = "n missing or non-positive"
        elif elapsed_ms is None or elapsed_ms <= 0:
            timing["_valid"] = False
            timing["_reason"] = "elapsed_ms missing or non-positive"
        else:
            recomputed = elapsed_ms * 1000.0 / n
            timing["_us_per_call"] = recomputed
            stored = _as_float(timing.get("us_per_call"))
            if stored is not None and not _close(stored, recomputed):
                timing["_valid"] = False
                timing["_reason"] = f"stored us_per_call {stored} != elapsed_ms*1000/n {recomputed}"
                issues.append(
                    issue("error", "us_per_call_mismatch", f"case '{case_id}' block {block} attempt {attempt} {strategy}: stored {stored} vs recomputed {recomputed}", case_id=case_id, mode=mode, block=block, attempt=attempt, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1])
                )
        quality = _norm_quality(timing.get("quality"))
        timing["_quality"] = quality
        if quality not in config.VALID_QUALITY:
            timing["_valid"] = False
            timing["_reason"] = timing["_reason"] or f"quality={timing.get('quality')!r} ({timing.get('reason') or 'no reason'})"
        if timing["_valid"]:
            timing_rows_valid += 1
        if not timing.get("artifact_hash"):
            for alias in ("artifact_sha256", "output_code_sha256"):
                if timing.get(alias):
                    timing["artifact_hash"] = timing[alias]
                    break
        order = _norm_order(timing.get("iteration_order"))
        if order not in {"SF", "FS"}:
            order = _norm_order(timing.get("block_pattern")) or order
        timing["_order"] = order
        if order not in {"SF", "FS"}:
            issues.append(issue("warning", "invalid_iteration_order", f"case '{case_id}' block {block} attempt {attempt} {strategy}: iteration_order={timing.get('iteration_order')!r}", case_id=case_id, mode=mode, block=block, attempt=attempt, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1]))
        window = timing.get("window")
        if not isinstance(window, dict) and (timing.get("window_start_utc") or timing.get("window_end_utc")):
            window = {"start": timing.get("window_start_utc"), "end": timing.get("window_end_utc")}
        if isinstance(window, dict):
            start = _parse_time(window.get("start"))
            end = _parse_time(window.get("end"))
            if start and end and start >= end:
                issues.append(issue("warning", "invalid_window", f"case '{case_id}' block {block} attempt {attempt} {strategy}: window start >= end", case_id=case_id, mode=mode, block=block, attempt=attempt, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1]))
            elif not (start and end):
                issues.append(issue("warning", "unparsable_window", f"case '{case_id}' block {block} attempt {attempt} {strategy}: unparsable window", case_id=case_id, mode=mode, block=block, attempt=attempt, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1]))
        if strategy == "default":
            # Independent default timings are descriptive; decisions map back
            # to split/fused artifacts and never mix into the paired table.
            continue
        if key in timing_index:
            timing_index[key]["_valid"] = False
            timing_index[key]["_reason"] = "duplicate timing key"
            timing["_valid"] = False
            timing["_reason"] = "duplicate timing key"
            issues.append(issue("error", "duplicate_timing", f"duplicate timing row for case '{case_id}' {mode} block {block} attempt {attempt} {strategy}", case_id=case_id, mode=mode, block=block, attempt=attempt, strategy=strategy, file=source_of(timing)[0], line=source_of(timing)[1]))
            continue
        timing_index[key] = timing

    # ------------------------------------------------------- pair selection
    gpu_uuids = sorted({str(row.get("gpu_uuid")) for row in timings if row.get("gpu_uuid")})
    if len(gpu_uuids) > 1:
        issues.append(issue("error", "gpu_uuid_mismatch", f"timings contain multiple GPU UUIDs: {gpu_uuids}", file="timings.jsonl"))
    groups: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (case_id, mode, block, attempt, strategy), timing in timing_index.items():
        groups[(case_id, mode, block, attempt)][strategy] = timing

    block_audit: list[dict[str, Any]] = []
    selected_pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    attempts_by_block: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for key in groups:
        case_id, mode, block, attempt = key
        attempts_by_block[(case_id, mode, block)].append(attempt)
    for key in attempts_by_block:
        attempts_by_block[key] = sorted(set(attempts_by_block[key]))

    for (case_id, mode, block), attempts in sorted(attempts_by_block.items()):
        selected_attempt: int | None = None
        candidate: dict[str, Any] | None = None
        for attempt in attempts:
            group = groups[(case_id, mode, block, attempt)]
            pair, reasons = _qualify_pair(run_dir, case_id, mode, block, attempt, group, variant_index, duplicate_variants, check_index)
            row = {
                "case_id": case_id,
                "mode": mode,
                "block": block,
                "attempt": attempt,
                "n": pair.get("n") if pair else None,
                "iteration_order": pair.get("iteration_order") if pair else None,
                "split_line": _line_of(group.get("split")),
                "fused_line": _line_of(group.get("fused")),
                "status": "rejected",
                "reason": "; ".join(reasons) if reasons else None,
            }
            if pair is not None and selected_attempt is None:
                selected_attempt = attempt
                candidate = pair
                row["status"] = "selected"
                row["reason"] = None
            elif selected_attempt is not None:
                row["status"] = "not_used"
                row["reason"] = f"first qualified attempt is {selected_attempt}"
            else:
                code = "pair_missing_half" if any("missing split" in r or "missing fused" in r for r in reasons) else "pair_rejected"
                issues.append(
                    issue("warning", code, f"case '{case_id}' {mode} block {block} attempt {attempt}: {'; '.join(reasons)}", case_id=case_id, mode=mode, block=block, attempt=attempt, file="timings.jsonl", detail=reasons)
                )
            block_audit.append(row)
        if selected_attempt is None:
            block_audit.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "block": block,
                    "attempt": None,
                    "n": None,
                    "iteration_order": None,
                    "split_line": None,
                    "fused_line": None,
                    "status": "no_qualified_attempt",
                    "reason": "all attempts rejected",
                }
            )
        else:
            candidate["selected_attempt"] = selected_attempt
            selected_pairs[(case_id, mode)].append(candidate)

    # validate block ranges, coverage, ordering
    case_mode: dict[tuple[str, str], dict[str, Any]] = {}
    modes_present = sorted({mode for (_case, mode) in selected_pairs} | {mode for (_c, mode, _b, _a) in groups})
    for case_id, case in case_index.items():
        for mode in ("direct", "replay"):
            if mode not in modes_present and mode == "replay":
                # absence of replay is allowed; direct is reported below.
                pass
            pairs = sorted(selected_pairs.get((case_id, mode), []), key=lambda row: row["block"])
            blocks_seen = [row["block"] for row in pairs]
            duplicate_blocks = len(blocks_seen) != len(set(blocks_seen))
            expected_set = set(range(blocks))
            actual_set = set(blocks_seen)
            complete = len(pairs) == blocks and not duplicate_blocks and actual_set == expected_set
            if pairs and not complete:
                reason = f"{len(pairs)}/{blocks} qualified blocks"
                if duplicate_blocks:
                    reason += "; duplicate block ids"
                if actual_set - expected_set:
                    reason += f"; out-of-range blocks {sorted(actual_set - expected_set)}"
                issues.append(issue("error", "incomplete_case", f"case '{case_id}' {mode}: {reason}", case_id=case_id, mode=mode, detail={"blocks": blocks_seen, "expected": blocks}))
            if len(pairs) > blocks and not duplicate_blocks:
                issues.append(issue("warning", "block_count_mismatch", f"case '{case_id}' {mode}: {len(pairs)} blocks > expected {blocks}", case_id=case_id, mode=mode))
            for pair in pairs:
                if pair["block"] not in expected_set:
                    issues.append(issue("error", "block_out_of_range", f"case '{case_id}' {mode} block {pair['block']} not in 0..{blocks - 1}", case_id=case_id, mode=mode, block=pair["block"]))
            if pairs:
                orders = Counter(pair.get("iteration_order") for pair in pairs)
                expected_half = blocks / 2.0
                for order in ("SF", "FS"):
                    if abs(orders.get(order, 0) - expected_half) > 0.5:
                        issues.append(issue("warning", "block_order_imbalance", f"case '{case_id}' {mode}: {order}={orders.get(order, 0)}, expected about {expected_half:g}", case_id=case_id, mode=mode, detail=dict(orders)))
            t_split = [pair["split_us"] for pair in pairs]
            t_fused = [pair["fused_us"] for pair in pairs]
            n_values = sorted({pair["n"] for pair in pairs})
            if len(n_values) > 1:
                issues.append(issue("warning", "case_n_mismatch", f"case '{case_id}' {mode}: multiple n values across blocks {n_values}", case_id=case_id, mode=mode, detail=n_values))
            case_mode[(case_id, mode)] = {
                "case_id": case_id,
                "mode": mode,
                "pairs": pairs,
                "complete": complete,
                "selected_attempts": {pair["block"]: pair["selected_attempt"] for pair in pairs},
                "n_blocks": len(pairs),
                "expected_blocks": blocks,
                "T_split_us_mean": mean(t_split) if t_split else None,
                "T_fused_us_mean": mean(t_fused) if t_fused else None,
            }

    # -------------------------------------------------------- evidence files
    evidence_hashes: dict[str, str | None] = {}
    for field, label in (("environment_lock", "environment lock"), ("gpu_snapshot", "GPU snapshot")):
        value = run.get(field)
        if value:
            path = run_dir / str(value)
            if not path.is_file():
                issues.append(issue("error", "missing_evidence", f"{label} does not exist: {value}", file=str(value)))
            else:
                evidence_hashes[str(value)] = sha256_file(path)
        elif not synthetic:
            issues.append(issue("warning", "missing_evidence", f"run.json lacks {field}", file="run.json"))
    environment_meta = run.get("environment")
    if isinstance(environment_meta, dict) and environment_meta.get("lock_sha256") and run.get("environment_lock"):
        lock_path = run_dir / str(run["environment_lock"])
        actual = sha256_file(lock_path)
        if actual and normalize_hash(environment_meta.get("lock_sha256")) != actual:
            issues.append(issue("error", "environment_lock_hash_mismatch", f"run.environment.lock_sha256 != {run['environment_lock']}", file=str(run["environment_lock"])))

    # --------------------------------------------------------- input fold leakage
    leak = _audit_input_folds(results_input, case_index, issues)

    # ----------------------------------------------------------- calibration
    calibration = derive_calibration(calibration_rows, run, issues, run_dir)

    # --------------------------------------------------------------- funnel
    template_counts = Counter(str(case.get("template")) for case in case_index.values())
    status_counts = Counter(str(case.get("planned_status")) for case in case_index.values())
    complete_by_mode = {
        mode: sum(1 for (_case, m), info in case_mode.items() if m == mode and info["complete"])
        for mode in config.VALID_MODES
    }
    funnel = {
        "expected_blocks": blocks,
        "cases_total": len(case_index),
        "cases_synthetic": synthetic,
        "case_status_counts": dict(sorted(status_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "variant_rows": len(variants),
        "variant_keys": len(variant_index),
        "timing_rows": len(timings),
        "timing_rows_valid": timing_rows_valid,
        "qualified_pairs": sum(len(info["pairs"]) for info in case_mode.values()),
        "qualified_pairs_by_mode": {
            mode: sum(len(info["pairs"]) for (case, m), info in case_mode.items() if m == mode)
            for mode in config.VALID_MODES
        },
        "complete_cases": complete_by_mode,
        "incomplete_cases": {
            mode: sum(1 for (_case, m), info in case_mode.items() if m == mode and info["pairs"] and not info["complete"])
            for mode in config.VALID_MODES
        },
        "modes_with_data": dict(sorted(Counter(mode for _case, mode in case_mode).items())),
        "telemetry_rows": len(telemetry),
        "profile_rows": len(profiles),
        "gpu_uuids": gpu_uuids,
    }

    timing_counts: Counter[tuple[str, str]] = Counter()
    for (case_id, mode, _block, _attempt, _strategy) in timing_index:
        timing_counts[(case_id, mode)] += 1
    case_inventory = []
    for case in sorted(case_index.values(), key=_case_sort_key):
        case_id = str(case["case_id"])
        variants_for_case = {strategy: variant_index.get((case_id, strategy)) for strategy in ("default", "split", "fused")}
        row = {
            "case_id": case_id,
            "template": case.get("template"),
            "dtype": case.get("dtype"),
            "shape": case.get("_shape"),
            "planned_status": case.get("planned_status"),
            "exclusion_reason": case.get("exclusion_reason"),
            "variants": ",".join(
                f"{strategy}:{variant.get('status') or 'recorded'}" for strategy, variant in variants_for_case.items() if variant
            ) or None,
            "checks_split": len(check_index.get((case_id, "split", "direct"), [])),
            "checks_fused": len(check_index.get((case_id, "fused", "direct"), [])),
            "timing_rows": sum(timing_counts.get((case_id, mode), 0) for mode in config.VALID_MODES),
            "qualified_direct": len(case_mode.get((case_id, "direct"), {}).get("pairs", [])),
            "qualified_replay": len(case_mode.get((case_id, "replay"), {}).get("pairs", [])),
            "complete_direct": case_mode.get((case_id, "direct"), {}).get("complete", False),
            "complete_replay": case_mode.get((case_id, "replay"), {}).get("complete", False),
        }
        analyzable = _case_is_analyzable(case)
        if not analyzable:
            row["inventory_status"] = "excluded"
        elif row["complete_direct"] or row["complete_replay"]:
            row["inventory_status"] = "complete"
        elif row["timing_rows"]:
            row["inventory_status"] = "incomplete"
        else:
            row["inventory_status"] = "no_timing"
        case_inventory.append(row)

    # ---------------------------------------------------------- feature rows
    feature_rows: list[dict[str, Any]] = []
    for (case_id, mode), info in sorted(case_mode.items()):
        if not info["complete"]:
            continue
        case = case_index[case_id]
        if not _case_is_analyzable(case):
            continue
        features = compute_features(case, mode, info["T_split_us_mean"], variant_index, calibration, run_dir)
        feature_rows.append(features)
        info["features"] = features

    severity_counts = Counter(row["severity"] for row in issues)
    code_counts = Counter(row["code"] for row in issues)
    summary = {
        "severity_counts": dict(sorted(severity_counts.items())),
        "code_counts": dict(sorted(code_counts.items())),
        "error_count": severity_counts.get("error", 0),
        "warning_count": severity_counts.get("warning", 0),
        "info_count": severity_counts.get("info", 0),
    }

    return AuditResult(
        run_dir=run_dir,
        run=run,
        synthetic=synthetic,
        expected_blocks=blocks,
        cases=case_index,
        cases_ordered=sorted(case_index.values(), key=_case_sort_key),
        variants=variant_index,
        checks=check_index,
        timings=timing_index,
        calibration_rows=calibration_rows,
        calibration=calibration,
        case_modes=case_mode,
        feature_rows=feature_rows,
        issues=issues,
        summary=summary,
        funnel=funnel,
        block_audit=block_audit,
        case_inventory=case_inventory,
        evidence_hashes=evidence_hashes,
        input_fold_leak=leak,
        results_input=results_input,
        telemetry_rows=telemetry,
        profile_rows=profiles,
        files={
            "run.json": sha256_file(run_dir / "run.json"),
            "cases.jsonl": sha256_file(run_dir / "cases.jsonl"),
            "variants.jsonl": sha256_file(run_dir / "variants.jsonl"),
            "checks.jsonl": sha256_file(run_dir / "checks.jsonl"),
            "timings.jsonl": sha256_file(run_dir / "timings.jsonl"),
            "calibration.jsonl": sha256_file(run_dir / "calibration.jsonl"),
            "telemetry.jsonl": sha256_file(run_dir / "telemetry.jsonl"),
            "profiles.jsonl": sha256_file(run_dir / "profiles.jsonl"),
            "results.jsonl": sha256_file(run_dir / "results.jsonl"),
        },
    )


def _case_is_analyzable(case: dict[str, Any]) -> bool:
    status = str(case.get("planned_status", "")).lower()
    return status in config.VALID_CASE_STATUS and not case.get("exclusion_reason")


def _line_of(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    return source_of(row)[1]


def _qualify_pair(
    run_dir: Path,
    case_id: str,
    mode: str,
    block: int,
    attempt: int,
    group: dict[str, dict[str, Any]],
    variant_index: dict[tuple[str, str], dict[str, Any]],
    duplicate_variants: set[tuple[str, str]],
    check_index: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    split = group.get("split")
    fused = group.get("fused")
    if split is None:
        reasons.append("missing split timing")
    if fused is None:
        reasons.append("missing fused timing")
    if split is None or fused is None:
        for strategy, row in (("split", split), ("fused", fused)):
            if row is not None and not row.get("_valid", False):
                reasons.append(f"{strategy}: {row.get('_reason')}")
        return None, reasons

    for strategy, row in (("split", split), ("fused", fused)):
        if not row.get("_valid", False):
            reasons.append(f"{strategy}: {row.get('_reason')}")
        variant = variant_index.get((case_id, strategy))
        if variant is None:
            reasons.append(f"missing {strategy} variant")
        elif (case_id, strategy) in duplicate_variants:
            reasons.append(f"duplicate {strategy} variant")
        else:
            variant_status = variant.get("status")
            if variant_status is not None and str(variant_status).lower() not in {"valid", "cache_only"}:
                reasons.append(f"{strategy}: variant status={variant_status!r}")
            timing_hash = normalize_hash(row.get("artifact_hash"))
            variant_hash = normalize_hash(variant.get("artifact_hash"))
            if not timing_hash:
                reasons.append(f"{strategy}: timing lacks artifact_hash")
            elif not variant_hash:
                reasons.append(f"{strategy}: variant lacks artifact_hash")
            elif timing_hash != variant_hash:
                reasons.append(f"{strategy}: timing artifact_hash != variant artifact_hash")
        checks = check_index.get((case_id, strategy, mode), [])
        if not checks:
            checks = check_index.get((case_id, strategy, "*"), [])
        if not checks:
            reasons.append(f"{strategy}: missing correctness check for mode {mode}")
        elif not all(check.get("_pass") for check in checks):
            reasons.append(f"{strategy}: correctness check failed")
        if not row.get("_valid", False):
            continue

    n_split = _as_int(split.get("n"))
    n_fused = _as_int(fused.get("n"))
    if n_split is None or n_fused is None or n_split != n_fused:
        reasons.append(f"n mismatch: split={n_split}, fused={n_fused}")
    order_split = split.get("_order")
    order_fused = fused.get("_order")
    if order_split and order_fused and order_split != order_fused:
        reasons.append(f"iteration order mismatch: split={order_split}, fused={order_fused}")

    if reasons:
        return None, reasons
    return (
        {
            "case_id": case_id,
            "mode": mode,
            "block": block,
            "attempt": attempt,
            "split_us": float(split["_us_per_call"]),
            "fused_us": float(fused["_us_per_call"]),
            "n": n_split,
            "iteration_order": order_split,
            "split_line": _line_of(split),
            "fused_line": _line_of(fused),
            "split_window": split.get("window"),
            "fused_window": fused.get("window"),
        },
        [],
    )


def _audit_input_folds(
    results_input: list[dict[str, Any]],
    case_index: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check pre-existing fold assignments for same-template train/test overlap.

    The analysis never trusts these assignments: it always rebuilds
    leave-one-template-out folds.  This check exists so a leaked input table is
    visible instead of silently reused.
    """
    assignments = [row for row in results_input if row.get("fold") is not None]
    leaks: list[dict[str, Any]] = []
    viewed = 0
    for row in assignments:
        train_ids = row.get("train_case_ids") or row.get("train_ids")
        test_ids = row.get("test_case_ids") or row.get("test_ids")
        if not isinstance(train_ids, list) or not isinstance(test_ids, list):
            continue
        viewed += 1
        train_templates = {str(case_index[cid].get("template")) for cid in train_ids if cid in case_index}
        test_templates = {str(case_index[cid].get("template")) for cid in test_ids if cid in case_index}
        overlap = sorted(train_templates & test_templates)
        if overlap:
            leaks.append(
                {
                    "fold": row.get("fold"),
                    "overlap_templates": overlap,
                    "train_case_ids": train_ids,
                    "test_case_ids": test_ids,
                    "file": source_of(row)[0],
                    "line": source_of(row)[1],
                }
            )
    if leaks:
        issues.append(
            issue(
                "error",
                "template_leak",
                f"input fold assignment leaks {len(leaks)} fold(s) across templates; assignments are ignored and leave-one-template-out folds are rebuilt",
                file="results.jsonl",
                detail=leaks,
            )
        )
    elif assignments:
        issues.append(issue("info", "input_fold_ignored", f"ignored {len(assignments)} existing fold assignment row(s); leave-one-template-out folds are rebuilt from case templates", file="results.jsonl"))
    return {"assignment_rows": len(assignments), "checked_rows": viewed, "leaks": leaks}


def derive_calibration(
    rows: list[dict[str, Any]],
    run: dict[str, Any],
    issues: list[dict[str, Any]],
    run_dir: Path | None = None,
) -> dict[str, Any]:
    explicit = run.get("calibration") if isinstance(run.get("calibration"), dict) else {}
    summary = explicit.get("summary") if isinstance(explicit.get("summary"), dict) else {}
    fit_file: dict[str, Any] = {}
    if isinstance(explicit.get("path"), str) and run_dir is not None:
        fit_path = run_dir / explicit["path"]
        if fit_path.is_file():
            try:
                import json

                fit_file = json.loads(fit_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                fit_file = {}
    modes: dict[str, Any] = {}
    l2_bytes = _as_float(explicit.get("l2_bytes")) or _as_float((fit_file.get("bandwidth") or {}).get("l2_bytes"))
    for mode in sorted(config.VALID_MODES):
        launch = _explicit_value(explicit.get("launch_us_per_kernel"), mode)
        bw = _explicit_value(explicit.get("bandwidth_bytes_per_us"), mode)
        launch_source = "run.json" if launch is not None else None
        bw_source = "run.json" if bw is not None else None
        launch_detail: dict[str, Any] = {}
        bw_detail: dict[str, Any] = {}
        if launch is None and summary:
            launch = _as_float(summary.get(f"launch_{mode}_us_per_kernel"))
            if launch is not None:
                launch_source = "run.json:calibration.summary"
        if bw is None and summary:
            gbs = _as_float(summary.get("bw_main_gbs"))
            if gbs is not None:
                bw = gbs * 1000.0
                bw_source = "run.json:calibration.summary"
        if launch is None and fit_file:
            launch = _as_float((fit_file.get(f"launch_{mode}") or {}).get("slope"))
            if launch is not None:
                launch_source = "calibration_fit.json"
        if bw is None and fit_file:
            bw = _as_float((fit_file.get("bandwidth") or {}).get("bw_main_bytes_per_us"))
            if bw is not None:
                bw_source = "calibration_fit.json"
        # The contract asks for calibration recomputed from raw blocks.  Raw
        # derivation therefore wins over a stored fit when the rows are present.
        derived_launch, derived_launch_detail = _derive_launch(rows, mode)
        if derived_launch is not None:
            launch = derived_launch
            launch_detail = derived_launch_detail
            launch_source = "calibration.jsonl"
        derived_bw, derived_bw_detail = _derive_bandwidth(rows, mode, l2_bytes)
        if derived_bw is not None:
            bw = derived_bw
            bw_detail = derived_bw_detail
            bw_source = "calibration.jsonl"
        if launch is None or launch <= 0:
            derived_launch, derived_detail = _derive_launch(rows, mode)
            if derived_launch is not None:
                launch = derived_launch
                launch_detail = derived_detail
                launch_source = "calibration.jsonl"
        if bw is None or bw <= 0:
            derived_bw, derived_detail = _derive_bandwidth(rows, mode, l2_bytes)
            if derived_bw is not None:
                bw = derived_bw
                bw_detail = derived_detail
                bw_source = "calibration.jsonl"
        if launch is None:
            issues.append(issue("warning", "calibration_missing", f"no launch calibration for mode {mode}; x1 unavailable", mode=mode, file="calibration.jsonl"))
        elif launch <= 0:
            issues.append(issue("warning", "calibration_invalid", f"launch calibration for mode {mode} is non-positive ({launch})", mode=mode, file="calibration.jsonl"))
        if bw is None:
            issues.append(issue("warning", "calibration_missing", f"no bandwidth calibration for mode {mode}; x2 unavailable", mode=mode, file="calibration.jsonl"))
        elif bw <= 0:
            issues.append(issue("warning", "calibration_invalid", f"bandwidth calibration for mode {mode} is non-positive ({bw})", mode=mode, file="calibration.jsonl"))
        modes[mode] = {
            "launch_us_per_kernel": launch,
            "bandwidth_bytes_per_us": bw,
            "launch_source": launch_source,
            "bandwidth_source": bw_source,
            "launch_detail": launch_detail,
            "bandwidth_detail": bw_detail,
        }
    return {
        "modes": modes,
        "l2_bytes": l2_bytes,
        "environment_ref": explicit.get("environment_ref"),
    }


def _explicit_value(value: Any, mode: str) -> float | None:
    if isinstance(value, dict):
        return _as_float(value.get(mode))
    return _as_float(value)


def _derive_launch(rows: list[dict[str, Any]], mode: str) -> tuple[float | None, dict[str, Any]]:
    per_k: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("mode")) != mode:
            continue
        probe = str(row.get("probe", "")).lower()
        kind = str(row.get("kind", "")).lower()
        if not (kind == "launch" or "launch" in probe or probe in {"l", "empty"} or "empty" in probe):
            continue
        k = _as_int(row.get("k", row.get("size_or_k")))
        repeats = _as_int(row.get("repeats"))
        elapsed = _as_float(row.get("elapsed_ms"))
        if k is None or k <= 0 or repeats is None or repeats <= 0 or elapsed is None or elapsed <= 0:
            continue
        per_k[k].append(elapsed * 1000.0 / repeats)
    if len(per_k) < 2:
        return None, {"points": len(per_k)}
    points = sorted((k, median(values)) for k, values in per_k.items())
    n = len(points)
    sx = sum(k for k, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(k * k for k, _ in points)
    sxy = sum(k * y for k, y in points)
    denominator = n * sxx - sx * sx
    if denominator == 0:
        return None, {"points": n, "reason": "degenerate k values"}
    slope = (n * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / n
    return slope, {"points": n, "intercept_us": intercept, "raw": points}


def _derive_bandwidth(rows: list[dict[str, Any]], mode: str, l2_bytes: float | None = None) -> tuple[float | None, dict[str, Any]]:
    per_size: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("mode")) != mode:
            continue
        probe = str(row.get("probe", "")).lower()
        kind = str(row.get("kind", "")).lower()
        if not (kind == "bandwidth" or any(name in probe for name in ("copy", "bandwidth", "memcpy", "bw"))):
            continue
        size = _as_int(row.get("buffer_bytes", row.get("size_or_k")))
        repeats = _as_int(row.get("repeats"))
        elapsed = _as_float(row.get("elapsed_ms"))
        if size is None or size <= 0 or repeats is None or repeats <= 0 or elapsed is None or elapsed <= 0:
            continue
        per_size[size].append(2.0 * size * repeats / (elapsed * 1000.0))
    if not per_size:
        return None, {"points": 0}
    points = sorted((size, median(values)) for size, values in per_size.items())
    above = [point for point in points if l2_bytes and point[0] > l2_bytes]
    pool = above if l2_bytes else points
    largest = pool[-3:]
    bw = median([value for _, value in largest]) if len(largest) == 3 else None
    return bw, {
        "points": len(points),
        "l2_bytes": l2_bytes,
        "above_l2_sizes_bytes": [size for size, _ in above],
        "largest_sizes_bytes": [size for size, _ in largest],
        "values_bytes_per_us": [value for _, value in largest],
    }


def compute_features(
    case: dict[str, Any],
    mode: str,
    t_split_us: float | None,
    variant_index: dict[tuple[str, str], dict[str, Any]],
    calibration: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    cal = (calibration.get("modes") or {}).get(mode, {})
    launch = cal.get("launch_us_per_kernel")
    bw = cal.get("bandwidth_bytes_per_us")
    saved_bytes = _as_float(case.get("logical_saved_bytes"))
    split_variant = variant_index.get((case_id, "split"))
    fused_variant = variant_index.get((case_id, "fused"))

    split_launches, split_launch_source = _launch_count(split_variant)
    fused_launches, fused_launch_source = _launch_count(fused_variant)
    x1 = None
    if launch is not None and split_launches is not None and fused_launches is not None:
        x1 = (split_launches - fused_launches) * float(launch)
    x2 = None
    if bw is not None and saved_bytes is not None and bw > 0:
        x2 = saved_bytes / float(bw)

    o_split, split_occ_source = _occupancy(split_variant)
    o_fused, fused_occ_source = _occupancy(fused_variant)
    x3 = None
    if t_split_us is not None and o_split is not None and o_fused is not None:
        x3 = float(t_split_us) * (o_split - o_fused)

    return {
        "case_id": case_id,
        "mode": mode,
        "template": case.get("template"),
        "dtype": case.get("dtype"),
        "shape": case.get("_shape", case.get("shape")),
        "logical_saved_bytes": saved_bytes,
        "x1_us": x1,
        "x2_us": x2,
        "x3_us": x3,
        "x1_source": {
            "formula": "(split static launches - fused static launches) * L",
            "launch_us_per_kernel": launch,
            "split_launches": split_launches,
            "fused_launches": fused_launches,
            "calibration_source": cal.get("launch_source"),
            "split_variant_line": source_of(split_variant)[1] if split_variant else None,
            "fused_variant_line": source_of(fused_variant)[1] if fused_variant else None,
            "launch_source": split_launch_source,
        },
        "x2_source": {
            "formula": "logical_saved_bytes / BW",
            "logical_saved_bytes": saved_bytes,
            "bandwidth_bytes_per_us": bw,
            "calibration_source": cal.get("bandwidth_source"),
        },
        "x3_source": {
            "formula": "T_split * (o_split - o_fused)",
            "T_split_us": t_split_us,
            "o_split": o_split,
            "o_fused": o_fused,
            "split_occupancy_source": split_occ_source,
            "fused_occupancy_source": fused_occ_source,
        },
        "whitelist": list(config.FEATURE_WHITELIST),
        "forbidden_fields": list(config.FORBIDDEN_MODEL_FIELDS),
    }


def _launch_count(variant: dict[str, Any] | None) -> tuple[int | None, str | None]:
    if not variant:
        return None, None
    for field in ("static_launch_count", "compiled_launches", "launch_count", "n_kernels", "kernel_count"):
        value = _as_int(variant.get(field))
        if value is not None:
            return value, f"variants.{field}"
    kernels = variant.get("kernels")
    if isinstance(kernels, list):
        return len(kernels), "len(variants.kernels)"
    return None, None


def _occupancy(variant: dict[str, Any] | None) -> tuple[float | None, str | None]:
    if not variant:
        return None, None
    for field in ("region_occupancy", "occupancy", "static_occupancy", "occupancy_proxy"):
        direct = _as_float(variant.get(field))
        if direct is not None:
            return direct, f"variants.{field}"
    kernels = variant.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        return None, None
    values: list[float] = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            continue
        if kernel.get("region") is False:
            continue
        occ = _as_float(kernel.get("occupancy"))
        if occ is None:
            blocks = _as_float(kernel.get("blocks_per_sm"))
            warps = _as_float(kernel.get("warps_per_block"))
            maximum = _as_float(kernel.get("sm_max_warps"))
            if blocks is not None and warps is not None and maximum and maximum > 0:
                occ = min(1.0, blocks * warps / maximum)
        if occ is None or not 0 <= occ <= 1:
            return None, None
        values.append(occ)
    if not values:
        return None, None
    return min(values), "min(kernels[].occupancy)"


def provenance_for_case(audit: AuditResult, case_id: str, mode: str) -> dict[str, Any]:
    info = audit["case_modes"].get((case_id, mode))
    if not info:
        return {}
    pairs = []
    for pair in info["pairs"]:
        split_variant = audit["variants"].get((case_id, "split"))
        fused_variant = audit["variants"].get((case_id, "fused"))
        split_check = audit["checks"].get((case_id, "split", mode), [])
        fused_check = audit["checks"].get((case_id, "fused", mode), [])
        pairs.append(
            {
                "block": pair["block"],
                "attempt": pair["attempt"],
                "n": pair["n"],
                "iteration_order": pair["iteration_order"],
                "split": {
                    "file": "timings.jsonl",
                    "line": pair["split_line"],
                    "elapsed_ms": _raw_timing_elapsed(audit, case_id, mode, pair["block"], pair["attempt"], "split"),
                    "n": pair["n"],
                    "us_per_call": pair["split_us"],
                    "artifact_hash": normalize_hash(split_variant.get("artifact_hash")) if split_variant else None,
                    "artifact_path": (split_variant.get("output_code") or split_variant.get("artifact_path")) if split_variant else None,
                },
                "fused": {
                    "file": "timings.jsonl",
                    "line": pair["fused_line"],
                    "elapsed_ms": _raw_timing_elapsed(audit, case_id, mode, pair["block"], pair["attempt"], "fused"),
                    "n": pair["n"],
                    "us_per_call": pair["fused_us"],
                    "artifact_hash": normalize_hash(fused_variant.get("artifact_hash")) if fused_variant else None,
                    "artifact_path": (fused_variant.get("output_code") or fused_variant.get("artifact_path")) if fused_variant else None,
                },
                "checks": {
                    "split": [{"line": source_of(check)[1], "input_id": check.get("input_id"), "pass": check.get("_pass")} for check in split_check],
                    "fused": [{"line": source_of(check)[1], "input_id": check.get("input_id"), "pass": check.get("_pass")} for check in fused_check],
                },
            }
        )
    return {
        "case_id": case_id,
        "mode": mode,
        "T_split_us": info.get("T_split_us_mean"),
        "T_fused_us": info.get("T_fused_us_mean"),
        "n_pairs": len(info["pairs"]),
        "selected_attempts": {str(key): value for key, value in info.get("selected_attempts", {}).items()},
        "pairs": pairs,
    }


def _raw_timing_elapsed(
    audit: AuditResult,
    case_id: str,
    mode: str,
    block: int,
    attempt: int,
    strategy: str,
) -> float | None:
    row = audit["timings"].get((case_id, mode, block, attempt, strategy))
    return _as_float(row.get("elapsed_ms")) if row else None


def public_run(audit: AuditResult) -> dict[str, Any]:
    return strip_private(audit["run"])
