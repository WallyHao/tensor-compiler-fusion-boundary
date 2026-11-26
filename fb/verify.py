"""Independent verification of a run directory against the data contract."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .rundir import RunContext
from .util import (
    atomic_write_json,
    linear_fit,
    median,
    read_json,
    read_jsonl,
    utcnow,
)


def _load(run_dir: Path, name: str) -> list[dict]:
    p = run_dir / f"{name}.jsonl"
    return read_jsonl(p) if p.exists() else []


def verify_run(run_dir: str | Path, write: bool = True,
               check_sums: bool = True) -> dict:
    run_dir = Path(run_dir)
    run = read_json(run_dir / "run.json")
    proto = run["protocol"]
    timings = _load(run_dir, "timings")
    variants = _load(run_dir, "variants")
    cases = _load(run_dir, "cases")
    checks = _load(run_dir, "checks")
    calibration = _load(run_dir, "calibration")

    problems: list[str] = []
    warnings: list[str] = []

    # global uniqueness
    seen: dict[tuple, dict] = {}
    for r in timings:
        key = (r["case_id"], r["mode"], r["block"], r["attempt"], r["strategy"])
        if key in seen:
            problems.append(f"duplicate timings key {key}")
        seen[key] = r

    variant_hashes: dict[tuple[str, str], set[str]] = {}
    for v in variants:
        accepted = {
            v.get("artifact_sha256"), v.get("artifact_hash"),
            v.get("output_code_sha256"),
        } - {None}
        variant_hashes.setdefault((v["case_id"], v["strategy"]), set()).update(accepted)
    case_ids = {c["case_id"] for c in cases}
    timing_cases = {r["case_id"] for r in timings}
    for cid in sorted(timing_cases - case_ids):
        problems.append(f"timings reference case missing from cases.jsonl: {cid}")

    per: dict[tuple[str, str], dict] = {}
    for (cid, mode, block, attempt, strategy), r in sorted(
        seen.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4])
    ):
        g = per.setdefault((cid, mode), {
            "case_id": cid,
            "mode": mode,
            "blocks": {},
            "n_values": set(),
            "attempts": 0,
            "errors": 0,
            "contaminated": 0,
            "pair_incomplete": 0,
            "patterns": {"SF": 0, "FS": 0},
            "reasons": {},
        })
        g["blocks"].setdefault(block, {}).setdefault(attempt, {})[strategy] = r
        g["attempts"] = max(g["attempts"], attempt)
        if r.get("quality") == "error":
            g["errors"] += 1
        elif r.get("quality") == "contaminated":
            g["contaminated"] += 1
        elif r.get("quality") == "pair_incomplete":
            g["pair_incomplete"] += 1
        if r.get("reason"):
            g["reasons"][r["reason"]] = g["reasons"].get(r["reason"], 0) + 1
        # Recompute the derived unit from the raw event time.
        if r.get("elapsed_ms") is not None and r.get("n"):
            expect = r["elapsed_ms"] * 1000.0 / r["n"]
            got = r.get("us_per_call")
            if got is None or not math.isclose(expect, got, rel_tol=1e-9, abs_tol=1e-9):
                problems.append(
                    f"{cid}/{mode} b{block} a{attempt} {strategy}: us_per_call mismatch "
                    f"{got} != {expect}"
                )
            if not (r["elapsed_ms"] > 0):
                problems.append(f"{cid}/{mode} b{block} a{attempt} {strategy}: elapsed_ms <= 0")
            if not (r["n"] > 0):
                problems.append(f"{cid}/{mode} b{block} a{attempt} {strategy}: n <= 0")
        if r.get("gpu_uuid") is None:
            warnings.append(f"{cid}/{mode}: missing gpu_uuid")
        if r.get("launch_verified") is False:
            problems.append(
                f"{cid}/{mode} b{block} a{attempt} {strategy}: observed kernels do not "
                f"match the artifact"
            )
        elif r.get("launch_verified") is None and not run.get("synthetic"):
            warnings.append(
                f"{cid}/{mode}: no launch verification recorded (reference/unknown candidate)"
            )

    expected_blocks = int(proto["blocks"])
    for c in cases:
        for mode in proto["modes"]:
            key = (c["case_id"], mode)
            if key not in per:
                per[key] = {
                    "case_id": c["case_id"],
                    "mode": mode,
                    "blocks": {},
                    "n_values": set(),
                    "attempts": 0,
                    "errors": 0,
                    "contaminated": 0,
                    "pair_incomplete": 0,
                    "patterns": {"SF": 0, "FS": 0},
                    "reasons": {},
                }
                warnings.append(f"{c['case_id']}/{mode}: no timing rows")
    per_rows: list[dict] = []
    total_valid = total_expected = 0
    for (cid, mode), g in sorted(per.items()):
        canonical: dict[int, int] = {}
        for block, attempts in g["blocks"].items():
            for a in sorted(attempts):
                pair = {k: v for k, v in attempts[a].items()
                        if k in ("split", "fused")}
                if len(pair) != 2 or set(pair) != {"split", "fused"}:
                    continue
                rows = list(pair.values())
                if not all(r.get("quality") == "ok" for r in rows):
                    continue
                ns = {r["n"] for r in rows}
                if len(ns) != 1:
                    problems.append(f"{cid}/{mode} b{block} a{a}: pair n mismatch {ns}")
                    continue
                for r in rows:
                    accepted = variant_hashes.get((cid, r["strategy"]), set())
                    if accepted and r.get("artifact_hash") not in accepted:
                        problems.append(
                            f"{cid}/{mode} b{block} a{a} {r['strategy']}: artifact hash mismatch"
                        )
                canonical[block] = a
                g["n_values"].add(rows[0]["n"])
        for b in sorted(canonical):
            pat = g["blocks"][b][canonical[b]]["split"].get("block_pattern")
            if pat in g["patterns"]:
                g["patterns"][pat] += 1
        if len(g["n_values"]) > 1:
            problems.append(f"{cid}/{mode}: multiple n across blocks {sorted(g['n_values'])}")
        distinct_blocks = len(canonical)
        failed_blocks = sorted(
            b for b in g["blocks"] if b not in canonical
        )
        total_valid += distinct_blocks
        total_expected += expected_blocks
        status = "valid" if distinct_blocks == expected_blocks else "incomplete"
        row = {
            "case_id": cid,
            "mode": mode,
            "expected_blocks": expected_blocks,
            "valid_pairs": distinct_blocks,
            "n": sorted(g["n_values"])[0] if g["n_values"] else None,
            "attempts_used": g["attempts"],
            "pattern_counts": g["patterns"],
            "failed_blocks": failed_blocks,
            "error_rows": g["errors"],
            "contaminated_rows": g["contaminated"],
            "pair_incomplete_rows": g["pair_incomplete"],
            "reasons": g["reasons"],
            "status": status,
        }
        per_rows.append(row)
        if distinct_blocks and distinct_blocks != expected_blocks:
            warnings.append(f"{cid}/{mode}: {distinct_blocks}/{expected_blocks} valid pairs")
        if g["patterns"] != {"SF": 0, "FS": 0} and g["patterns"]["SF"] != g["patterns"]["FS"]:
            warnings.append(
                f"{cid}/{mode}: unbalanced pattern counts {g['patterns']}"
            )

    # correctness gate
    bad_checks = [c for c in checks if c.get("pass") is False]
    if bad_checks:
        problems.append(f"{len(bad_checks)} correctness checks failed")
    per_check = {}
    for c in checks:
        key = (c["case_id"], c["strategy"])
        per_check.setdefault(key, True)
        per_check[key] = per_check[key] and c.get("pass") is not False

    # calibration recomputation
    calib_check = None
    fit_path = run_dir / "calibration_fit.json"
    if calibration and fit_path.exists():
        fit = read_json(fit_path)
        calib_check = {}
        for mode in ("direct", "replay"):
            blocks: dict[int, list[float]] = {}
            for r in calibration:
                if r.get("kind") == "launch" and r.get("mode") == mode:
                    segment = r.get("segment_us")
                    if segment is None and r.get("elapsed_ms") and r.get("repeats"):
                        segment = r["elapsed_ms"] * 1000.0 / r["repeats"]
                    if segment is not None:
                        blocks.setdefault(int(r["k"]), []).append(segment)
            if blocks:
                med = {k: median(v) for k, v in sorted(blocks.items())}
                refit = linear_fit(sorted(med), [med[k] for k in sorted(med)])
                stored = fit.get(f"launch_{mode}", {})
                calib_check[mode] = {
                    "refit_slope": refit["slope"],
                    "stored_slope": stored.get("slope"),
                    "delta": refit["slope"] - (stored.get("slope") or 0),
                    "points_segment_us": med,
                }
                if stored.get("slope") is not None and not math.isclose(
                    refit["slope"], stored["slope"], rel_tol=1e-9
                ):
                    problems.append(f"calibration {mode} slope does not recompute")
    bw = None
    if fit_path.exists():
        fit = read_json(fit_path)
        bw = fit.get("bandwidth")

    sums_path = run_dir / "SHA256SUMS"
    sums_verified = None
    if sums_path.exists() and check_sums:
        sums_verified = True
        for line in sums_path.read_text().splitlines():
            if not line.strip():
                continue
            digest, rel = line.split(maxsplit=1)
            p = run_dir / rel.strip()
            if not p.exists():
                problems.append(f"checksum target missing: {rel}")
                sums_verified = False
                continue
            from .util import sha256_file

            if sha256_file(p) != digest:
                problems.append(f"checksum mismatch: {rel}")
                sums_verified = False

    result = {
        "run_id": run["run_id"],
        "checked_utc": utcnow(),
        "protocol_hash": run.get("protocol_hash"),
        "synthetic": run.get("synthetic", False),
        "expected_pairs_per_case_mode": expected_blocks,
        "totals": {
            "case_mode_groups": len(per_rows),
            "valid_pairs": total_valid,
            "expected_pairs": total_expected,
            "error_rows": sum(r["error_rows"] for r in per_rows),
            "contaminated_rows": sum(r["contaminated_rows"] for r in per_rows),
            "pair_incomplete_rows": sum(r["pair_incomplete_rows"] for r in per_rows),
        },
        "per_case_mode": per_rows,
        "calibration_recompute": calib_check,
        "bandwidth": bw,
        "checksums_verified": sums_verified,
        "problems": problems,
        "warnings": warnings,
        "integrity_pass": not problems,
        "complete": total_valid == total_expected and total_expected > 0,
    }
    if write:
        atomic_write_json(run_dir / "verification.json", result)
        _write_report(run_dir / "verification.md", result)
    return result


def _write_report(path: Path, r: dict) -> None:
    lines = [
        f"# Run verification: {r['run_id']}",
        "",
        f"- synthetic: {r['synthetic']}",
        f"- integrity_pass: {r['integrity_pass']}",
        f"- complete: {r['complete']} ({r['totals']['valid_pairs']}/{r['totals']['expected_pairs']} pairs)",
        f"- errors: {r['totals']['error_rows']}, contaminated: {r['totals']['contaminated_rows']}, "
        f"pair_incomplete: {r['totals']['pair_incomplete_rows']}",
        "",
        "| case | mode | valid/expected | n | attempts | SF/FS | errors | contaminated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in r["per_case_mode"]:
        lines.append(
            f"| {row['case_id']} | {row['mode']} | {row['valid_pairs']}/{row['expected_blocks']} | "
            f"{row['n']} | {row['attempts_used']} | {row['pattern_counts']['SF']}/{row['pattern_counts']['FS']} | "
            f"{row['error_rows']} | {row['contaminated_rows']} |"
        )
    if r["problems"]:
        lines += ["", "## Problems", *[f"- {p}" for p in r["problems"]]]
    if r["warnings"]:
        lines += ["", "## Warnings", *[f"- {w}" for w in r["warnings"][:100]]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_verification(r: dict) -> None:
    t = r["totals"]
    print(
        f"{r['run_id']}: integrity={'PASS' if r['integrity_pass'] else 'FAIL'} "
        f"pairs={t['valid_pairs']}/{t['expected_pairs']} "
        f"errors={t['error_rows']} contaminated={t['contaminated_rows']}"
    )
    for row in r["per_case_mode"]:
        print(
            f"  {row['case_id']:34s} {row['mode']:7s} "
            f"{row['valid_pairs']:2d}/{row['expected_blocks']:<2d} n={row['n']} "
            f"attempts={row['attempts_used']} {row['status']}"
        )
    for p in r["problems"][:20]:
        print(f"  PROBLEM {p}")
