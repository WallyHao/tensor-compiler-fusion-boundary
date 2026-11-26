"""End-to-end self-test on the marked synthetic fixture."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .fixtures import RUN_ID, write_collection_schema_copy, write_fixture_run
from .pipeline import build_run
from .stats import fit_linear, predict_linear


def run_selftest(*, workdir: Path | None = None, keep: bool = False) -> int:
    checks: list[str] = []
    if workdir is None:
        temporary = tempfile.TemporaryDirectory(prefix="fusionaudit-selftest-")
        root = Path(temporary.name)
    else:
        temporary = None
        root = workdir
        root.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = write_fixture_run(root / "runs", force=True)
        out_dir = root / "out" / RUN_ID
        summary = build_run(run_dir, out_dir)
        _assert_outputs(out_dir)
        checks.append("output files exist")

        audit = _load_json(out_dir / "audit.json")
        results = _load_jsonl(out_dir / "results.jsonl")
        features = _load_jsonl(out_dir / "features.jsonl")
        decisions = _load_jsonl(out_dir / "decisions.jsonl")
        models = _load_json(out_dir / "models.json")
        report = (out_dir / "report.md").read_text(encoding="utf-8")
        provenance = _load_json(out_dir / "provenance.json")

        by_case = {(row["case_id"], row["mode"]): row for row in results}

        # 1. one half of a pair is missing: it must be visible and must not
        #    silently enter the result table.
        codes = audit["summary"]["code_counts"]
        assert codes.get("pair_missing_half", 0) >= 1, "missing-half block was not reported"
        assert ("P1-n65536-f16-contig-t", "direct") not in by_case, "incomplete missing-half case entered results"
        missing_rows = [row for row in audit["block_audit"] if row["case_id"] == "P1-n65536-f16-contig-t" and row["status"] == "no_qualified_attempt"]
        assert missing_rows, "missing-half block has no no_qualified_attempt audit row"
        checks.append("missing half detected and excluded")

        # 2. first complete qualified attempt, whole-pair retest.
        retry = by_case[("P2-n16384-f32-contig-t", "direct")]
        assert retry["selected_attempts"] == {"0": 2, "1": 2, "2": 1, "3": 1}, f"wrong attempts: {retry['selected_attempts']}"
        assert retry["n_pairs"] == 4
        retry_audit = [
            row for row in audit["block_audit"] if row["case_id"] == "P2-n16384-f32-contig-t" and row["mode"] == "direct" and row["block"] in (0, 1)
        ]
        assert any(row["attempt"] == 1 and row["status"] == "rejected" for row in retry_audit)
        assert any(row["attempt"] == 2 and row["status"] == "selected" for row in retry_audit)
        checks.append("first qualified retest attempt selected")

        # 3. near-zero benefit inside +/-2%.
        near_zero = by_case[("R1-m64-k512-f32-contig-t", "direct")]
        assert near_zero["label"] == "equivalent", f"near-zero label={near_zero['label']}"
        lo, hi = near_zero["relative_ci95"]
        assert lo is not None and hi is not None and -0.02 <= lo <= hi <= 0.02
        assert by_case[("P1-n65536-f32-contig-t", "direct")]["label"] == "fuse"
        assert by_case[("P2-n131072-f16-contig-t", "direct")]["label"] == "split"
        checks.append("equivalent/fuse/split classification")

        # 4. supplied leaky folds are rejected; rebuilt folds are template-disjoint.
        assert audit["input_fold_check"]["leaks"], "input template leak not detected"
        assert codes.get("template_leak", 0) >= 1
        for mode, evaluation in models["evaluations"].items():
            for fold in evaluation["folds"]:
                assert not (set(fold["train_templates"]) & set(fold["test_templates"])), f"leak in generated fold {fold}"
                assert len(fold["test_templates"]) == 1
        all_templates = {row["template"] for row in results}
        tested = {fold["test_template"] for fold in models["evaluations"]["direct"]["folds"]}
        assert tested == all_templates
        checks.append("template leak rejected; leave-one-template-out folds disjoint")

        # Every complete case has the three whitelisted features and no model
        # row carries fused timing.
        assert features, "no feature rows"
        for row in features:
            for field in ("x1_us", "x2_us", "x3_us"):
                assert row.get(field) is not None, f"{row['case_id']} lacks {field}"
            assert "T_fused_us" not in row
        checks.append("feature whitelist populated without fused timing")

        # Coverage and regret are reported for constants, default, threshold and models.
        methods = models["evaluations"]["direct"]["methods"]
        for method in ("always_fused", "always_split", "default_inductor", "byte_threshold", "model_2term", "model_3term"):
            assert method in methods, f"missing method {method}"
        assert methods["default_inductor"]["coverage"] < 1.0, "default coverage fixture did not exercise missing mapping"
        assert methods["byte_threshold"]["mean_regret_us"] is not None
        assert decisions
        checks.append("coverage and regret reported for all decision methods")

        # Trace one report number back to raw block lines and artifacts.
        direct_results = [row for row in results if row["mode"] == "direct"]
        preferred = [row for row in direct_results if row["label"] == "fuse"]
        example = max(preferred or direct_results, key=lambda row: abs(row.get("B_us") or 0.0))
        example_id = example["case_id"]
        trace = provenance[example_id]["direct"]
        assert example_id in report
        assert trace["pairs"], "no provenance pairs"
        for pair in trace["pairs"]:
            assert pair["split"]["line"] and pair["fused"]["line"]
            assert pair["split"]["artifact_hash"] and pair["fused"]["artifact_hash"]
        recomputed_split = sum(pair["split"]["us_per_call"] for pair in trace["pairs"]) / len(trace["pairs"])
        recomputed_fused = sum(pair["fused"]["us_per_call"] for pair in trace["pairs"]) / len(trace["pairs"])
        result = by_case[(example_id, "direct")]
        assert math.isclose(recomputed_split, result["T_split_us"], rel_tol=1e-12)
        assert math.isclose(recomputed_fused, result["T_fused_us"], rel_tol=1e-12)
        assert "timings.jsonl" in report and result["T_split_us"] != result["T_fused_us"]
        assert any(pair["split"]["artifact_hash"] in report for pair in trace["pairs"])
        checks.append("report number traced to raw block lines and artifacts")

        assert summary["results"] >= 8
        assert (out_dir / "freeze" / "sha256sums.txt").is_file()
        checks.append("freeze package written")

        # 5. the collection harness spellings must read without rewriting raw data.
        compat_run = write_collection_schema_copy(run_dir, root / "runs" / "synthetic-real-schema")
        compat_out = root / "out" / "synthetic-real-schema"
        build_run(compat_run, compat_out)
        compat_audit = _load_json(compat_out / "audit.json")
        compat_results = _load_jsonl(compat_out / "results.jsonl")
        compat_models = _load_json(compat_out / "models.json")
        compat_issues = compat_audit["summary"]["code_counts"]
        for forbidden in ("invalid_check", "invalid_timing", "artifact_hash_mismatch", "missing_artifact_hash", "missing_run_field"):
            assert forbidden not in compat_issues, f"compat fixture produced {forbidden}: {compat_issues}"
        assert len(compat_results) == len(results)
        assert compat_audit["run"]["started_at"] == "2026-11-25T00:00:00Z"
        assert compat_audit["calibration"]["modes"]["direct"]["launch_source"] == "calibration.jsonl"
        assert math.isclose(compat_audit["calibration"]["modes"]["direct"]["launch_us_per_kernel"], 4.0, rel_tol=1e-9)
        assert math.isclose(compat_audit["calibration"]["modes"]["direct"]["bandwidth_bytes_per_us"], 500.0, rel_tol=1e-9)
        shapes = [row["shape"] for row in compat_results]
        assert any("N" in shape for shape in shapes) and any("M" in shape and "K" in shape for shape in shapes)
        assert compat_models["evaluations"]["direct"]["methods"]["default_inductor"]["coverage"] < 1.0
        checks.append("collection-schema aliases read without rewriting raw data")

        # 6. rank-deficient OLS reports non-identifiability but still predicts.
        rank_rows = [
            {"x1_us": float(value), "x2_us": float(value), "B_us": 2.0 * value}
            for value in (1.0, 2.0, 3.0, 4.0)
        ]
        rank_model = fit_linear(rank_rows, ("x1_us", "x2_us"), "B_us")
        assert rank_model["rank"] == 1 and not rank_model["identifiable"]
        assert rank_model["condition_number"] is None
        assert predict_linear(rank_model, {"x1_us": 5.0, "x2_us": 5.0}) is not None
        checks.append("rank-deficient model reports non-identifiability")

        print("SELFTEST PASS")
        for check in checks:
            print(f"  ok  {check}")
        print(f"  run_dir: {run_dir}")
        print(f"  out_dir: {out_dir}")
        return 0
    except AssertionError as exc:
        print(f"SELFTEST FAIL: {exc}")
        raise
    finally:
        if temporary is not None and not keep:
            temporary.cleanup()


def _assert_outputs(out_dir: Path) -> None:
    required = [
        "audit.json",
        "audit_issues.csv",
        "block_audit.csv",
        "case_inventory.csv",
        "results.jsonl",
        "results.csv",
        "features.jsonl",
        "decisions.jsonl",
        "models.json",
        "representatives.json",
        "diagnostics.json",
        "provenance.json",
        "report.md",
        "freeze/manifest.json",
        "freeze/models.json",
        "freeze/features.json",
        "freeze/sha256sums.txt",
    ]
    for name in required:
        assert (out_dir / name).is_file(), f"missing output {name}"
    assert list((out_dir / "figures").glob("*.svg")), "no figures"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
