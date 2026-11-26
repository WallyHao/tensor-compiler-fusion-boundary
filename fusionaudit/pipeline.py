"""Build all audit tables, result tables, figures and the freeze package."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from . import config
from .analysis import analyze, case_trace, public_results
from .audit import audit_run, public_run
from .figures import render_all
from .io import sha256_file, write_csv, write_json, write_jsonl, write_text
from .report import render_report

AUDIT_ISSUE_COLUMNS = [
    "severity",
    "code",
    "message",
    "case_id",
    "mode",
    "block",
    "attempt",
    "strategy",
    "file",
    "line",
    "detail",
]

BLOCK_COLUMNS = [
    "case_id",
    "mode",
    "block",
    "attempt",
    "n",
    "iteration_order",
    "split_line",
    "fused_line",
    "status",
    "reason",
]

INVENTORY_COLUMNS = [
    "case_id",
    "template",
    "dtype",
    "shape",
    "planned_status",
    "exclusion_reason",
    "variants",
    "checks_split",
    "checks_fused",
    "timing_rows",
    "qualified_direct",
    "qualified_replay",
    "complete_direct",
    "complete_replay",
    "inventory_status",
]

RESULT_COLUMNS = [
    "run_id",
    "case_id",
    "mode",
    "template",
    "group",
    "fold",
    "shape",
    "dtype",
    "layout",
    "edge",
    "n_pairs",
    "expected_blocks",
    "T_split_us",
    "T_fused_us",
    "B_us",
    "relative_benefit",
    "relative_ci95",
    "label",
    "labels_by_threshold",
    "default_decision",
    "selected_attempts",
]

FEATURE_COLUMNS = [
    "case_id",
    "mode",
    "template",
    "dtype",
    "shape",
    "logical_saved_bytes",
    "x1_us",
    "x2_us",
    "x3_us",
]

DECISION_COLUMNS = [
    "fold",
    "method",
    "mode",
    "case_id",
    "template",
    "dtype",
    "shape",
    "actual_label",
    "predicted",
    "coverage_basis",
    "reason",
    "predicted_B_us",
    "T_split_us",
    "T_fused_us",
    "T_chosen_us",
    "regret_us",
    "regret_relative",
    "default_covered",
]


def build_run(
    run_dir: Path,
    out_dir: Path,
    *,
    expected_blocks: int | None = None,
    iterations: int = config.BOOTSTRAP_ITERATIONS,
    seed: int = config.BOOTSTRAP_SEED,
    allow_partial: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    if out_dir == run_dir:
        raise ValueError("output directory must differ from the raw run directory; derived tables must not overwrite raw data")
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_run(run_dir, expected_blocks=expected_blocks)
    analysis = analyze(audit, iterations=iterations, seed=seed, allow_partial=allow_partial)
    figures = render_all(out_dir, analysis, audit)

    _write_audit(out_dir, audit)
    results = public_results(analysis["results"])
    _write_results(out_dir, results)
    _write_features(out_dir, analysis["feature_rows"])
    decisions = _all_decisions(analysis)
    _write_decisions(out_dir, decisions)
    models = _models_payload(audit, analysis)
    write_json(out_dir / "models.json", models)
    write_json(out_dir / "representatives.json", analysis["representatives"])
    write_json(out_dir / "diagnostics.json", analysis["diagnostics"])
    provenance: dict[str, dict[str, Any]] = {}
    for result in analysis["results"]:
        provenance.setdefault(result["case_id"], {})[result["mode"]] = case_trace(audit, result["case_id"], result["mode"])
    write_json(out_dir / "provenance.json", provenance)

    report = render_report(audit, analysis, figures, out_dir)
    write_text(out_dir / "report.md", report)

    _write_freeze(out_dir, audit, analysis, models, figures)

    return {
        "run_id": audit["run"].get("run_id"),
        "out_dir": str(out_dir),
        "report": str(out_dir / "report.md"),
        "results": len(results),
        "errors": audit["summary"]["error_count"],
        "warnings": audit["summary"]["warning_count"],
        "figures": figures,
    }


def _write_audit(out_dir: Path, audit: dict[str, Any]) -> None:
    payload = {
        "run_id": audit["run"].get("run_id"),
        "synthetic": audit["synthetic"],
        "run": public_run(audit),
        "expected_blocks": audit["expected_blocks"],
        "summary": audit["summary"],
        "funnel": audit["funnel"],
        "calibration": audit["calibration"],
        "evidence_hashes": audit["evidence_hashes"],
        "input_fold_check": audit["input_fold_leak"],
        "input_hashes": audit["files"],
        "issues": audit["issues"],
        "block_audit": audit["block_audit"],
        "case_inventory": audit["case_inventory"],
    }
    write_json(out_dir / "audit.json", payload)
    write_csv(out_dir / "audit_issues.csv", audit["issues"], AUDIT_ISSUE_COLUMNS)
    write_csv(out_dir / "block_audit.csv", audit["block_audit"], BLOCK_COLUMNS)
    write_csv(out_dir / "case_inventory.csv", audit["case_inventory"], INVENTORY_COLUMNS)


def _write_results(out_dir: Path, results: list[dict[str, Any]]) -> None:
    write_jsonl(out_dir / "results.jsonl", results)
    write_csv(out_dir / "results.csv", results, RESULT_COLUMNS)


def _write_features(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(out_dir / "features.jsonl", rows)
    write_csv(out_dir / "features.csv", rows, FEATURE_COLUMNS)


def _write_decisions(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(out_dir / "decisions.jsonl", rows)
    write_csv(out_dir / "decisions.csv", rows, DECISION_COLUMNS)


def _all_decisions(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("direct", "replay"):
        rows.extend(analysis["evaluations"].get(mode, {}).get("decisions", []))
    return rows


def _models_payload(audit: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    evaluations = {}
    for mode, evaluation in analysis["evaluations"].items():
        evaluations[mode] = {
            "mode": mode,
            "n_cases": evaluation["n_cases"],
            "label_counts": evaluation["label_counts"],
            "folds": evaluation["folds"],
            "methods": evaluation["methods"],
            "matched_methods": evaluation["matched_methods"],
            "unavailable_reason": evaluation.get("unavailable_reason"),
        }
    return {
        "run_id": audit["run"].get("run_id"),
        "synthetic": audit["synthetic"],
        "analysis_version": config.ANALYSIS_VERSION,
        "feature_whitelist": list(config.FEATURE_WHITELIST),
        "forbidden_model_fields": list(config.FORBIDDEN_MODEL_FIELDS),
        "feature_units": config.FEATURE_UNITS,
        "practical_band": config.PRACTICAL_BAND,
        "sensitivity_bands": list(config.SENSITIVITY_BANDS),
        "bootstrap": {"iterations": config.BOOTSTRAP_ITERATIONS, "seed": config.BOOTSTRAP_SEED, "unit": "block_pair"},
        "fold_rule": "leave-one-template-out; all shapes and dtypes follow the template",
        "evaluations": evaluations,
        "full_fits": analysis["full_fits"],
        "representatives": analysis.get("representatives"),
        "calibration": audit["calibration"],
    }


def _write_freeze(
    out_dir: Path,
    audit: dict[str, Any],
    analysis: dict[str, Any],
    models: dict[str, Any],
    figures: list[str],
) -> None:
    freeze = out_dir / "freeze"
    freeze.mkdir(parents=True, exist_ok=True)
    write_json(freeze / "models.json", models)
    write_json(
        freeze / "features.json",
        {
            "feature_whitelist": list(config.FEATURE_WHITELIST),
            "forbidden_model_fields": list(config.FORBIDDEN_MODEL_FIELDS),
            "units": config.FEATURE_UNITS,
            "rows": analysis["feature_rows"],
            "calibration": audit["calibration"],
        },
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for case_id, case in sorted(audit["cases"].items()):
        groups[str(case.get("template"))].append(case_id)
    folds = {}
    for mode, evaluation in analysis["evaluations"].items():
        folds[mode] = [
            {
                "fold": fold["fold"],
                "test_template": fold["test_template"],
                "train_templates": fold["train_templates"],
                "test_templates": fold["test_templates"],
                "train_case_ids": fold["train_case_ids"],
                "test_case_ids": fold["test_case_ids"],
            }
            for fold in evaluation["folds"]
        ]

    outputs = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(out_dir))
        if relative.startswith("freeze/"):
            continue
        outputs[relative] = sha256_file(path)

    code_hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code_hashes[path.name] = sha256_file(path)

    manifest = {
        "analysis_version": config.ANALYSIS_VERSION,
        "run_id": audit["run"].get("run_id"),
        "synthetic": audit["synthetic"],
        "purpose": audit["run"].get("purpose"),
        "command_to_rebuild": f"python -m fusionaudit build --run {audit['run_dir']} --out {out_dir}",
        "input_hashes": audit["files"],
        "evidence_hashes": audit["evidence_hashes"],
        "analysis_code_hashes": code_hashes,
        "config": {
            "expected_blocks": audit["expected_blocks"],
            "bootstrap_iterations": config.BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": config.BOOTSTRAP_SEED,
            "practical_band": config.PRACTICAL_BAND,
            "sensitivity_bands": list(config.SENSITIVITY_BANDS),
            "feature_whitelist": list(config.FEATURE_WHITELIST),
            "forbidden_model_fields": list(config.FORBIDDEN_MODEL_FIELDS),
            "fold_rule": "leave-one-template-out",
            "label_rule": "relative CI vs band: fuse/split/equivalent/uncertain",
            "regret": "T_chosen - min(T_split, T_fused), us/call",
        },
        "groups": dict(sorted(groups.items())),
        "folds": folds,
        "figures": figures,
        "outputs": outputs,
        "cross_device_status": "not executed; optional extension",
    }
    write_json(freeze / "manifest.json", manifest)
    write_text(freeze / "README.md", _freeze_readme(audit, out_dir))
    _write_sums(out_dir)


def _write_sums(out_dir: Path) -> None:
    lines = []
    sums_path = out_dir / "freeze" / "sha256sums.txt"
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path == sums_path:
            continue
        digest = sha256_file(path)
        if digest:
            lines.append(f"{digest}  {path.relative_to(out_dir)}")
    write_text(sums_path, "\n".join(lines) + "\n")


def _freeze_readme(audit: dict[str, Any], out_dir: Path) -> str:
    synthetic = "**SYNTHETIC — 不可用于实验结论。**" if audit["synthetic"] else "单卡正式数据冻结包。"
    return f"""# Freeze package

{synthetic}

- run_id: `{audit['run'].get('run_id')}`
- analysis version: `{config.ANALYSIS_VERSION}`
- rebuild: `python -m fusionaudit build --run {audit['run_dir']} --out {out_dir}`
- Cross-device verification: **not executed**; the package only fixes the corpus, feature whitelist, fold rule, thresholds, models and metrics for a later optional second device.

## Frozen decisions

- Four-way label: relative-benefit 95% bootstrap interval against ±2%; sensitivity ±1% and ±5%.
- Bootstrap: 10,000 paired-block resamples, seed 2026.
- Folds: leave-one-template-out; every shape, dtype and repeat follows its template.
- Models: no-intercept OLS on `x1_us`, `x2_us` (two-term) and `x1_us`, `x2_us`, `x3_us` (three-term, x3-complete subset only).
- Byte rule: `logical_saved_bytes > tau` => fused; tau selected on the training fold, ties choose the larger tau.
- Regret: `T_chosen - min(T_split, T_fused)` in µs/call; coverage is always reported with the loss.

## Forbidden inputs

Only `x1_us`, `x2_us`, `x3_us` enter a model. Fused timing, the measured difference, post-run counters, measured DRAM traffic, achieved occupancy and the best measured kernel are labels or diagnostics. See `features.json`.

## Pending cross-device checks

1. Recalibrate `L` and `BW` on the second device; keep the same submission path.
2. Recompile the same candidate set and recompute `x1` and `x3` from that device's compile metadata.
3. Measure split only, write a timestamped prediction table, then unseal fused labels.
4. Report per-template degradation, coverage differences and any software-version change separately.
"""


def build_all(runs_root: Path, out_root: Path, **kwargs: Any) -> list[dict[str, Any]]:
    results = []
    for run_dir in sorted(runs_root.iterdir()):
        if not (run_dir / "run.json").is_file():
            continue
        results.append(build_run(run_dir, out_root / run_dir.name, **kwargs))
    return results


def load_and_trace(run_dir: Path, case_id: str, mode: str) -> dict[str, Any]:
    audit = audit_run(run_dir)
    return case_trace(audit, case_id, mode)
