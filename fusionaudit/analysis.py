"""Paired benefit estimates, leave-one-template-out evaluation and freeze data."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Sequence

from . import config
from .audit import AuditResult, provenance_for_case
from .io import normalize_hash, source_of
from .stats import (
    aggregate_metrics,
    classify_interval,
    fit_byte_threshold,
    fit_linear,
    iqr,
    mean,
    median,
    paired_bootstrap,
    percentile,
    predict_linear,
    regret_us,
    threshold_labels,
)

METHODS = (
    "always_fused",
    "always_split",
    "default_inductor",
    "byte_threshold",
    "model_2term",
    "model_2term_common",
    "model_3term",
)


def analyze(
    audit: AuditResult,
    *,
    iterations: int = config.BOOTSTRAP_ITERATIONS,
    seed: int = config.BOOTSTRAP_SEED,
    allow_partial: bool = False,
) -> dict[str, Any]:
    run_id = str(audit["run"].get("run_id", "unknown"))
    results: list[dict[str, Any]] = []
    for (case_id, mode), info in sorted(audit["case_modes"].items()):
        case = audit["cases"].get(case_id)
        if case is None:
            continue
        status = str(case.get("planned_status", "")).lower()
        if status not in config.VALID_CASE_STATUS or case.get("exclusion_reason"):
            continue
        if not info["complete"] and not allow_partial:
            continue
        if not info["pairs"]:
            continue
        result = _build_result(audit, case, info, iterations=iterations, seed=seed)
        result["run_id"] = run_id
        results.append(result)

    feature_rows = [row for row in audit["feature_rows"] if row["mode"] in {result["mode"] for result in results}]
    # Features must remain free of labels; check that the pre-registered names
    # are the only model inputs that exist in this table.
    for row in feature_rows:
        for field in config.FEATURE_WHITELIST:
            if field not in row:
                raise AssertionError(f"feature row lacks whitelisted field {field}")
        for forbidden in config.FORBIDDEN_MODEL_FIELDS:
            if forbidden in row:
                raise AssertionError(f"feature row contains forbidden model field {forbidden}")

    evaluations = {}
    full_fits = {}
    for mode in ("direct", "replay"):
        mode_results = [result for result in results if result["mode"] == mode]
        if mode_results:
            evaluations[mode] = evaluate_mode(audit, mode_results, mode=mode)
            full_fits[mode] = full_fit(audit, mode_results, mode=mode)

    sign_flips = detect_sign_flips(results)
    representatives = _representative_cases(evaluations.get("direct", {}))
    diagnostics = summarize_diagnostics(audit)

    return {
        "run_id": run_id,
        "synthetic": audit["synthetic"],
        "results": results,
        "feature_rows": feature_rows,
        "evaluations": evaluations,
        "full_fits": full_fits,
        "sign_flips": sign_flips,
        "representatives": representatives,
        "diagnostics": diagnostics,
    }


def summarize_diagnostics(audit: AuditResult) -> dict[str, Any]:
    telemetry = audit.get("telemetry_rows") or []
    temperatures = [_num(row.get("temperature_c", row.get("temperature_gpu"))) for row in telemetry]
    powers = [_num(row.get("power_w", row.get("power_draw"))) for row in telemetry]
    clocks = [_num(row.get("clock_sm_mhz", row.get("clocks_sm"))) for row in telemetry]
    temperatures = [value for value in temperatures if value is not None]
    powers = [value for value in powers if value is not None]
    clocks = [value for value in clocks if value is not None]
    throttle = Counter(str(row.get("throttle")) for row in telemetry if row.get("throttle") is not None)
    external = sum(1 for row in telemetry if row.get("external_activity") is True)

    profiles: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    units: dict[tuple[str, str, str], Any] = {}
    sources: dict[tuple[str, str, str], Any] = {}
    for row in audit.get("profile_rows") or []:
        value = _num(row.get("value", row.get("sample", row.get("mean"))))
        if value is None:
            continue
        key = (str(row.get("case_id")), str(row.get("strategy")), str(row.get("metric")))
        profiles[key].append(value)
        units.setdefault(key, row.get("unit"))
        sources.setdefault(key, row.get("source"))
    profile_summary = [
        {
            "case_id": key[0],
            "strategy": key[1],
            "metric": key[2],
            "n": len(values),
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
            "unit": units.get(key),
            "source": sources.get(key),
        }
        for key, values in sorted(profiles.items())
    ]
    return {
        "note": "telemetry and profiler counters are diagnostics only; they never enter the predictive feature table",
        "telemetry": {
            "rows": len(telemetry),
            "temperature_c": {"min": min(temperatures), "max": max(temperatures)} if temperatures else None,
            "power_w": {"min": min(powers), "max": max(powers)} if powers else None,
            "clock_sm_mhz": {"min": min(clocks), "max": max(clocks)} if clocks else None,
            "throttle_states": dict(sorted(throttle.items())),
            "external_activity_rows": external,
        },
        "profiles": profile_summary,
    }


def _representative_cases(evaluation: dict[str, Any]) -> dict[str, Any]:
    decisions = evaluation.get("decisions") or []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if row.get("predicted") in {"split", "fused"} and row.get("regret_us") is not None:
            by_method[str(row["method"])].append(row)
    selected = {}
    for method in METHODS:
        rows = by_method.get(method, [])
        if not rows:
            continue
        worst = max(rows, key=lambda row: float(row["regret_us"]))
        selected[method] = {
            "case_id": worst["case_id"],
            "template": worst["template"],
            "dtype": worst["dtype"],
            "shape": worst["shape"],
            "predicted": worst["predicted"],
            "actual_label": worst["actual_label"],
            "T_split_us": worst["T_split_us"],
            "T_fused_us": worst["T_fused_us"],
            "T_chosen_us": worst["T_chosen_us"],
            "regret_us": worst["regret_us"],
            "regret_relative": worst["regret_relative"],
            "fold": worst.get("fold"),
        }
    return {
        "selection_rule": "post hoc display only: maximum measured regret per method on complete direct test cases; not used for fitting or threshold selection",
        "methods": selected,
    }


def _build_result(
    audit: AuditResult,
    case: dict[str, Any],
    info: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    pairs = info["pairs"]
    splits = [pair["split_us"] for pair in pairs]
    fused = [pair["fused_us"] for pair in pairs]
    losses = [split - fuse for split, fuse in zip(splits, fused)]
    boot = paired_bootstrap(losses, iterations=iterations, seed=seed)
    ratio_boot = paired_bootstrap_ratio(splits, fused, iterations=iterations, seed=seed)
    t_split_mean = mean(splits)
    t_fused_mean = mean(fused)
    b_mean = mean(losses)
    relative = (b_mean / t_split_mean) if t_split_mean else None
    thresholds = config.SENSITIVITY_BANDS
    labels = threshold_labels(ratio_boot["ci95"], thresholds)
    main_threshold = config.PRACTICAL_BAND
    main_label = labels.get(str(main_threshold), "uncertain")
    split_lo, split_hi = iqr(splits)
    fused_lo, fused_hi = iqr(fused)
    b_lo, b_hi = iqr(losses)
    default_decision = _default_decision(audit, str(case["case_id"]))
    result = {
        "run_id": audit["run"].get("run_id"),
        "case_id": case["case_id"],
        "mode": info["mode"],
        "template": case.get("template"),
        "group": case.get("group") or case.get("template"),
        "fold": f"test:{case.get('template')}",
        "dtype": case.get("dtype"),
        "layout": case.get("layout"),
        "edge": case.get("edge"),
        "shape": case.get("_shape", case.get("shape")),
        "t_shape": case.get("t_shape"),
        "t_dtype": case.get("t_dtype"),
        "logical_saved_bytes": case.get("logical_saved_bytes"),
        "n_pairs": len(pairs),
        "expected_blocks": info["expected_blocks"],
        "selected_attempts": {str(key): value for key, value in info["selected_attempts"].items()},
        "T_split_us": t_split_mean,
        "T_fused_us": t_fused_mean,
        "T_split_us_median": median(splits),
        "T_split_us_iqr": [split_lo, split_hi],
        "T_fused_us_median": median(fused),
        "T_fused_us_iqr": [fused_lo, fused_hi],
        "B_us": b_mean,
        "B_us_median": median(losses),
        "B_us_iqr": [b_lo, b_hi],
        "relative_benefit": relative,
        "B_ci95": boot["ci95"],
        "relative_ci95": ratio_boot["ci95"],
        "bootstrap": {
            "seed": seed,
            "iterations": iterations,
            "unit": "block_pair",
            "relative_denominator": "bootstrap mean T_split",
        },
        "label": main_label,
        "labels_by_threshold": labels,
        "thresholds": {str(thr): classify_interval(ratio_boot["ci95"][0], ratio_boot["ci95"][1], thr) for thr in thresholds},
        "default_decision": default_decision,
        "synthetic": audit["synthetic"],
    }
    result["_pairs"] = pairs
    result["_case"] = case
    return result


def paired_bootstrap_ratio(
    splits: Sequence[float],
    fused: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not splits or len(splits) != len(fused):
        return {"mean": None, "ci95": [None, None], "iterations": iterations, "seed": seed}
    rng = random.Random(seed)
    n = len(splits)
    ratios: list[float] = []
    gains: list[float] = []
    for _ in range(iterations):
        total_split = 0.0
        total_fused = 0.0
        for _ in range(n):
            index = rng.randrange(n)
            total_split += splits[index]
            total_fused += fused[index]
        b = (total_split - total_fused) / n
        gains.append(b)
        ratios.append(b / (total_split / n) if total_split else 0.0)
    return {
        "mean": mean(ratios),
        "ci95": [percentile(ratios, 2.5), percentile(ratios, 97.5)],
        "B_ci95": [percentile(gains, 2.5), percentile(gains, 97.5)],
        "iterations": iterations,
        "seed": seed,
    }


def _default_decision(audit: AuditResult, case_id: str) -> str | None:
    candidates = [audit["variants"].get((case_id, strategy)) for strategy in ("split", "fused", "default")]
    for variant in candidates:
        if not variant:
            continue
        for field in ("verified_default_decision", "default_maps_to", "default_equivalent"):
            decision = variant.get(field)
            if decision in {"split", "fused"}:
                return decision
        if variant.get("partition_equal_default_fused") is True:
            return "fused"
        if variant.get("partition_equal_default_split") is True:
            return "split"
    return None


def detect_sign_flips(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["mode"], str(result["template"]), str(result["dtype"]))].append(result)
    flips: list[dict[str, Any]] = []
    for (mode, template, dtype), rows in sorted(grouped.items()):
        candidates: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        if all("N" in (row.get("shape") or {}) for row in rows):
            ordered = sorted(rows, key=lambda row: _num((row.get("shape") or {}).get("N")) or 0.0)
            candidates.extend(zip(ordered, ordered[1:], ["N"] * max(0, len(ordered) - 1)))
        else:
            by_m: dict[float, list[dict[str, Any]]] = defaultdict(list)
            by_k: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                shape = row.get("shape") or {}
                m = _num(shape.get("M"))
                k = _num(shape.get("K"))
                if m is not None:
                    by_m[m].append(row)
                if k is not None:
                    by_k[k].append(row)
            for m, items in by_m.items():
                ordered = sorted(items, key=lambda row: _num((row.get("shape") or {}).get("K")) or 0.0)
                candidates.extend(zip(ordered, ordered[1:], ["K"] * max(0, len(ordered) - 1)))
            for k, items in by_k.items():
                ordered = sorted(items, key=lambda row: _num((row.get("shape") or {}).get("M")) or 0.0)
                candidates.extend(zip(ordered, ordered[1:], ["M"] * max(0, len(ordered) - 1)))
        seen: set[tuple[str, str]] = set()
        for left, right, axis in candidates:
            if {left["label"], right["label"]} != {"fuse", "split"}:
                continue
            key = (left["case_id"], right["case_id"])
            if key in seen:
                continue
            seen.add(key)
            flips.append(
                {
                    "mode": mode,
                    "template": template,
                    "dtype": dtype,
                    "axis": axis,
                    "left_case_id": left["case_id"],
                    "left_label": left["label"],
                    "left_relative_benefit": left["relative_benefit"],
                    "right_case_id": right["case_id"],
                    "right_label": right["label"],
                    "right_relative_benefit": right["relative_benefit"],
                }
            )
    return flips


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_mode(audit: AuditResult, results: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    rows = []
    for result in results:
        info = audit["case_modes"].get((result["case_id"], mode), {})
        features = info.get("features") or {}
        row = {
            "case_id": result["case_id"],
            "template": result["template"],
            "dtype": result["dtype"],
            "shape": result["shape"],
            "logical_saved_bytes": result["logical_saved_bytes"],
            "T_split_us": result["T_split_us"],
            "T_fused_us": result["T_fused_us"],
            "B_us": result["B_us"],
            "actual_label": result["label"],
            "default_decision": result.get("default_decision"),
            "x1_us": features.get("x1_us"),
            "x2_us": features.get("x2_us"),
            "x3_us": features.get("x3_us"),
        }
        rows.append(row)

    templates = sorted({str(row["template"]) for row in rows})
    folds: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    if len(templates) < 2:
        return {
            "mode": mode,
            "folds": [],
            "methods": {method: aggregate_metrics([], len(rows)) for method in METHODS},
            "matched_methods": {method: aggregate_metrics([], 0) for method in METHODS},
            "label_counts": dict(Counter(row["actual_label"] for row in rows)),
            "n_cases": len(rows),
            "unavailable_reason": "leave-one-template-out needs at least two templates",
        }

    for test_template in templates:
        train_rows = [row for row in rows if str(row["template"]) != test_template]
        test_rows = [row for row in rows if str(row["template"]) == test_template]
        train_templates = sorted({str(row["template"]) for row in train_rows})
        test_templates = [test_template]
        if set(train_templates) & set(test_templates):
            raise AssertionError("template leakage in generated folds")

        byte_fit = fit_byte_threshold(train_rows)
        model_2 = _fit_model(train_rows, ("x1_us", "x2_us"))
        common_train = [row for row in train_rows if all(row.get(name) is not None for name in ("x1_us", "x2_us", "x3_us"))]
        common_test = [row for row in test_rows if all(row.get(name) is not None for name in ("x1_us", "x2_us", "x3_us"))]
        model_2_common = _fit_model(common_train, ("x1_us", "x2_us"))
        model_3 = _fit_model(common_train, ("x1_us", "x2_us", "x3_us"))

        decisions_by_method: dict[str, list[dict[str, Any]]] = {}
        decisions_by_method["always_fused"] = [_decision(row, "fused", "constant", None) for row in test_rows]
        decisions_by_method["always_split"] = [_decision(row, "split", "constant", None) for row in test_rows]
        decisions_by_method["default_inductor"] = [
            _decision(
                row,
                row.get("default_decision"),
                "verified_default_decision",
                None if row.get("default_decision") in {"split", "fused"} else "no mappable default decision",
            )
            for row in test_rows
        ]
        decisions_by_method["byte_threshold"] = [
            _decision(
                row,
                ("fused" if float(row["logical_saved_bytes"]) > float(byte_fit["tau"]) else "split")
                if byte_fit.get("tau") is not None and row.get("logical_saved_bytes") is not None
                else None,
                "logical_saved_bytes > tau",
                None if byte_fit.get("tau") is not None and row.get("logical_saved_bytes") is not None else "missing bytes or threshold",
            )
            for row in test_rows
        ]
        decisions_by_method["model_2term"] = _model_decisions(test_rows, model_2, "x1_us,x2_us")
        decisions_by_method["model_2term_common"] = _model_decisions(
            test_rows, model_2_common, "x1_us,x2_us on x3-complete subset", common_ids={row["case_id"] for row in common_test}
        )
        decisions_by_method["model_3term"] = _model_decisions(
            test_rows, model_3, "x1_us,x2_us,x3_us", common_ids={row["case_id"] for row in common_test}
        )

        fold = {
            "fold": test_template,
            "test_template": test_template,
            "train_templates": train_templates,
            "test_templates": test_templates,
            "train_case_ids": [row["case_id"] for row in train_rows],
            "test_case_ids": [row["case_id"] for row in test_rows],
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "byte_threshold": byte_fit,
            "models": {
                "model_2term": model_2,
                "model_2term_common": model_2_common,
                "model_3term": model_3,
            },
            "methods": {},
            "matched_methods": {},
        }
        for method in METHODS:
            decisions = decisions_by_method[method]
            default_covered_ids = {
                row["case_id"] for row in decisions_by_method["default_inductor"] if row["predicted"] in {"split", "fused"}
            }
            for decision in decisions:
                decision["fold"] = test_template
                decision["method"] = method
                decision["mode"] = mode
                decision["run_id"] = audit["run"].get("run_id")
                decision["default_covered"] = decision["case_id"] in default_covered_ids
            fold["methods"][method] = aggregate_metrics(decisions, len(test_rows))
            matched = [row for row in decisions if row["default_covered"]]
            fold["matched_methods"][method] = aggregate_metrics(matched, len(default_covered_ids))
            all_decisions.extend(decisions)
        folds.append(fold)

    method_aggregate: dict[str, Any] = {}
    matched_aggregate: dict[str, Any] = {}
    for method in METHODS:
        method_decisions = [row for row in all_decisions if row["method"] == method]
        method_aggregate[method] = aggregate_metrics(method_decisions, sum(fold["n_test"] for fold in folds))
        matched_rows = [row for row in method_decisions if row["default_covered"]]
        # Denominator: test cases with a mappable default decision in each fold.
        matched_denominator = sum(
            1
            for row in all_decisions
            if row["method"] == "default_inductor" and row["default_covered"]
        )
        matched_aggregate[method] = aggregate_metrics(matched_rows, matched_denominator)

    return {
        "mode": mode,
        "folds": folds,
        "methods": method_aggregate,
        "matched_methods": matched_aggregate,
        "label_counts": dict(Counter(row["actual_label"] for row in rows)),
        "n_cases": len(rows),
        "decisions": all_decisions,
        "rows": rows,
        "unavailable_reason": None,
    }


def _fit_model(rows: Sequence[dict[str, Any]], features: Sequence[str]) -> dict[str, Any]:
    usable = [row for row in rows if all(row.get(name) is not None for name in features)]
    model = fit_linear(usable, features, "B_us")
    model["n_available"] = len(usable)
    model["n_train"] = len(rows)
    coefficients = model.get("coefficients") or {}
    if "x3_us" in features:
        beta = coefficients.get("x3_us")
        model["formula"] = "B_hat_us = a*x1_us + b*x2_us - c*x3_us"
        model["c_penalty_us"] = None if beta is None else -beta
    else:
        model["formula"] = "B_hat_us = a*x1_us + b*x2_us"
    return model


def _model_decisions(
    test_rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
    basis: str,
    *,
    common_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    decisions = []
    for row in test_rows:
        if common_ids is not None and row["case_id"] not in common_ids:
            decisions.append(_decision(row, None, basis, "x3/resource feature missing"))
            continue
        if model.get("n_available", model.get("n", 0)) == 0:
            decisions.append(_decision(row, None, basis, "no training rows with complete features"))
            continue
        prediction = predict_linear(model, row)
        if prediction is None:
            decisions.append(_decision(row, None, basis, "feature missing"))
            continue
        decisions.append(_decision(row, "fused" if prediction > 0 else "split", basis, None, predicted_B_us=prediction))
    return decisions


def _decision(
    row: dict[str, Any],
    predicted: str | None,
    basis: str,
    reason: str | None,
    *,
    predicted_B_us: float | None = None,
) -> dict[str, Any]:
    chosen = predicted if predicted in {"split", "fused"} else None
    if chosen is None:
        regret = None
        relative = None
        chosen_time = None
    else:
        regret = regret_us(float(row["T_split_us"]), float(row["T_fused_us"]), chosen)
        minimum = min(float(row["T_split_us"]), float(row["T_fused_us"]))
        relative = regret / minimum if minimum else None
        chosen_time = float(row["T_fused_us"]) if chosen == "fused" else float(row["T_split_us"])
    return {
        "case_id": row["case_id"],
        "template": row["template"],
        "dtype": row["dtype"],
        "shape": row["shape"],
        "actual_label": row["actual_label"],
        "T_split_us": row["T_split_us"],
        "T_fused_us": row["T_fused_us"],
        "predicted": chosen,
        "reason": reason,
        "coverage_basis": basis,
        "predicted_B_us": predicted_B_us,
        "T_chosen_us": chosen_time,
        "regret_us": regret,
        "regret_relative": relative,
        "logical_saved_bytes": row.get("logical_saved_bytes"),
        "x1_us": row.get("x1_us"),
        "x2_us": row.get("x2_us"),
        "x3_us": row.get("x3_us"),
    }


def full_fit(audit: AuditResult, results: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    rows = []
    for result in results:
        if result["mode"] != mode:
            continue
        info = audit["case_modes"].get((result["case_id"], mode), {})
        features = info.get("features") or {}
        rows.append(
            {
                "case_id": result["case_id"],
                "template": result["template"],
                "B_us": result["B_us"],
                "T_split_us": result["T_split_us"],
                "T_fused_us": result["T_fused_us"],
                "logical_saved_bytes": result["logical_saved_bytes"],
                "x1_us": features.get("x1_us"),
                "x2_us": features.get("x2_us"),
                "x3_us": features.get("x3_us"),
            }
        )
    common = [row for row in rows if all(row.get(name) is not None for name in ("x1_us", "x2_us", "x3_us"))]
    return {
        "mode": mode,
        "n_cases": len(rows),
        "model_2term": _fit_model(rows, ("x1_us", "x2_us")),
        "model_2term_common": _fit_model(common, ("x1_us", "x2_us")),
        "model_3term": _fit_model(common, ("x1_us", "x2_us", "x3_us")),
        "byte_threshold": fit_byte_threshold(rows),
        "feature_whitelist": list(config.FEATURE_WHITELIST),
        "forbidden_fields": list(config.FORBIDDEN_MODEL_FIELDS),
        "feature_units": config.FEATURE_UNITS,
    }


def case_trace(audit: AuditResult, case_id: str, mode: str) -> dict[str, Any]:
    provenance = provenance_for_case(audit, case_id, mode)
    info = audit["case_modes"].get((case_id, mode))
    if not info or not provenance:
        return {"error": f"no selected pairs for case '{case_id}' mode '{mode}'"}
    splits = [pair["split_us"] for pair in info["pairs"]]
    fused = [pair["fused_us"] for pair in info["pairs"]]
    losses = [split - fuse for split, fuse in zip(splits, fused)]
    return {
        "run_id": audit["run"].get("run_id"),
        "case_id": case_id,
        "mode": mode,
        "formula": {
            "us_per_call": "elapsed_ms * 1000 / n",
            "T_split_us": "mean(split us_per_call over selected blocks)",
            "T_fused_us": "mean(fused us_per_call over selected blocks)",
            "B_us": "mean(split us_per_call - fused us_per_call)",
        },
        "T_split_us": mean(splits),
        "T_fused_us": mean(fused),
        "B_us": mean(losses),
        "n_pairs": len(losses),
        "selected_attempts": provenance.get("selected_attempts", {}),
        "pairs": provenance["pairs"],
        "artifacts": {
            "split": _artifact(audit, case_id, "split"),
            "fused": _artifact(audit, case_id, "fused"),
        },
        "checks": provenance.get("checks"),
    }


def _artifact(audit: AuditResult, case_id: str, strategy: str) -> dict[str, Any]:
    variant = audit["variants"].get((case_id, strategy)) or {}
    return {
        "line": source_of(variant)[1],
        "path": variant.get("output_code") or variant.get("artifact_path"),
        "hash": normalize_hash(variant.get("artifact_hash")),
        "trace_path": variant.get("trace_path"),
        "trace_verdict": variant.get("trace_verdict"),
        "verified_default_decision": variant.get("verified_default_decision"),
    }


def public_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in result.items() if not key.startswith("_")} for result in results]
