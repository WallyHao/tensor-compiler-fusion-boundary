"""Freeze second-GPU predictions using split-only timings, then score paired data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

from fb.backend import CudaBackend
from fb.cli import configure_torch, gpu_lock
from fb.fusionboundary_provider import FusionBoundaryProvider
from fb.protocol import Protocol
from fb.runner import case_order
from fb.telemetry import TelemetrySampler, list_external_gpu_processes, our_process_ids
from fb.util import atomic_write_json, read_json, read_jsonl, sha256_file, utcnow
from fusionaudit.audit import compute_features, derive_calibration
from fusionaudit.stats import aggregate_metrics, mean, predict_linear, regret_us


MODELS = ("model_2term", "model_2term_common", "model_3term")
METHODS = (*MODELS, "byte_threshold", "default_inductor", "always_fused", "always_split")


def collect_mode(backend, variant, sampler, proto, rows, case_id, mode):
    backend.warmup(variant, proto.warmup_calls)
    until = time.perf_counter() + proto.stabilize_ms / 1000
    while time.perf_counter() < until:
        backend.call_n(variant, proto.warmup_calls)
    backend.sync()
    trial_us = backend.estimate_us(variant, proto.trial_iters)
    n = proto.choose_n(trial_us, trial_us)
    for _ in range(2):
        elapsed_ms = backend.time_batch(variant, n)
        if elapsed_ms >= 0.9 * proto.target_batch_ms:
            break
        n = min(proto.n_cap, math.ceil(n * proto.target_batch_ms * proto.n_margin / elapsed_ms))

    qualified = []
    pids = our_process_ids()
    for block in range(proto.blocks):
        for attempt in range(1, proto.max_attempts + 1):
            start_utc = utcnow()
            start = time.time()
            backend.warmup(variant, proto.untimed_calls)
            elapsed_ms = backend.time_batch(variant, n)
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise RuntimeError(f"invalid CUDA event timing: {elapsed_ms}")
            quality = sampler.assess(start, time.time(), pids)
            us = elapsed_ms * 1000 / n
            row = {
                "case_id": case_id, "mode": mode, "strategy": "split", "block": block,
                "attempt": attempt, "window_start_utc": start_utc, "window_end_utc": utcnow(),
                "n": n, "trial_us": trial_us, "elapsed_ms": elapsed_ms, "us_per_call": us,
                "artifact_hash": variant.meta["artifact_sha256"], **quality,
            }
            rows.write(json.dumps(row, allow_nan=False) + "\n")
            rows.flush()
            if quality["quality"] == "ok":
                qualified.append(us)
                break
            sampler.settle()
    return mean(qualified) if len(qualified) == proto.blocks else None


def freeze_predictions(args):
    source = read_json(args.models)
    if source["synthetic"] or source.get("run_id") is None:
        raise ValueError("a real, frozen source model is required")
    cal_fit = read_json(args.calibration / "calibration_fit.json")
    if not all(cal_fit[key]["valid"] for key in ("launch_direct", "launch_replay", "bandwidth")):
        raise ValueError("the target calibration did not pass")
    calibration = derive_calibration(read_jsonl(args.calibration / "calibration.jsonl"),
                                     read_json(args.calibration / "run.json"), [], args.calibration)
    proto = Protocol()
    provider = FusionBoundaryProvider(args.candidates)
    cases = {case["case_id"]: case for case in provider.catalog()}
    if args.case_list is not None:
        selected = args.case_list.read_text(encoding="utf-8").split()
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("case list must contain unique case IDs")
        cases = {case_id: cases[case_id] for case_id in selected}
    args.output.mkdir(parents=True)
    variants = provider.export_provenance(args.output, "split-only-prediction", sorted(cases))
    variant_index = {(row["case_id"], row["strategy"]): row for row in variants}
    atomic_write_json(args.output / "variants.json", variants)
    predictions = []
    with gpu_lock(args.calibration.parent / "gpu.lock", 10, "transfer split-only predictions"):
        backend = CudaBackend()
        configure_torch(backend.torch, str(backend.device))
        device = backend.device_info()
        if device["gpu_uuid"] != args.uuid or cal_fit["device"]["gpu_uuid"] != args.uuid:
            raise RuntimeError("target device and calibration must match the requested UUID")
        if list_external_gpu_processes(our_process_ids()):
            raise RuntimeError("another CUDA process is active")
        sampler = TelemetrySampler(interval_s=proto.telemetry_interval_s).start()
        try:
            if not sampler.available:
                raise RuntimeError(f"telemetry unavailable: {sampler.error}")
            with (args.output / "split_timings.jsonl").open("x", encoding="utf-8") as rows:
                ordered_ids = case_order(list(cases), proto.seed)
                for index, case_id in enumerate(ordered_ids):
                    case = cases[case_id]
                    t_splits, errors = {}, {}
                    if case["planned_status"] == "planned":
                        try:
                            split = provider.prepare(case, "split")
                            seeds = [str(seed) if seed else "zero" for seed in case["input_seeds"]]
                            if not all(split.check(seed).get("pass") is True for seed in seeds):
                                raise RuntimeError("split numerical check failed")
                            with backend.torch.inference_mode():
                                modes = proto.modes if index % 2 == 0 else reversed(proto.modes)
                                for mode in modes:
                                    try:
                                        variant = split if mode == "direct" else backend.make_replay(split)[0]
                                        if mode == "replay" and variant.check("17").get("pass") is not True:
                                            raise RuntimeError("split replay numerical check failed")
                                        t_splits[mode] = collect_mode(backend, variant, sampler, proto, rows, case_id, mode)
                                        if t_splits[mode] is None:
                                            errors[mode] = "incomplete split-only blocks"
                                    except Exception as exc:
                                        errors[mode] = str(exc)
                        except Exception as exc:
                            errors = dict.fromkeys(proto.modes, str(exc))
                        finally:
                            provider.release_case(case_id)
                    else:
                        errors = dict.fromkeys(proto.modes, case["exclusion_reason"])
                    for mode in proto.modes:
                        predictions.append(predict_case(case, mode, t_splits.get(mode), errors.get(mode),
                                                        variant_index, calibration, source, args.output))
                    print(f"[{index + 1}/{len(cases)}] {case_id}: {t_splits or errors}", flush=True)
        finally:
            sampler.stop()
            provider.close()
            atomic_write_json(args.output / "telemetry.json", sampler.rows)

    counts = dict(Counter(row["mode"] for row in predictions if row["timing_error"] is None))
    if any(not counts.get(mode) for mode in proto.modes):
        raise RuntimeError(f"no usable split-only data for at least one mode: {counts}")
    atomic_write_json(args.output / "predictions.json", {
        "frozen_utc": utcnow(), "target": device, "protocol": proto.to_dict(),
        "source_run_id": source["run_id"], "source_models_sha256": sha256_file(args.models),
        "target_candidates_sha256": sha256_file(args.candidates / "variants.jsonl"),
        "target_calibration_sha256": sha256_file(args.calibration / "calibration.jsonl"),
        "split_timings_sha256": sha256_file(args.output / "split_timings.jsonl"),
        "script_sha256": sha256_file(__file__), "calibration": calibration,
        "usable_counts": counts, "predictions": predictions,
    })


def predict_case(case, mode, t_split, error, variants, calibration, source, out):
    features = compute_features(case, mode, t_split, variants, calibration, out)
    choices = dict.fromkeys(METHODS)
    benefits = dict.fromkeys(MODELS)
    fits = source["full_fits"][mode]
    if not error:
        for name in MODELS:
            model = fits[name]
            if name == "model_2term_common" and features["x3_us"] is None:
                continue
            if model["identifiable"]:
                benefits[name] = predict_linear(model, features)
            if benefits[name] is not None:
                choices[name] = "fused" if benefits[name] > 0 else "split"
        threshold = fits["byte_threshold"]["tau"]
        if threshold is not None and features["logical_saved_bytes"] is not None:
            choices["byte_threshold"] = "fused" if features["logical_saved_bytes"] > threshold else "split"
        default = variants.get((case["case_id"], "default"), {}).get("default_maps_to")
        choices["default_inductor"] = default if default in {"split", "fused"} else None
        choices.update(always_fused="fused", always_split="split")
    return {**features, "split_only_us": t_split, "timing_error": error,
            "choices": choices, "predicted_benefit_us": benefits}


def compare(args):
    frozen = read_json(args.predictions)
    target_run = read_json(args.target_run / "run.json")
    if frozen["frozen_utc"] >= target_run["started_utc"]:
        raise ValueError("predictions must predate paired measurements")
    device = target_run["device_check"]
    if not device["pass"] or frozen["target"]["gpu_uuid"] not in device["observed_uuids"]:
        raise ValueError("paired measurements must use the same GPU as the frozen predictions")
    source = {(r["case_id"], r["mode"]): r for r in read_jsonl(args.source_results)}
    target = {(r["case_id"], r["mode"]): r for r in read_jsonl(args.target_results)}
    if any(r["run_id"] != frozen["source_run_id"] for r in source.values()):
        raise ValueError("source results must come from the frozen model's run")
    if any(r["run_id"] != target_run["run_id"] for r in target.values()):
        raise ValueError("target results must come from the paired target run")
    predictions = {(r["case_id"], r["mode"]): r for r in frozen["predictions"]}
    common = source.keys() & target.keys()
    decisions = defaultdict(list)
    label_transitions = Counter()
    for key in sorted(common):
        row = target[key]
        label_transitions[row["mode"], row["template"], source[key]["label"], row["label"]] += 1
        for method, choice in predictions[key]["choices"].items():
            if choice is None:
                continue
            split, fused = row["T_split_us"], row["T_fused_us"]
            regret = regret_us(split, fused, choice)
            decision = {"case_id": key[0], "mode": key[1], "template": row["template"],
                        "method": method, "predicted": choice, "actual_label": row["label"],
                        "T_split_us": split, "T_fused_us": fused, "regret_us": regret,
                        "regret_relative": regret / min(split, fused)}
            decisions[key[1], "all", method].append(decision)
            decisions[key[1], row["template"], method].append(decision)
    metrics = []
    for mode in ("direct", "replay"):
        for template in ("all", "P1", "P2", "R1", "R2"):
            n = sum(key[1] == mode and (template == "all" or target[key]["template"] == template)
                    for key in common)
            for method in METHODS:
                metrics.append({"mode": mode, "template": template, "method": method,
                                **aggregate_metrics(decisions[mode, template, method], n)})
    atomic_write_json(args.output, {
        "source_run_id": frozen["source_run_id"], "target_run_id": target_run["run_id"],
        "predictions_sha256": sha256_file(args.predictions), "frozen_utc": frozen["frozen_utc"],
        "paired_started_utc": target_run["started_utc"], "n_common_case_modes": len(common),
        "source_only": sorted(source.keys() - target.keys()), "target_only": sorted(target.keys() - source.keys()),
        "label_transitions": [dict(mode=m, template=t, source_label=s, target_label=d, count=n)
                              for (m, t, s, d), n in sorted(label_transitions.items())],
        "metrics": metrics, "decisions": [row for (m, t, name), rows in decisions.items() if t == "all" for row in rows],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    freeze = commands.add_parser("freeze")
    for name in ("models", "calibration", "candidates", "output"):
        freeze.add_argument(f"--{name}", type=Path, required=True)
    freeze.add_argument("--case-list", type=Path, help="optional frozen list of case IDs, one per line")
    freeze.add_argument("--uuid", required=True)
    freeze.set_defaults(execute=freeze_predictions)
    score = commands.add_parser("compare")
    for name in ("predictions", "source-results", "target-results", "target-run", "output"):
        score.add_argument(f"--{name}", type=Path, required=True)
    score.set_defaults(execute=compare)
    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()
