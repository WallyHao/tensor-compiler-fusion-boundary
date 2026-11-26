"""Command line entry points.

    # 72-case manifest
    python -m fusionboundary manifest --run runs/pilot-20261125

    # two pilots, all three strategies, isolated caches, one subprocess each
    python -m fusionboundary pilot --run runs/pilot-20261125

    # any subset / full corpus
    python -m fusionboundary compile --run runs/full-20261126 --pilot
    python -m fusionboundary compile --run runs/full-20261126 --all
    python -m fusionboundary compile --run runs/full-20261126 \
        --case R1-m4096-k8192-f16-contig-t --strategy split

    # reload through the measurement interface and re-check numerics
    python -m fusionboundary verify --run runs/pilot-20261125 --pilot
    python -m fusionboundary status --run runs/pilot-20261125

The ``worker`` subcommand is internal: one (case, strategy) pair in its own
process with an isolated ``TORCHINDUCTOR_CACHE_DIR``. The driver only ever
appends to the run tables, so the JSONL files have a single writer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from . import corpus, records


# --------------------------------------------------------------------------
# helpers (no torch import at module scope: the worker sets its cache dir first)
# --------------------------------------------------------------------------
def _rundir(path: str) -> records.RunDir:
    path = os.path.abspath(path)
    return records.RunDir(os.path.dirname(path), os.path.basename(path))


def _rel(run: records.RunDir, path: str) -> str:
    return os.path.relpath(path, run.path)


def _worker_env(cache_dir: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    # Fallback evidence channel: torch's artifact logger writes the generated
    # code into the worker log even on a version whose trace directory layout
    # differs from the parser's expectations. Only names valid in this torch
    # version may appear here (TORCH_LOGS rejects unknown settings at import).
    env.setdefault("TORCH_LOGS", "output_code")
    if extra:
        env.update(extra)
    return env


def _select_cases(args: argparse.Namespace) -> list[corpus.Case]:
    if getattr(args, "followup", False):
        if getattr(args, "pilot", False):
            return corpus.followup_pilot_cases()
        ids = getattr(args, "case", None) or []
        if not ids:
            return corpus.followup_cases()
    if getattr(args, "pilot", False):
        return corpus.pilot_cases()
    if getattr(args, "all", False):
        cases = list(corpus.iter_cases(
            templates=tuple(args.templates.split(",")) if getattr(args, "templates", None) else None,
            dtypes=tuple(args.dtypes.split(",")) if getattr(args, "dtypes", None) else None,
        ))
        if getattr(args, "limit", 0):
            cases = cases[: args.limit]
        return cases
    ids = getattr(args, "case", None) or []
    if not ids:
        raise SystemExit("select cases with --pilot, --all, --followup or --case ID (repeatable)")
    index = {**corpus.corpus_index(), **corpus.followup_index()}
    missing = [c for c in ids if c not in index]
    if missing:
        raise SystemExit(f"unknown case ids: {missing}")
    return [index[c] for c in ids]


def _strategies(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "strategy", None) or "default,split,fused"
    out: list[str] = []
    for s in raw.split(","):
        s = s.strip()
        if s:
            if s not in ("default", "split", "fused", *("diag_clone", "diag_graph_break", "diag_no_fusion")):
                raise SystemExit(f"unknown strategy {s!r}")
            out.append(s)
    return out


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def cmd_manifest(args: argparse.Namespace) -> int:
    run = _rundir(args.run)
    if getattr(args, "followup", False):
        summary = corpus.write_followup_manifest(run.cases_jsonl)
    else:
        summary = corpus.write_manifest(run.cases_jsonl)
    records.write_json(os.path.join(run.path, "cases.json"),
                       {"summary": summary, "generated": records.utc_now()})
    print(json.dumps(summary, indent=2))
    print(f"manifest -> {run.cases_jsonl}")
    return 0


# --------------------------------------------------------------------------
# worker: one (case, strategy) in an isolated process
# --------------------------------------------------------------------------
def cmd_worker(args: argparse.Namespace) -> int:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

    import torch

    from . import strategies, validate

    torch.manual_seed(0)
    case = corpus.get_case(args.case)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = args.artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)

    t_start = time.perf_counter()
    result = strategies.compile_variant(case, args.strategy, artifact_dir, device=device,
                                       run_trace=True, gate_index=args.gate_index)
    checks: list[dict[str, Any]] = []
    if result.compiled is not None and result.status in ("valid", "partition_mismatch", "unsupported"):
        with torch.no_grad():
            checks = validate.check_case(case, lambda x: result.compiled(x), args.strategy, device)
        failed = [c for c in checks if not c["pass"]]
        if failed:
            result.status = "incorrect"
            result.errors.append(
                f"{len(failed)}/{len(checks)} checks failed; first: {failed[0]['reason']}")

    # graph-level evidence (graph break count, FX graph text)
    fx = {}
    if result.compiled is not None:
        fn, _overrides, _note = strategies.entry_fn(case, args.strategy)
        fx = strategies._capture_fx(case, fn, args.strategy, args.gate_index)

    # ---- compile key + per-variant tables --------------------------------
    code_hash = records.sha256_file(__file__)
    compile_key = {
        "case_id": case.case_id,
        "strategy": args.strategy,
        "device": device,
        "gpu": records.gpu_snapshot() if device == "cuda" else None,
        "shape": list(case.shape),
        "dtype": case.dtype,
        "layout": case.layout,
        "input_stride": list(corpus.input_stride(tuple(case.shape), case.layout)),
        "t_shape": list(case.t_shape),
        "t_dtype": case.t_dtype,
        "guard": ({"symbol": result.gate_symbol, "edge": "t",
                   "index": result.gate_index,
                   "blocks": len(result.guard_log)} if args.strategy == "split" else None),
        # The configuration re-applied while warming the archived wrapper.
        "config_overrides": dict(strategies.DEFAULT_OVERRIDES,
                                 **(_overrides_for(args.strategy, case))),
        "inductor_config": result.config,
        "versions": _versions(),
        "cache_dir": args.cache_dir,
        "code_sha256": {"cli.py": code_hash, "template_fn": case.template},
        "artifact_hashes": result.trace.get("files", {}),
    }
    records.write_json(os.path.join(artifact_dir, "compile_key.json"), compile_key)
    records.write_json(os.path.join(artifact_dir, "fx_graph.json"), fx)
    if checks:
        records.write_jsonl(os.path.join(artifact_dir, "checks.jsonl"), checks)

    row = {
        "case_id": case.case_id,
        "strategy": args.strategy,
        "status": result.status,
        "group": case.group,
        "layout": case.layout,
        "input_stride": list(corpus.input_stride(tuple(case.shape), case.layout)),
        "artifact_dir": _rel(_rundir(args.run), artifact_dir) if args.run else artifact_dir,
        "output_code": result.trace.get("output_code"),
        "output_code_sha256": (result.evidence or {}).get("output_code_sha256"),
        "kernel_names": (result.evidence or {}).get("kernels", []),
        "kernel_count": (result.evidence or {}).get("kernel_count"),
        "t_materialized": (result.evidence or {}).get("t_materialized"),
        "t_alloc_dtype": (result.evidence or {}).get("t_alloc_dtype"),
        "t_alloc_shape": (result.evidence or {}).get("t_alloc_shape"),
        "edge_verdict": (result.evidence or {}).get("verdict"),
        "fusion_decision_stage": ((result.evidence or {}).get("fusion_decision") or {}).get("stage"),
        "scheduler_fusion_pairs": ((result.evidence or {}).get("fusion_decision") or {}).get(
            "scheduler_fusion_pairs_considered"),
        "guard_blocks": len(result.guard_log),
        "realize_events": [
            {"fx_node": e["fx_node"], "buffer": e.get("buffer"), "status": e["status"]}
            for e in result.realize_log
        ],
        "guard_symbols": [h["gate"] for h in result.guard_log][:1],
        "gate_symbol": result.gate_symbol,
        "gate_index": result.gate_index,
        "graph_count": fx.get("graph_count"),
        "graph_break_count": fx.get("graph_break_count"),
        "compile_ms": result.compile_ms,
        "first_call_ms": result.first_call_ms,
        "second_call_ms": result.second_call_ms,
        "checks_total": len(checks),
        "checks_failed": sum(1 for c in checks if not c["pass"]),
        "errors": result.errors,
        "notes": [n for n in result.notes if n],
        "wall_ms": (time.perf_counter() - t_start) * 1e3,
        "finished": records.utc_now(),
    }
    records.write_json(os.path.join(artifact_dir, "variant_row.json"), row)
    records.write_json(os.path.join(artifact_dir, "checks_summary.json"), {
        "total": len(checks), "failed": sum(1 for c in checks if not c["pass"])})
    print(json.dumps({"case_id": case.case_id, "strategy": args.strategy,
                      "status": result.status, "kernels": row["kernel_count"],
                      "checks": len(checks), "errors": result.errors}, sort_keys=True))
    return 0 if result.status == "valid" else 1


def _overrides_for(strategy: str, case) -> dict[str, Any]:  # noqa: ANN001
    """Extra Inductor overrides that a diagnostic control needs."""
    if strategy == "diag_no_fusion":
        return {"max_fusion_size": 0}
    return {}


def _versions() -> dict[str, Any]:
    import torch

    v = {"torch": torch.__version__, "cuda": torch.version.cuda,
         "torch_git": getattr(torch.version, "git_version", None)}
    try:
        import triton

        v["triton"] = triton.__version__
    except Exception:
        v["triton"] = None
    return v


# --------------------------------------------------------------------------
# driver: compile
# --------------------------------------------------------------------------
def cmd_compile(args: argparse.Namespace) -> int:
    run = _rundir(args.run)
    if not os.path.exists(run.cases_jsonl):
        if getattr(args, "followup", False):
            corpus.write_followup_manifest(run.cases_jsonl)
        else:
            corpus.write_manifest(run.cases_jsonl)
    cases = _select_cases(args)
    strategies_wanted = _strategies(args)
    if not os.path.exists(run.run_json):
        records.write_json(run.run_json, {
            "run_id": run.run_id,
            "purpose": getattr(args, "purpose", None) or "corpus",
            "created": records.utc_now(),
            "command": " ".join(sys.argv),
            "env": records.env_snapshot(),
            "git": records.git_state(),
            "cases_requested": [c.case_id for c in cases] if len(cases) <= 8 else
                               f"{len(cases)} cases ({args.templates or 'all templates'})",
            "strategies": _strategies(args),
        })

    device = args.device or ""
    failures = 0
    rows_by_case: dict[str, dict[str, dict[str, Any]]] = {}

    for case in cases:
        rows_by_case[case.case_id] = {}
        for strategy in strategies_wanted:
            if _already_done(run, case.case_id, strategy) and not args.force:
                print(f"[skip] {case.case_id} {strategy} (already recorded)")
                rows_by_case[case.case_id][strategy] = _existing_row(run, case.case_id, strategy)
                continue
            rc = _run_worker(run, case, strategy, device, args.force, gate_index=0)
            row = _load_variant_row(run, case.case_id, strategy)
            # A split guard that provably did not fire (or fired but still fused)
            # means the admission path moved; retry the next candidate in a fresh
            # process with a fresh cache instead of reporting a silent no-op.
            gate_index = 0
            while (strategy == "split" and row is not None
                   and row["status"] in ("unsupported", "partition_mismatch")
                   and gate_index + 1 < len(strategies_gates())):
                gate_index += 1
                print(f"    retry split with gate #{gate_index}: {strategies_gates()[gate_index]}")
                _wipe(os.path.join(run.path, "cache", case.case_id, strategy))
                _wipe(run.artifact_dir(case.case_id, strategy))
                rc = _run_worker(run, case, strategy, device, args.force, gate_index=gate_index)
                row = _load_variant_row(run, case.case_id, strategy)
            if row is None:
                failures += 1
                print(f"[fail] {case.case_id} {strategy}: no variant row (see logs)")
                continue
            _append_variant(run, row)
            _append_checks(run, case.case_id, strategy)
            rows_by_case[case.case_id][strategy] = row
            print(f"[{row['status']}] {case.case_id} {strategy} kernels={row.get('kernel_count')} "
                  f"checks_failed={row.get('checks_failed')}")
            if row["status"] != "valid":
                failures += 1
        _annotate_mapping(run, case, rows_by_case[case.case_id])
    print(f"done: {len(cases)} cases, {failures} non-valid/failed variants")
    return 0 if failures == 0 else 2


def strategies_gates() -> tuple[tuple[str, str], ...]:
    from .strategies import _GATE_CANDIDATES

    return _GATE_CANDIDATES


def _wipe(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _run_worker(run: records.RunDir, case, strategy: str, device: str, force: bool,  # noqa: ANN001
                gate_index: int = 0) -> int:
    artifact_dir = run.artifact_dir(case.case_id, strategy)
    cache_dir = os.path.join(run.path, "cache", case.case_id, strategy)
    if gate_index:
        # each gate attempt must re-run the scheduler, so it needs its own
        # cache: the FX-graph cache key does not include the monkeypatch.
        cache_dir = f"{cache_dir}-gate{gate_index}"
    if force:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(artifact_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "fusionboundary", "worker",
           "--run", run.path, "--case", case.case_id, "--strategy", strategy,
           "--artifact-dir", artifact_dir, "--cache-dir", cache_dir,
           "--gate-index", str(gate_index)]
    if device:
        cmd += ["--device", device]
    log = run.log_path(f"{case.case_id}-{strategy}.log")
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              env=_worker_env(cache_dir), timeout=3600)
    print(f"    worker rc={proc.returncode} in {time.perf_counter() - t0:.1f}s log={log}")
    return proc.returncode


def _load_variant_row(run: records.RunDir, case_id: str, strategy: str) -> dict[str, Any] | None:
    path = os.path.join(run.artifact_dir(case_id, strategy), "variant_row.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _existing_row(run: records.RunDir, case_id: str, strategy: str) -> dict[str, Any] | None:
    for row in records.read_jsonl(run.variants_jsonl):
        if row["case_id"] == case_id and row["strategy"] == strategy:
            return row
    return None


def _already_done(run: records.RunDir, case_id: str, strategy: str) -> bool:
    row = _existing_row(run, case_id, strategy)
    return row is not None and row.get("status") in ("valid", "unsupported", "incorrect",
                                                     "partition_mismatch", "compile_error")


def _append_variant(run: records.RunDir, row: dict[str, Any]) -> None:
    existing = [r for r in records.read_jsonl(run.variants_jsonl)
                if not (r["case_id"] == row["case_id"] and r["strategy"] == row["strategy"])]
    existing.append(row)
    records.write_jsonl(run.variants_jsonl,
                        sorted(existing, key=lambda r: (r["case_id"], r["strategy"])))


def _append_checks(run: records.RunDir, case_id: str, strategy: str) -> None:
    src = os.path.join(run.artifact_dir(case_id, strategy), "checks.jsonl")
    if not os.path.exists(src):
        return
    new = records.read_jsonl(src)
    kept = [r for r in records.read_jsonl(run.checks_jsonl)
            if not (r["case_id"] == case_id and r["strategy"] == strategy and r.get("mode") == "direct")]
    records.write_jsonl(run.checks_jsonl, kept + new)


def _annotate_mapping(run: records.RunDir, case, rows: dict[str, dict[str, Any]]) -> None:  # noqa: ANN001
    """Record whether default == fused == the same partition, and flag rewrites.

    Uses kernel-level signatures (name, aten set, buffers, grid, code hash) from
    each variant's kernels.json, so the comparison is over generated code, not
    over strategy labels.
    """
    from . import forensics

    parsed: dict[str, Any] = {}
    for strategy, row in rows.items():
        if not row or row.get("status") != "valid":
            continue
        kernels_json = os.path.join(run.artifact_dir(case.case_id, strategy), "kernels.json")
        if not os.path.exists(kernels_json):
            continue
        with open(kernels_json, encoding="utf-8") as fh:
            data = json.load(fh)
        parsed[strategy] = data

    if len(parsed) < 2:
        return

    def signature(data: dict[str, Any]) -> list[tuple]:
        sig = []
        for k in data.get("kernels", []):
            sig.append((k["name"], tuple(k["source_nodes"]), tuple(k["original_aten"]),
                        tuple(k["reads"]), tuple(k["writes"]), k["grid"], k["source_sha256"]))
        return sig

    def op_union(data: dict[str, Any]) -> set[str]:
        ops: set[str] = set()
        for k in data.get("kernels", []):
            ops.update(k.get("source_nodes") or [])
            ops.update(k.get("original_aten") or [])
        return ops

    sigs = {s: signature(d) for s, d in parsed.items()}
    notes: dict[str, Any] = {}
    if "default" in sigs and "fused" in sigs:
        notes["partition_equal_default_fused"] = sigs["default"] == sigs["fused"]
    if "default" in sigs and "split" in sigs:
        notes["partition_equal_default_split"] = sigs["default"] == sigs["split"]
    if "split" in sigs and "fused" in sigs:
        notes["partitions_identical_split_fused"] = sigs["split"] == sigs["fused"]
        notes["same_op_set_split_fused"] = op_union(parsed["split"]) == op_union(parsed["fused"])
        # Ops added or dropped anywhere in the graph mean the control changed the
        # program, not only the target edge.
        # A clean pair differs by two local kernels becoming one. For
        # multi-stage reductions, require identical shared kernels in order.
        shared = set(sigs["split"]) & set(sigs["fused"])
        shared_split = [k for k in sigs["split"] if k in shared]
        shared_fused = [k for k in sigs["fused"] if k in shared]
        notes["unrelated_partition_change"] = not notes["partitions_identical_split_fused"] and (
            not notes["same_op_set_split_fused"]
            or shared_split != shared_fused
            or len(sigs["split"]) - len(shared_split) != 2
            or len(sigs["fused"]) - len(shared_fused) != 1
        )
        notes["shared_kernel_count"] = len(shared)
        notes["kernel_count_split"] = len(sigs["split"])
        notes["kernel_count_fused"] = len(sigs["fused"])

    # map default to the candidate it actually corresponds to (default 二选一评价)
    mapping = "distinct" if "default" in sigs else None
    if notes.get("partition_equal_default_fused"):
        mapping = "fused"
    elif notes.get("partition_equal_default_split"):
        mapping = "split"
    notes["default_maps_to"] = mapping

    # persist onto the rows (single writer: the driver)
    rows_all = records.read_jsonl(run.variants_jsonl)
    for row in rows_all:
        if row["case_id"] != case.case_id:
            continue
        if row["strategy"] == "default":
            row["default_maps_to"] = mapping
        row.update({k: v for k, v in notes.items() if k.startswith(("partition_", "same_op_set",
                                                                   "unrelated_", "shared_",
                                                                   "kernel_count_", "partitions_"))})
    records.write_jsonl(run.variants_jsonl, sorted(rows_all, key=lambda r: (r["case_id"], r["strategy"])))
    records.write_json(os.path.join(run.path, "artifacts", case.case_id, "partition_map.json"),
                       notes)


# --------------------------------------------------------------------------
# verify (through the measurement interface)
# --------------------------------------------------------------------------
def cmd_verify(args: argparse.Namespace) -> int:
    from .runtime import CandidateStore

    run = _rundir(args.run)
    store = CandidateStore(run.path)
    cases = _select_cases(args)
    strategies_wanted = [s for s in _strategies(args) if s in ("split", "fused", "default")]
    failures = 0
    for case in cases:
        for strategy in strategies_wanted:
            if store.status(case.case_id, strategy) is None:
                print(f"[skip] {case.case_id} {strategy}: not recorded as valid")
                continue
            cand = store.load(case.case_id, strategy, device=args.device or None)
            rows = cand.verify()
            ok = all(r["pass"] for r in rows)
            rows.append({
                "case_id": case.case_id, "strategy": strategy, "input_id": "load_report",
                "mode": "load", "pass": True,
                "reason": json.dumps(cand.report.summary(), sort_keys=True),
            })
            if not ok:
                failures += 1
            kept = [r for r in records.read_jsonl(run.checks_jsonl)
                    if not (r["case_id"] == case.case_id and r["strategy"] == strategy
                            and r.get("mode") in ("direct", "load"))]
            records.write_jsonl(run.checks_jsonl, kept + rows)
            print(f"[{'ok' if ok else 'FAIL'}] {case.case_id} {strategy} "
                  f"cache_reused={cand.report.cache_reused} "
                  f"first_call_ms={cand.report.first_call_ms:.1f} kernels={cand.report.kernel_count}")
    return 0 if failures == 0 else 2


# --------------------------------------------------------------------------
# pilot: compile + evidence + status in one command
# --------------------------------------------------------------------------
def cmd_pilot(args: argparse.Namespace) -> int:
    run = _rundir(args.run)
    if not os.path.exists(run.cases_jsonl):
        corpus.write_manifest(run.cases_jsonl)
    if not os.path.exists(run.run_json):
        records.write_json(run.run_json, {
            "run_id": run.run_id,
            "purpose": getattr(args, "purpose", "pilot"),
            "created": records.utc_now(),
            "command": " ".join(sys.argv),
            "env": records.env_snapshot(),
            "git": records.git_state(),
            "pilot_cases": [c.case_id for c in corpus.pilot_cases()],
            "strategies": ["default", "split", "fused"],
        })
    args.pilot = True
    args.all = False
    args.case = []
    args.strategy = "default,split,fused"
    rc = cmd_compile(args)
    rc2 = cmd_verify(args)
    _write_pilot_evidence(run)
    _print_status(run)
    return 0 if rc == 0 and rc2 == 0 else 2


def _write_pilot_evidence(run: records.RunDir) -> None:
    rows = records.read_jsonl(run.variants_jsonl)
    checks = records.read_jsonl(run.checks_jsonl)
    lines = [f"# Pilot evidence · run `{run.run_id}`", "",
             f"generated {records.utc_now()} by `{sys.argv[0]}`", ""]
    for case in corpus.pilot_cases():
        crows = [r for r in rows if r["case_id"] == case.case_id]
        lines += [f"## {case.case_id}", "",
                  f"- template `{case.template}` shape={list(case.shape)} dtype={case.dtype} "
                  f"t=({list(case.t_shape)},{case.t_dtype}) logical_save_bytes={case.logical_save_bytes}",
                  f"- tolerance atol={case.atol} rtol={case.rtol}", ""]
        for r in sorted(crows, key=lambda r: r["strategy"]):
            lines.append(f"- **{r['strategy']}**: status=`{r['status']}` kernels={r.get('kernel_count')} "
                         f"t_materialized={r.get('t_materialized')} verdict=`{r.get('edge_verdict')}` "
                         f"checks_failed={r.get('checks_failed')} "
                         f"graph_breaks={r.get('graph_break_count')} "
                         f"default_maps_to={r.get('default_maps_to')}")
            if r.get("errors"):
                lines.append(f"  - errors: {r['errors']}")
        m = os.path.join(run.path, "artifacts", case.case_id, "partition_map.json")
        if os.path.exists(m):
            with open(m, encoding="utf-8") as fh:
                lines += ["", "```json", json.dumps(json.load(fh), indent=2, sort_keys=True), "```", ""]
        for r in sorted(crows, key=lambda r: r["strategy"]):
            ev = os.path.join(run.path, r["artifact_dir"], "evidence.md")
            if os.path.exists(ev):
                with open(ev, encoding="utf-8") as fh:
                    lines += [f"### {r['strategy']} · generated-code evidence", "", fh.read(), ""]
        cchecks = [c for c in checks if c["case_id"] == case.case_id]
        if cchecks:
            lines += ["### numerical checks", "",
                      "| strategy | input | pass | max_abs | max_rel | fp64 max_abs |",
                      "|---|---|---|---|---|---|"]
            for c in cchecks:
                fp64 = (c.get("fp64") or {}).get("max_abs_err")

                def _fmt(value: Any) -> str:
                    return "—" if value is None else f"{value:.3e}"

                lines.append(f"| {c['strategy']} | {c['input_id']} | {c['pass']} | "
                             f"{_fmt(c.get('max_abs_err'))} | {_fmt(c.get('max_rel_err'))} | "
                             f"{_fmt(fp64)} |")
            lines.append("")
    out = run.evidence_path("pilot-evidence.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"evidence -> {out}")


# --------------------------------------------------------------------------
# consistency check over a finished run (plan/phase1-3090/04-data.md 交接验算)
# --------------------------------------------------------------------------
def cmd_check_run(args: argparse.Namespace) -> int:
    run = _rundir(args.run)
    problems: list[str] = []
    cases = corpus.read_manifest(run.cases_jsonl) if os.path.exists(run.cases_jsonl) else []
    rows = records.read_jsonl(run.variants_jsonl)
    checks = records.read_jsonl(run.checks_jsonl)

    followup_ids = set(corpus.followup_index())
    manifest_is_followup = bool(cases) and all(c.case_id in followup_ids for c in cases)
    expected_cases = len(followup_ids) if manifest_is_followup else 72
    if cases and len(cases) != expected_cases:
        problems.append(f"cases.jsonl has {len(cases)} rows, expected {expected_cases}")
    seen = set()
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["case_id"], row["strategy"])
        if key in seen:
            problems.append(f"duplicate variant row {key}")
        seen.add(key)
        by_case.setdefault(row["case_id"], {})[row["strategy"]] = row

    known = {c.case_id for c in cases} if cases else set(by_case)
    for cid in by_case:
        if cases and cid not in known:
            problems.append(f"variant row for unknown case {cid}")

    for cid, strategies in sorted(by_case.items()):
        main = [s for s in ("default", "split", "fused") if s in strategies]
        if not args.partial and len(main) != 3:
            problems.append(f"{cid}: only {main} recorded")
        for strategy, row in sorted(strategies.items()):
            if row["status"] != "valid":
                continue
            artifact = os.path.join(run.path, row["artifact_dir"])
            for name in ("output_code.py", "kernels.json", "edge_evidence.json",
                         "compile_key.json", "evidence.md"):
                path = os.path.join(artifact, name)
                if not os.path.exists(path):
                    problems.append(f"{cid}/{strategy}: missing {name}")
            if not row.get("kernel_count"):
                problems.append(f"{cid}/{strategy}: kernel_count is empty")
            direct = [c for c in checks if c["case_id"] == cid and c["strategy"] == strategy
                      and c.get("mode") == "direct"]
            if len(direct) != 4:
                problems.append(f"{cid}/{strategy}: {len(direct)} direct check rows, expected 4")
            failed = [c for c in direct if not c["pass"]]
            if failed:
                problems.append(f"{cid}/{strategy}: {len(failed)} failed check rows")
            oc = os.path.join(artifact, "output_code.py")
            if os.path.exists(oc) and row.get("output_code_sha256"):
                if records.sha256_file(oc) != row["output_code_sha256"]:
                    problems.append(f"{cid}/{strategy}: output_code.py hash differs from row")

    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(json.dumps({
        "run": run.run_id,
        "cases_with_rows": len(by_case),
        "variant_rows": len(rows),
        "check_rows": len(checks),
        "statuses": statuses,
        "problems": problems,
    }, indent=2))
    return 0 if not problems else 2


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    _print_status(_rundir(args.run))
    return 0


def _print_status(run: records.RunDir) -> None:
    rows = records.read_jsonl(run.variants_jsonl)
    checks = records.read_jsonl(run.checks_jsonl)
    cases = corpus.read_manifest(run.cases_jsonl) if os.path.exists(run.cases_jsonl) else corpus.build_corpus()
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["strategy"]] = r
    print(f"\n=== run {run.run_id}: {len(cases)} cases, {len(rows)} variant rows, "
          f"{len(checks)} check rows ===")
    print(f"{'case_id':<32}{'default':<20}{'split':<20}{'fused':<20}{'checks':<8}maps_to")
    for case in cases:
        if case.case_id not in by_case:
            continue
        r = by_case[case.case_id]
        cells = []
        for s in ("default", "split", "fused"):
            row = r.get(s)
            cells.append("-" if row is None else f"{row['status']}:{row.get('kernel_count')}k")
        cfail = sum(1 for c in checks if c["case_id"] == case.case_id and not c["pass"])
        ctot = sum(1 for c in checks if c["case_id"] == case.case_id)
        mapping = (r.get("default") or {}).get("default_maps_to", "-")
        print(f"{case.case_id:<32}{cells[0]:<20}{cells[1]:<20}{cells[2]:<20}"
              f"{f'{ctot - cfail}/{ctot}':<8}{mapping}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nvariant statuses: {counts}")
    print(f"checked rows: {len(checks)} ({sum(1 for c in checks if not c['pass'])} failed)")
    missing = [c.case_id for c in cases if c.case_id not in by_case]
    if missing:
        print(f"unstarted cases: {len(missing)} (first: {missing[:3]})")


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m fusionboundary", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_case_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--case", action="append", help="case_id (repeatable)")
        p.add_argument("--pilot", action="store_true", help="the two fixed pilots")
        p.add_argument("--all", action="store_true", help="the full 72-case corpus")
        p.add_argument("--followup", action="store_true",
                       help="the 20-case followup plan (plan/followup-3090): "
                            "all 20, or the 3 pilots with --pilot")
        p.add_argument("--templates", help="comma list, e.g. P1,R1")
        p.add_argument("--dtypes", help="comma list, e.g. float32")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--strategy", default=None,
                       help="comma list of default,split,fused[,diag_*] (default: all three)")
        p.add_argument("--device", default=None, help="cuda (default) or cpu (dry run)")
        p.add_argument("--force", action="store_true", help="recompile, wiping artifacts+cache")
        p.add_argument("--run", default="runs/pilot")

    p = sub.add_parser("manifest", help="write the 72-case manifest")
    p.add_argument("--run", default="runs/pilot")
    p.add_argument("--followup", action="store_true",
                   help="write the 20-case followup manifest instead")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("worker", help="internal: compile one variant in an isolated process")
    p.add_argument("--run", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--gate-index", type=int, default=0,
                   help="index into strategies._GATE_CANDIDATES (split only)")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("compile", help="compile variants (one subprocess per case/strategy)")
    add_case_args(p)
    p.add_argument("--purpose", default=None,
                   help="run.json purpose (e.g. local-smoke-4060); written once per run")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("verify", help="reload candidates through the measurement interface")
    add_case_args(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("pilot", help="compile + verify + evidence for the two pilots")
    p.add_argument("--run", default="runs/pilot")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--purpose", default="pilot",
                   help="run.json purpose, e.g. local-smoke-4060 (not 3090 data)")
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("check-run", help="cross-table consistency check of a run")
    p.add_argument("--run", default="runs/pilot")
    p.add_argument("--partial", action="store_true",
                   help="allow cases that do not have all three main strategies yet")
    p.set_defaults(func=cmd_check_run)

    p = sub.add_parser("status", help="table of case x strategy status")
    p.add_argument("--run", default="runs/pilot")
    p.set_defaults(func=cmd_status)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
