"""Paired block collection: n policy, order, attempts, resume."""
from __future__ import annotations

import math
import random
import time
import traceback
from pathlib import Path
from typing import Any

from .backend import Backend, PreparedVariant
from .protocol import Protocol
from .provider import variant_contract_problems
from .rundir import RunContext
from .util import sha256_text, utcnow


def stable_seed(*parts: Any) -> int:
    return int(sha256_text("|".join(map(str, parts)))[:16], 16)


def normalize_case_filter(spec: str | None, catalog: list[dict]) -> list[str]:
    if not spec:
        return [c["case_id"] for c in catalog]
    wanted: set[str] = set()
    by_id = {c["case_id"]: c for c in catalog}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token in by_id:
            wanted.add(token)
        elif token.upper() in ("P1", "P2", "R1", "R2", "N1"):
            wanted.update(c["case_id"] for c in catalog if c["template"] == token.upper())
        elif token in ("anchor", "long", "layout", "rmsnorm"):
            # followup group selectors (plan/followup-3090)
            wanted.update(c["case_id"] for c in catalog if c.get("group") == token)
        elif token in ("P", "ELEM"):
            wanted.update(c["case_id"] for c in catalog if c["template"].startswith("P"))
        elif token in ("R", "RED"):
            wanted.update(c["case_id"] for c in catalog if c["template"].startswith("R"))
        elif token == "PILOT":
            from .corpus import PILOT_CASES

            wanted.update(PILOT_CASES)
        elif token == "PROFILE":
            # 03-collection mechanism diagnostics: P1 min/max N, R1 four corners.
            wanted.update({
                "P1-n1024-f32-contig-t", "P1-n16777216-f32-contig-t",
                "R1-m16-k128-f32-contig-t", "R1-m16-k8192-f32-contig-t",
                "R1-m4096-k128-f32-contig-t", "R1-m4096-k8192-f32-contig-t",
            })
        else:
            raise ValueError(f"unknown case selector {token!r}")
    selected = [c["case_id"] for c in catalog if c["case_id"] in wanted]
    if not selected:
        raise ValueError(f"selector {spec!r} matched no case in the catalog")
    return selected


def case_order(ids: list[str], seed: int) -> list[str]:
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order


def block_patterns(case_id: str, mode: str, proto: Protocol) -> list[str]:
    rng = random.Random(stable_seed(proto.seed, case_id, mode, "patterns"))
    pats = ["SF"] * proto.pattern_blocks[0] + ["FS"] * proto.pattern_blocks[1]
    rng.shuffle(pats)
    return pats


class Runner:
    def __init__(
        self,
        ctx: RunContext,
        provider,
        backend: Backend,
        protocol: Protocol,
        telemetry=None,
        *,
        formal: bool = True,
        allow_unverified: bool = False,
        stop_after_blocks: int | None = None,
    ):
        self.ctx = ctx
        self.provider = provider
        self.backend = backend
        self.proto = protocol
        self.telemetry = telemetry
        self.formal = formal
        self.allow_unverified = allow_unverified
        self.stop_after_blocks = stop_after_blocks
        self.catalog = {c["case_id"]: c for c in provider.catalog()}
        self.existing: dict[tuple, dict] = {}
        self._executed_blocks = 0
        self._variants_ready: dict[str, dict[str, PreparedVariant]] = {}
        self._case_checks: dict[str, bool] = {}
        self._dev_info = backend.device_info()
        self._written_variants: set[tuple[str, str]] = {
            (v["case_id"], v["strategy"]) for v in ctx.read_table("variants")
        }
        self._written_checks: set[tuple[str, str, str, str | None]] = {
            (c["case_id"], c["strategy"], c["input_id"], c.get("mode"))
            for c in ctx.read_table("checks")
        }
        self.our_pids = None
        self._load_existing()

    # ---------------- resume index ----------------
    def _load_existing(self) -> None:
        self.ctx.recover_timings()
        rows = self.ctx.read_table("timings")
        for r in rows:
            key = (r["case_id"], r["mode"], r["block"], r["attempt"], r["strategy"])
            if key in self.existing:
                raise RuntimeError(f"duplicate timing key on disk: {key}")
            self.existing[key] = r

    def rows_for(self, case_id: str, mode: str) -> list[dict]:
        return [
            r
            for (c, m, _b, _a, _s), r in self.existing.items()
            if c == case_id and m == mode
        ]

    def canonical_blocks(self, case_id: str, mode: str) -> dict[int, int]:
        """block -> first attempt with two ok rows sharing n."""
        by: dict[tuple[int, int], dict[str, dict]] = {}
        for (c, m, b, a, s), r in self.existing.items():
            if c == case_id and m == mode and s in self.proto.strategies:
                by.setdefault((b, a), {})[s] = r
        canonical: dict[int, int] = {}
        for (b, a), pair in by.items():
            if set(pair) == set(self.proto.strategies):
                rows = list(pair.values())
                ns = {r["n"] for r in rows}
                if all(r["quality"] == "ok" for r in rows) and len(ns) == 1:
                    if b not in canonical or a < canonical[b]:
                        canonical[b] = a
        return canonical

    def attempt_numbers(self, case_id: str, mode: str, block: int) -> list[int]:
        return sorted(
            {
                a
                for (c, m, b, a, s) in self.existing
                if c == case_id and m == mode and b == block
                and s in ("split", "fused")
            }
        )

    # ---------------- preparation ----------------
    def _ensure_variants(self, case: dict) -> dict[str, PreparedVariant]:
        cid = case["case_id"]
        if cid in self._variants_ready:
            return self._variants_ready[cid]
        prepared: dict[str, PreparedVariant] = {}
        for strategy in self.proto.strategies:
            v = self.provider.prepare(case, strategy)
            prepared[strategy] = v
            problems = variant_contract_problems(v.meta, formal=self.formal)
            if problems and not self.allow_unverified:
                raise RuntimeError(
                    f"{cid}/{strategy} violates the candidate contract: {problems}; "
                    "pass --allow-unverified-candidates only for dev/synthetic runs"
                )
            if problems:
                v.meta["contract_problems"] = problems
        # checks run once per case, outside all timed regions
        checks_ok = True
        for strategy, v in prepared.items():
            for input_id in dict.fromkeys("zero" if s == 0 else str(s)
                                           for s in case["input_seeds"]):
                res = v.check(input_id)
                key = (cid, strategy, input_id, "direct")
                if key not in self._written_checks:
                    row = {
                        "run_id": self.ctx.run_id,
                        "case_id": cid,
                        "strategy": strategy,
                        "input_id": input_id,
                        "mode": "direct",
                        "ts_utc": utcnow(),
                    }
                    row.update(res)
                    self.ctx.append("checks", row)
                    self._written_checks.add(key)
                if res.get("pass") is not True:
                    checks_ok = False
        self._case_checks[cid] = checks_ok
        for strategy, v in prepared.items():
            if (cid, strategy) in self._written_variants:
                continue
            meta = v.meta
            row = {
                "run_id": self.ctx.run_id,
                "case_id": cid,
                "strategy": strategy,
                "ts_utc": utcnow(),
                "template": case["template"],
                "compiled": True,
                "artifact_sha256": meta.get("artifact_sha256"),
                "artifacts": meta.get("artifacts", []),
                "n_kernels": meta.get("n_kernels"),
                "preallocated": meta.get("preallocated"),
                "compile_ms": meta.get("compile_ms"),
                "submit_modes": list(self.proto.modes),
                "kernel_list": meta.get("kernel_list"),
                "default_equivalent": meta.get("default_equivalent"),
                "default_partition": meta.get("default_partition"),
                "provider": meta.get("provider"),
                "notes": meta.get("notes"),
                "contract_problems": meta.get("contract_problems", []),
                "preallocated_inputs": meta.get("preallocated_inputs"),
                "artifact_path": meta.get("artifact_dir") or (
                    (meta.get("artifacts") or [None])[0]
                ),
                "artifact_dir": meta.get("artifact_dir"),
                "candidate_run": meta.get("candidate_run"),
                "cache_reused": meta.get("cache_reused"),
                "load_ms": meta.get("load_ms"),
                "output_code_sha256": meta.get("output_code_sha256"),
                "kernel_names": meta.get("kernel_names"),
                "t_materialized": meta.get("t_materialized"),
                "edge_verdict": meta.get("edge_verdict"),
                "default_maps_to": meta.get("default_maps_to"),
                "synthetic": self.ctx.synthetic,
            }
            self.ctx.append("variants", row)
            self._written_variants.add((cid, strategy))
        self._variants_ready[cid] = prepared
        return prepared

    def _prepare_mode(self, case: dict, mode: str,
                      direct: dict[str, PreparedVariant]) -> tuple[dict[str, PreparedVariant], dict]:
        if mode == "direct":
            return direct, {}
        if mode != "replay":
            raise ValueError(f"unknown mode {mode}")
        wrapped: dict[str, PreparedVariant] = {}
        capture_ms: dict[str, float] = {}
        for strategy, v in direct.items():
            w, ms = self.backend.make_replay(v)
            wrapped[strategy] = w
            capture_ms[strategy] = ms
        # Replay must be checked at least once after capture (03-collection).
        first_input = str(case["input_seeds"][0]) if case.get("input_seeds") else "17"
        for strategy, w in wrapped.items():
            ckey = (case["case_id"], strategy, first_input, "replay")
            res = w.check(first_input)
            if ckey not in self._written_checks:
                row = {
                    "run_id": self.ctx.run_id,
                    "case_id": case["case_id"],
                    "strategy": strategy,
                    "input_id": first_input,
                    "mode": "replay",
                    "ts_utc": utcnow(),
                }
                row.update(res)
                self.ctx.append("checks", row)
                self._written_checks.add(ckey)
            if res.get("pass") is not True:
                raise RuntimeError(f"{case['case_id']}/{strategy}: replay did not pass correctness checks")
        return wrapped, {"capture_ms": capture_ms}

    # ---------------- n policy ----------------
    def _determine_n(self, prepared: dict[str, PreparedVariant], state: dict) -> tuple[int, dict]:
        for s, v in prepared.items():
            self.backend.warmup(v, self.proto.warmup_calls)
        stabilize = None
        if self.backend.name == "cuda" and self.proto.stabilize_ms > 0:
            # Clocks and allocator state settle under sustained load; without
            # this the trial batch can overestimate per-call time by 2x on a
            # laptop-class GPU and the 5 ms rule would be missed later.
            stabilize = {}
            for s, v in prepared.items():
                chunks = 0
                t0 = time.perf_counter()
                while (time.perf_counter() - t0) * 1000.0 < self.proto.stabilize_ms:
                    self.backend.call_n(v, 50)
                    chunks += 1
                self.backend.sync()
                stabilize[s] = {"chunks": chunks,
                                "ms": (time.perf_counter() - t0) * 1000.0}
        trials = {
            s: self.backend.estimate_us(v, self.proto.trial_iters)
            for s, v in prepared.items()
        }
        n = self.proto.choose_n(trials["split"], trials["fused"])
        # Verify the 5 ms rule on a real batch and enlarge n at most twice.
        refine: list[dict] = []
        for _ in range(2):
            times = {s: self.backend.time_batch(v, n, phase="refine")
                     for s, v in prepared.items()}
            fastest_ms = min(times.values())
            refine.append({"n": n, "elapsed_ms": times, "fastest_ms": fastest_ms})
            if fastest_ms >= 0.9 * self.proto.target_batch_ms:
                break
            factor = self.proto.target_batch_ms * self.proto.n_margin / fastest_ms
            n = max(n + 1, min(self.proto.n_cap,
                               math.ceil(n * factor)))
        state["n"] = n
        state["n_trials"] = trials
        state["n_refine"] = refine
        state["stabilize"] = stabilize
        state["n_policy"] = self.proto.n_policy
        state["target_batch_ms"] = self.proto.target_batch_ms
        state["n_margin"] = self.proto.n_margin
        return n, trials

    # ---------------- one block attempt ----------------
    def _run_attempt(
        self,
        case: dict,
        mode: str,
        block: int,
        attempt: int,
        prepared: dict[str, PreparedVariant],
        n: int,
        trials: dict,
        pattern: str,
        order_index: int,
        refine: list | None = None,
    ) -> tuple[list[dict], str]:
        order = ["split", "fused"] if pattern == "SF" else ["fused", "split"]
        rows: list[dict] = []
        pair_failed = False
        for pos, strategy in enumerate(order, 1):
            v = prepared[strategy]
            t0 = time.time()
            w0 = utcnow()
            row = {
                "run_id": self.ctx.run_id,
                "case_id": case["case_id"],
                "mode": mode,
                "block": block,
                "attempt": attempt,
                "strategy": strategy,
                "iteration_order": order,
                "strategy_position": pos,
                "block_pattern": pattern,
                "case_order": order_index,
                "n": n,
                "n_trial_split_us": trials.get("split"),
                "n_trial_fused_us": trials.get("fused"),
                "n_refine": refine,
                "target_batch_ms": self.proto.target_batch_ms,
                "n_policy": self.proto.n_policy,
                "warmup_calls": self.proto.warmup_calls,
                "untimed_calls": self.proto.untimed_calls,
                "elapsed_ms": None,
                "us_per_call": None,
                "window_start_utc": w0,
                "window_end_utc": None,
                "device": self._dev_info.get("device"),
                "gpu_uuid": self.backend.gpu_uuid,
                "stream": "default",
                "submit_mode": mode,
                "artifact_hash": v.meta.get("artifact_sha256"),
                "n_kernels": v.meta.get("n_kernels"),
                "preallocated": v.meta.get("preallocated"),
                "observed_kernel_names": v.meta.get("observed_kernel_names"),
                "replay_prologue_launches_in_trace": v.meta.get("replay_prologue_launches"),
                "launch_verified": v.meta.get("launch_verified"),
                "quality": "ok",
                "reason": None,
                "error": None,
                "ts_utc": utcnow(),
            }
            try:
                self.backend.warmup(v, self.proto.untimed_calls)
                elapsed_ms = self.backend.time_batch(v, n)
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise RuntimeError(f"non-positive/non-finite elapsed_ms={elapsed_ms}")
                quality = {
                    "quality": "ok",
                    "reasons": [],
                    "detail": {},
                }
                if self.telemetry is not None:
                    quality = self.telemetry.assess(t0, time.time(), self.our_pids or set())
                row.update(
                    elapsed_ms=elapsed_ms,
                    us_per_call=elapsed_ms * 1000.0 / n,
                    window_end_utc=utcnow(),
                    quality=quality["quality"],
                    reason=";".join(quality["reasons"]) or None,
                    telemetry_detail=quality.get("detail"),
                )
                if quality["quality"] != "ok":
                    pair_failed = True
            except Exception as exc:
                pair_failed = True
                row.update(
                    window_end_utc=utcnow(),
                    quality="error",
                    reason=type(exc).__name__,
                    error=f"{exc}",
                    traceback="".join(traceback.format_exception_only(type(exc), exc)),
                )
                rows.append(row)
                # First batch failed: do not issue the second batch; the pair is
                # retried as a whole. The half row is kept per protocol.
                break
            rows.append(row)

        # Any failed/contaminated batch invalidates the pair: mark the partner
        # row as pair_incomplete so analysis never pairs it.
        if pair_failed:
            errored = any(r["quality"] == "error" for r in rows)
            for r in rows:
                if r["quality"] == "ok":
                    r["quality"] = "pair_incomplete"
                    r["reason"] = (
                        "partner_error" if errored else "partner_contaminated"
                    )
        status = "complete" if len(rows) == 2 and all(r["quality"] == "ok" for r in rows) else "retry"
        return rows, status

    # ---------------- state ----------------
    def _init_state(self, case: dict, mode: str) -> dict:
        state = self.ctx.load_state(case["case_id"], mode)
        ph = self.ctx.meta["protocol_hash"]
        if state is not None:
            if state.get("protocol_hash") != ph:
                raise RuntimeError(
                    f"{case['case_id']}/{mode}: state protocol hash mismatch"
                )
            return state
        patterns = block_patterns(case["case_id"], mode, self.proto)
        state = {
            "case_id": case["case_id"],
            "mode": mode,
            "protocol_hash": ph,
            "n": None,
            "n_trials": None,
            "patterns": patterns,
            "status": "planned",
            "valid_pairs": 0,
            "attempt_blocks": {},
            "created_utc": utcnow(),
            "updated_utc": utcnow(),
        }
        self.ctx.save_state(state)
        return state

    def _refresh_state(self, state: dict, final: bool = False) -> None:
        cid, mode = state["case_id"], state["mode"]
        canon = self.canonical_blocks(cid, mode)
        state["valid_pairs"] = len(canon)
        state["attempt_blocks"] = {
            str(b): self.attempt_numbers(cid, mode, b) for b in range(self.proto.blocks)
            if self.attempt_numbers(cid, mode, b)
        }
        if len(canon) == self.proto.blocks:
            state["status"] = "valid"
        elif final and state["status"] not in ("replay_capture_error", "prepare_error"):
            failed = [
                b for b in range(self.proto.blocks)
                if b not in canon
                and len(self.attempt_numbers(cid, mode, b)) >= self.proto.max_attempts
            ]
            state["status"] = "incomplete"
            state["failed_blocks"] = failed
        state["updated_utc"] = utcnow()
        self.ctx.save_state(state)

    def _ensure_cases(self, case_ids: list[str]) -> None:
        if (self.ctx.run_dir / "cases.jsonl").exists():
            return
        rows = []
        for cid in case_ids:
            case = self.catalog[cid]
            row = dict(case)
            row.update(run_id=self.ctx.run_id, ts_utc=utcnow(),
                       synthetic=self.ctx.synthetic)
            rows.append(row)
        self.ctx.append("cases", rows)

    def _time_default(self, case: dict, mode: str, state: dict, n: int,
                      order_index: int) -> None:
        """Descriptive, unpaired default timing when it is a third partition.

        03-collection step 4: independent default timing uses the same n and 30
        batches and is marked unpaired. Pairs are never formed with it.
        """
        dm = (case.get("source") or {}).get("default_maps_to") or case.get("default_maps_to")
        if dm != "distinct":
            return
        if len(self.canonical_blocks(case["case_id"], mode)) != self.proto.blocks:
            return
        cid = case["case_id"]
        existing = {
            r["block"] for r in self.rows_for(cid, mode) if r["strategy"] == "default"
        }
        if len(existing) == self.proto.blocks:
            return
        try:
            v = self.provider.prepare(case, "default")
            problems = variant_contract_problems(v.meta, formal=self.formal)
            if problems and not self.allow_unverified:
                state["default_timing"] = {"status": "unverified_candidate",
                                           "problems": problems}
                self.ctx.save_state(state)
                return
            if mode == "replay":
                v, _ = self.backend.make_replay(v)
            first_input = str(case["input_seeds"][0])
            if v.check(first_input).get("pass") is not True:
                raise RuntimeError(f"{cid}/default/{mode}: correctness check failed")
            self.backend.warmup(v, self.proto.warmup_calls)
        except Exception as exc:
            state["default_timing"] = {
                "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
            self.ctx.save_state(state)
            self.ctx.log(f"{cid}/{mode}: default timing skipped: {exc}")
            return
        rows_written = 0
        for block in range(self.proto.blocks):
            if block in existing:
                continue
            t0 = time.time()
            row = {
                "run_id": self.ctx.run_id,
                "case_id": cid,
                "mode": mode,
                "block": block,
                "attempt": 1,
                "strategy": "default",
                "iteration_order": ["default"],
                "strategy_position": 1,
                "block_pattern": "default",
                "case_order": order_index,
                "n": n,
                "n_trial_split_us": (state.get("n_trials") or {}).get("split"),
                "n_trial_fused_us": (state.get("n_trials") or {}).get("fused"),
                "n_refine": state.get("n_refine"),
                "target_batch_ms": self.proto.target_batch_ms,
                "n_policy": self.proto.n_policy,
                "warmup_calls": self.proto.warmup_calls,
                "untimed_calls": self.proto.untimed_calls,
                "elapsed_ms": None,
                "us_per_call": None,
                "window_start_utc": utcnow(),
                "window_end_utc": None,
                "device": self._dev_info.get("device"),
                "gpu_uuid": self.backend.gpu_uuid,
                "stream": "default",
                "submit_mode": mode,
                "artifact_hash": v.meta.get("artifact_sha256"),
                "n_kernels": v.meta.get("n_kernels"),
                "preallocated": v.meta.get("preallocated"),
                "paired": False,
                "observed_kernel_names": v.meta.get("observed_kernel_names"),
                "replay_prologue_launches_in_trace": v.meta.get("replay_prologue_launches"),
                "launch_verified": v.meta.get("launch_verified"),
                "quality": "ok",
                "reason": None,
                "error": None,
                "ts_utc": utcnow(),
            }
            try:
                self.backend.warmup(v, self.proto.untimed_calls)
                elapsed_ms = self.backend.time_batch(v, n)
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise RuntimeError(f"non-positive/non-finite elapsed_ms={elapsed_ms}")
                quality = {"quality": "ok", "reasons": [], "detail": {}}
                if self.telemetry is not None:
                    quality = self.telemetry.assess(t0, time.time(), self.our_pids or set())
                row.update(elapsed_ms=elapsed_ms,
                           us_per_call=elapsed_ms * 1000.0 / n,
                           window_end_utc=utcnow(),
                           quality=quality["quality"],
                           reason=";".join(quality["reasons"]) or None,
                           telemetry_detail=quality.get("detail"))
            except Exception as exc:
                row.update(window_end_utc=utcnow(), quality="error",
                           reason=type(exc).__name__, error=str(exc))
            self.ctx.commit_block(cid, mode, block, 1, [row], "default", tag="default")
            self.existing[(cid, mode, block, 1, "default")] = row
            rows_written += 1
        state["default_timing"] = {"status": "done", "rows": len(existing) + rows_written,
                                   "n": n}
        self.ctx.save_state(state)
        self.ctx.log(f"{cid}/{mode}: default (unpaired) rows={len(existing) + rows_written}")

    # ---------------- main ----------------
    def run(self, case_ids: list[str], case_order_list: list[str] | None = None) -> dict:
        from .telemetry import our_process_ids

        self.our_pids = our_process_ids()
        self._ensure_cases(case_ids)
        order = case_order_list or case_order(case_ids, self.proto.seed)
        order_index = {cid: i for i, cid in enumerate(order)}
        from .util import write_jsonl_atomic

        order_path = self.ctx.run_dir / "case_order.jsonl"
        if not order_path.exists():
            write_jsonl_atomic(
                order_path,
                [{"order": i, "case_id": cid} for i, cid in enumerate(order)],
            )
        summary: dict[str, Any] = {
            "started_utc": utcnow(),
            "cases": {},
            "executed_blocks": 0,
            "paused": False,
        }
        for cid in order:
            case = self.catalog[cid]
            if case.get("planned_status") not in (None, "planned", "valid"):
                summary["cases"][cid] = {"status": "excluded", "reason": case.get("exclusion_reason")}
                continue
            already = {
                mode: len(self.canonical_blocks(cid, mode)) == self.proto.blocks
                for mode in self.proto.modes
            }
            default_map = (case.get("source") or {}).get("default_maps_to") or case.get("default_maps_to")
            if all(already.values()) and default_map != "distinct":
                summary["cases"][cid] = {
                    mode: {"status": "valid", "valid_pairs": self.proto.blocks}
                    for mode in self.proto.modes
                }
                continue
            try:
                direct = self._ensure_variants(case)
            except Exception as exc:
                self.ctx.log(f"{cid}: preparation/contract failure: {exc}")
                for mode in self.proto.modes:
                    state = self._init_state(case, mode)
                    state.update(status="prepare_error", last_error=str(exc), updated_utc=utcnow())
                    self.ctx.save_state(state)
                summary["cases"][cid] = {"status": "prepare_error", "error": str(exc)}
                if hasattr(self.provider, "release_case"):
                    self.provider.release_case(cid)
                continue
            if not self._case_checks.get(cid, False):
                for mode in self.proto.modes:
                    state = self._init_state(case, mode)
                    state.update(status="incorrect", reason="correctness_check_failed",
                                 updated_utc=utcnow())
                    self.ctx.save_state(state)
                summary["cases"][cid] = {"status": "incorrect"}
                self.ctx.log(f"{cid}: correctness check failed; skipped")
                self._variants_ready.pop(cid, None)
                if hasattr(self.provider, "release_case"):
                    self.provider.release_case(cid)
                continue
            modes = self.proto.modes if order_index[cid] % 2 == 0 else self.proto.modes[::-1]
            for mode in modes:
                state = self._init_state(case, mode)
                if len(self.canonical_blocks(cid, mode)) == self.proto.blocks:
                    state["status"] = "valid"
                    self.ctx.save_state(state)
                    self._time_default(case, mode, state, state["n"], order_index[cid])
                    summary["cases"].setdefault(cid, {})[mode] = {
                        "status": "valid",
                        "valid_pairs": self.proto.blocks,
                    }
                    continue
                if self.stop_after_blocks is not None and self._executed_blocks >= self.stop_after_blocks:
                    state["status"] = "paused"
                    self.ctx.save_state(state)
                    summary["paused"] = True
                    summary["cases"].setdefault(cid, {})[mode] = {
                        "status": "paused",
                        "valid_pairs": len(self.canonical_blocks(cid, mode)),
                    }
                    continue
                try:
                    prepared, mode_extra = self._prepare_mode(case, mode, direct)
                except Exception as exc:
                    state.update(
                        status="replay_capture_error" if mode == "replay" else "prepare_error",
                        last_error=f"{type(exc).__name__}: {exc}",
                        updated_utc=utcnow(),
                    )
                    self.ctx.save_state(state)
                    self.ctx.log(f"{cid}/{mode}: prepare failed: {exc}")
                    summary["cases"].setdefault(cid, {})[mode] = {
                        "status": state["status"],
                        "error": state["last_error"],
                    }
                    continue
                if state.get("capture_ms") is None and mode_extra:
                    state.update(mode_extra)
                    self.ctx.save_state(state)
                if state.get("n") is None:
                    try:
                        n, trials = self._determine_n(prepared, state)
                    except Exception as exc:
                        state.update(status="prepare_error",
                                     last_error=f"{type(exc).__name__}: {exc}",
                                     updated_utc=utcnow())
                        self.ctx.save_state(state)
                        self.ctx.log(f"{cid}/{mode}: n determination failed: {exc}")
                        summary["cases"].setdefault(cid, {})[mode] = {
                            "status": "prepare_error", "error": state["last_error"]}
                        continue
                    state["n_trials"] = trials
                    self.ctx.save_state(state)
                n = state["n"]
                trials = state["n_trials"]
                state["status"] = "running"
                self.ctx.save_state(state)

                patterns = state["patterns"]
                for block in range(self.proto.blocks):
                    canon = self.canonical_blocks(cid, mode)
                    if block in canon:
                        continue
                    attempts = self.attempt_numbers(cid, mode, block)
                    while block not in self.canonical_blocks(cid, mode):
                        if len(attempts) >= self.proto.max_attempts:
                            break
                        if (self.stop_after_blocks is not None
                                and self._executed_blocks >= self.stop_after_blocks):
                            break
                        attempt = (attempts[-1] + 1) if attempts else 1
                        rows, status = self._run_attempt(
                            case, mode, block, attempt, prepared, n, trials,
                            patterns[block], order_index[cid],
                            state.get("n_refine"),
                        )
                        self.ctx.commit_block(
                            cid, mode, block, attempt, rows,
                            "complete" if status == "complete" else "retry",
                        )
                        for r in rows:
                            self.existing[
                                (r["case_id"], r["mode"], r["block"], r["attempt"], r["strategy"])
                            ] = r
                        attempts.append(attempt)
                        if status != "complete" and self.telemetry is not None and any(
                            r["quality"] == "contaminated" for r in rows
                        ):
                            # Let the 1 Hz sampler observe the new steady state
                            # before re-timing the pair.
                            self.telemetry.settle()
                        self._executed_blocks += 1
                        summary["executed_blocks"] += 1
                        ok = all(r["quality"] == "ok" for r in rows) and len(rows) == 2
                        if ok:
                            self.ctx.log(
                                f"{cid}/{mode} block {block} attempt {attempt} "
                                f"n={n} elapsed_ms={rows[0]['elapsed_ms']:.4f}"
                            )
                        else:
                            self.ctx.log(
                                f"{cid}/{mode} block {block} attempt {attempt} retry "
                                f"({'; '.join(str(r['reason']) for r in rows)})"
                            )
                        self._refresh_state(state)
                        if self.stop_after_blocks is not None and self._executed_blocks >= self.stop_after_blocks:
                            state["status"] = "paused"
                            self.ctx.save_state(state)
                            summary["paused"] = True
                            break
                    if summary["paused"]:
                        break
                self._refresh_state(state, final=(not summary["paused"]))
                if not summary["paused"]:
                    self._time_default(case, mode, state, n, order_index[cid])
                summary["cases"].setdefault(cid, {})[mode] = {
                    "status": state["status"],
                    "valid_pairs": state["valid_pairs"],
                    "n": state["n"],
                    "n_trials": state["n_trials"],
                    "capture_ms": state.get("capture_ms"),
                }
                if summary["paused"]:
                    break
            if summary["paused"]:
                break
            prepared_direct = self._variants_ready.get(cid)
            if prepared_direct:
                for v in prepared_direct.values():
                    v.close()
                self._variants_ready.pop(cid, None)
            if hasattr(self.provider, "release_case"):
                self.provider.release_case(cid)
        summary["finished_utc"] = utcnow()
        self.ctx.update(last_run_summary=summary, status="paused" if summary["paused"] else "running")
        return summary
