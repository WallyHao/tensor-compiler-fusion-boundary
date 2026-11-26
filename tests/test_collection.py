"""Fixtures for the collection protocol (no GPU/torch required)."""
from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from fb.backend import CandidateError, FakeBackend, PreparedVariant
from fb.corpus import PILOT_CASES, build_catalog
from fb.protocol import Protocol
from fb.provider import variant_contract_problems
from fb.rundir import RunContext, RunError
from fb.runner import Runner, block_patterns, case_order, normalize_case_filter
from fb.util import read_json, sha256_json
from fb.verify import verify_run

CASES = {c["case_id"]: c for c in build_catalog() if c["case_id"] in PILOT_CASES}


class FakeProvider:
    provider = "fake-tests"

    def __init__(self, cases=None):
        self.cases = list((cases or CASES).values()) if isinstance(cases or CASES, dict) else cases
        if not isinstance(self.cases, list):
            self.cases = list(self.cases)

    def catalog(self):
        return self.cases

    def prepare(self, case: dict, strategy: str) -> PreparedVariant:
        meta = {
            "provider": "fake-tests",
            "strategy": strategy,
            "artifact_sha256": sha256_json(["fake", case["case_id"], strategy]),
            "artifacts": [f"fake/{case['case_id']}/{strategy}.bin"],
            "n_kernels": 2 if strategy == "split" else 1,
            "preallocated": True,
            "compile_ms": 1.0,
            "submit_mode": "direct",
        }
        return PreparedVariant(
            case_id=case["case_id"], strategy=strategy, meta=meta,
            _call=lambda: None,
            _check=lambda input_id: {"pass": True, "reason": None,
                                     "max_abs_err": 0.0, "max_rel_err": 0.0,
                                     "tolerance": {"atol": 0, "rtol": 0},
                                     "output_shape": case["t_shape"],
                                     "output_dtype": case["t_dtype"],
                                     "nonfinite_count": 0, "reference": "eager"},
        )

    def close(self):
        pass


def make_protocol(**kw) -> Protocol:
    blocks = kw.pop("blocks", 6)
    return Protocol(blocks=blocks, pattern_blocks=((blocks + 1) // 2, blocks // 2),
                    modes=kw.pop("modes", ("direct",)), **kw)


class TelemetryStub:
    def __init__(self, contaminated_calls=0):
        self.calls = 0
        self.contaminated_calls = contaminated_calls

    def settle(self):
        pass

    def assess(self, t0, t1, pids):
        self.calls += 1
        if self.calls <= self.contaminated_calls:
            return {"quality": "contaminated", "reasons": ["external_cuda_process"],
                    "detail": {"calls": self.calls}}
        return {"quality": "ok", "reasons": [], "detail": {}}


def open_run(tmp: str, rid: str, proto: Protocol):
    return RunContext.create(tmp, rid, proto, "dryrun", argv=["test"], synthetic=True)


class CollectionProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fb-tests-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, proto, costs=None, stop=None, telemetry=None, rid="t1",
             provider=None, allow_unverified=True):
        ctx = open_run(self.tmp, rid, proto)
        cost_map = costs or {
            (cid, s): v for cid in CASES for s, v in (("split", 5.0), ("fused", 4.0))
        }
        backend = FakeBackend(cost_map, jitter_us=0.0,
                              replay_launch_saving_us=0.0, seed=3)
        runner = Runner(ctx, provider or FakeProvider(), backend, proto,
                        telemetry=telemetry, formal=False,
                        allow_unverified=allow_unverified,
                        stop_after_blocks=stop)
        summary = runner.run(list(CASES))
        ctx.finish("paused" if summary.get("paused") else "done")
        return ctx, backend, summary

    # 1 ---------------------------------------------------------------
    def test_same_n_for_both_strategies_and_blocks(self):
        proto = make_protocol(blocks=6)
        ctx, backend, _ = self._run(proto)
        rows = [r for r in ctx.read_table("timings")]
        self.assertEqual(len(rows), 2 * 6 * 2)
        by_block = {}
        for r in rows:
            by_block.setdefault(r["block"], {})[r["strategy"]] = r
        ns = {r["n"] for r in rows}
        self.assertEqual(ns, {1375})  # ceil(5ms * 1.1 / 4.0us)
        for block, pair in by_block.items():
            self.assertEqual(pair["split"]["n"], pair["fused"]["n"])
            self.assertEqual(pair["split"]["n"], 1375)
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair["split"]["iteration_order"],
                             pair["fused"]["iteration_order"])
        for r in rows:
            self.assertAlmostEqual(r["us_per_call"], r["elapsed_ms"] * 1000.0 / r["n"])
            self.assertEqual(r["quality"], "ok")
        self.assertEqual(verify_run(ctx.run_dir, write=False)["integrity_pass"], True)

    # 2 ---------------------------------------------------------------
    def test_failure_retries_whole_pair_and_first_good_attempt_wins(self):
        proto = make_protocol(blocks=4)
        costs = {c: {"split": 5.0, "fused": 4.0} for c in CASES}
        provider = FakeProvider()
        # fail the first split batch issued by block 0, whatever its position
        base_prepare = provider.prepare

        def prepare(case, strategy):
            v = base_prepare(case, strategy)
            if strategy == "split":
                v.meta["fail_batch_indices"] = {0}
            return v

        provider.prepare = prepare
        ctx, backend, _ = self._run(proto, costs=costs, provider=provider)
        rows = ctx.read_table("timings")
        attempts = sorted({(r["block"], r["attempt"]) for r in rows})
        self.assertIn((0, 1), attempts)
        self.assertIn((0, 2), attempts)
        a1 = [r for r in rows if r["case_id"] == "P1-n65536-f32-contig-t"
              and r["block"] == 0 and r["attempt"] == 1]
        self.assertTrue(any(r["quality"] == "error" for r in a1))
        self.assertTrue(all(r["quality"] != "ok" for r in a1))
        a2 = [r for r in rows if r["case_id"] == "P1-n65536-f32-contig-t"
              and r["block"] == 0 and r["attempt"] == 2]
        self.assertEqual(len(a2), 2)
        self.assertTrue(all(r["quality"] == "ok" for r in a2))
        # retry must re-run both strategies, not only the failed one
        self.assertEqual({r["strategy"] for r in a2}, {"split", "fused"})
        # verify's canonical pair uses attempt 2
        v = verify_run(ctx.run_dir, write=False)
        row0 = next(r for r in v["per_case_mode"] if r["case_id"] == "P1-n65536-f32-contig-t")
        self.assertEqual(row0["valid_pairs"], 4)
        self.assertEqual(row0["attempts_used"], 2)
        self.assertEqual(row0["error_rows"], 1)

    # 3 ---------------------------------------------------------------
    def test_resume_counts_completed_pairs_once(self):
        proto = make_protocol(blocks=6)
        costs = {(cid, s): v for cid in CASES for s, v in (("split", 5.0), ("fused", 4.0))}
        ctx1, _, summary1 = self._run(proto, costs=costs, stop=2, rid="resume")
        self.assertTrue(summary1["paused"])
        rows1 = ctx1.read_table("timings")
        self.assertEqual(len(rows1), 4)  # exactly two committed pairs
        self.assertEqual(len({(r["case_id"], r["block"]) for r in rows1}), 2)

        ctx2 = open_run(self.tmp, "resume", proto)
        backend2 = FakeBackend(costs, jitter_us=0.0, seed=3)
        runner2 = Runner(ctx2, FakeProvider(), backend2, proto, telemetry=None,
                         formal=False, allow_unverified=True, stop_after_blocks=None)
        summary2 = runner2.run(list(CASES))
        ctx2.finish("done")
        self.assertFalse(summary2["paused"])
        rows = ctx2.read_table("timings")
        keys = [(r["case_id"], r["mode"], r["block"], r["attempt"], r["strategy"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(rows), 2 * 6 * 2)
        for block in (0, 1):
            attempts = [r["attempt"] for r in rows if r["block"] == block]
            self.assertEqual(attempts, [1, 1, 1, 1])
        # no completed block was timed again
        for r in rows:
            self.assertEqual(r["attempt"], 1)
        v = verify_run(ctx2.run_dir, write=False)
        self.assertTrue(v["integrity_pass"], v["problems"])
        self.assertTrue(v["complete"])

    # 4 ---------------------------------------------------------------
    def test_protocol_hash_mismatch_refuses_resume(self):
        proto = make_protocol(blocks=6)
        self._run(proto, rid="hash")
        with self.assertRaises(RunError):
            open_run(self.tmp, "hash", make_protocol(blocks=8))

    # 5 ---------------------------------------------------------------
    def test_attempt_cap_leaves_incomplete_and_keeps_failures(self):
        proto = make_protocol(blocks=2, max_attempts=2)
        provider = FakeProvider()
        base_prepare = provider.prepare

        def prepare(case, strategy):
            v = base_prepare(case, strategy)
            if strategy == "split":
                v.meta["fail_batch_indices"] = set(range(10))
            return v

        provider.prepare = prepare
        ctx, _, _ = self._run(proto, provider=provider, rid="cap")
        rows = ctx.read_table("timings")
        self.assertFalse(any(r["quality"] == "ok" for r in rows))
        # every block consumed exactly max_attempts attempts, each with an error
        for cid in CASES:
            for block in (0, 1):
                attempts = [r["attempt"] for r in rows
                            if r["case_id"] == cid and r["block"] == block]
                self.assertTrue(attempts)
                self.assertEqual(max(attempts), 2)
                self.assertEqual(min(attempts), 1)
        state = read_json(ctx.run_dir / "state" / "P1-n65536-f32-contig-t__direct.json")
        self.assertEqual(state["status"], "incomplete")
        v = verify_run(ctx.run_dir, write=False)
        self.assertEqual(v["totals"]["valid_pairs"], 0)
        self.assertEqual(v["totals"]["error_rows"], 8)

    # 6 ---------------------------------------------------------------
    def test_order_is_balanced_and_deterministic(self):
        proto = make_protocol(blocks=30)
        pats = block_patterns("P1-n65536-f32-contig-t", "direct", proto)
        self.assertEqual(pats.count("SF"), 15)
        self.assertEqual(pats.count("FS"), 15)
        self.assertEqual(pats, block_patterns("P1-n65536-f32-contig-t", "direct", proto))
        order_a = case_order(list(CASES), 17)
        order_b = case_order(list(CASES), 17)
        self.assertEqual(order_a, order_b)
        self.assertNotEqual(order_a, case_order(list(CASES), 18))

    # 7 ---------------------------------------------------------------
    def test_contamination_retries_and_is_excluded(self):
        proto = make_protocol(blocks=2)
        telemetry = TelemetryStub(contaminated_calls=1)
        ctx, _, _ = self._run(proto, telemetry=telemetry, rid="contam")
        rows = ctx.read_table("timings")
        bad = next(r for r in rows if r["quality"] == "contaminated")
        cid, block = bad["case_id"], bad["block"]
        a1 = [r for r in rows if r["case_id"] == cid and r["block"] == block
              and r["attempt"] == 1]
        self.assertTrue(any(r["quality"] == "pair_incomplete" for r in a1))
        self.assertFalse(any(r["quality"] == "ok" for r in a1))
        a2 = [r for r in rows if r["case_id"] == cid and r["block"] == block
              and r["attempt"] == 2]
        self.assertEqual(len(a2), 2)
        self.assertTrue(all(r["quality"] == "ok" for r in a2))
        self.assertEqual(verify_run(ctx.run_dir, write=False)["totals"]["valid_pairs"], 4)

    # 8 ---------------------------------------------------------------
    def test_verify_detects_tampered_derived_unit(self):
        proto = make_protocol(blocks=2)
        ctx, _, _ = self._run(proto, rid="tamper")
        rows = ctx.read_table("timings")
        rows[0]["us_per_call"] = rows[0]["us_per_call"] * 1.5
        from fb.util import write_jsonl_atomic

        write_jsonl_atomic(ctx.run_dir / "timings.jsonl", rows)
        v = verify_run(ctx.run_dir, write=False)
        self.assertFalse(v["integrity_pass"])
        self.assertTrue(any("us_per_call mismatch" in p for p in v["problems"]))

    # 8b --------------------------------------------------------------
    def test_distinct_default_is_timed_unpaired(self):
        proto = make_protocol(blocks=2)
        cases = {cid: dict(c) for cid, c in CASES.items()}
        cid = "P1-n65536-f32-contig-t"
        cases[cid]["source"] = {"default_maps_to": "distinct"}
        provider = FakeProvider(cases)
        ctx, _, _ = self._run(proto, provider=provider, rid="default")
        rows = ctx.read_table("timings")
        default_rows = [r for r in rows if r["strategy"] == "default"]
        self.assertEqual(len(default_rows), 2)
        self.assertTrue(all(r["paired"] is False for r in default_rows))
        self.assertTrue(all(r["n"] == default_rows[0]["n"] for r in default_rows))
        # the pair count is unaffected by the third, unpaired strategy
        v = verify_run(ctx.run_dir, write=False)
        self.assertEqual(v["totals"]["valid_pairs"], 4)
        self.assertTrue(v["integrity_pass"], v["problems"])

    # 9 ---------------------------------------------------------------
    def test_n_policy_units(self):
        proto = Protocol()
        self.assertEqual(proto.choose_n(5.0, 4.0), 1375)
        self.assertEqual(proto.choose_n(1000.0, 800.0), 7)
        self.assertLessEqual(proto.choose_n(1000.0, 1600.0) * 1600.0, proto.max_batch_us)
        with self.assertRaises(ValueError):
            proto.choose_n(0.0, 1.0)

    # 10 --------------------------------------------------------------
    def test_contract_rejects_unverified_formal_candidates(self):
        meta = {"artifact_sha256": None, "n_kernels": None, "preallocated": False,
                "artifacts": []}
        self.assertTrue(variant_contract_problems(meta, formal=True))
        self.assertEqual(variant_contract_problems(meta, formal=False), [])

    def test_case_selector(self):
        catalog = build_catalog()
        self.assertEqual(len(normalize_case_filter(None, catalog)), 72)
        self.assertEqual(len(normalize_case_filter("PILOT", catalog)), 2)
        self.assertEqual(len(normalize_case_filter("R1", catalog)), 24)
        self.assertEqual(len(normalize_case_filter("P1,P2", catalog)), 24)
        with self.assertRaises(ValueError):
            normalize_case_filter("nope", catalog)


if __name__ == "__main__":
    unittest.main()
