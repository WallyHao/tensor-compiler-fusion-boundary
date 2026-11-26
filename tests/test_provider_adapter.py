"""Adapter tests for Agent A's CandidateStore contract (no GPU, no torch)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from fb import fusionboundary_provider as fp
from fb.util import sha256_file, write_jsonl_atomic

CASES = [
    {
        "case_id": "P1-n1024-f32-contig-t", "template": "P1", "kind": "elementwise",
        "shape": [1024], "dtype": "float32", "layout": "contig", "edge": "t",
        "producer": "x * 1.25 + 0.25", "consumer": "relu(t)",
        "t_shape": [1024], "t_dtype": "float32",
        "out_shape": [1024], "out_dtype": "float32", "group": "P1",
        "tolerance": {"atol": 1e-6, "rtol": 1e-5}, "fp64_reference": False,
        "logical_saved_bytes": 8192, "seeds": (17, 29, 43), "zero_input": True,
        "planned_status": "planned", "exclusion_reason": None,
    },
    {
        "case_id": "R1-m16-k128-f32-contig-t", "template": "R1", "kind": "reduction",
        "shape": [16, 128], "dtype": "float32", "layout": "contig", "edge": "t",
        "producer": "sum(x, dim=1, dtype=float32)", "consumer": "relu(t)",
        "t_shape": [16], "t_dtype": "float32",
        "out_shape": [16], "out_dtype": "float32", "group": "R1",
        "tolerance": {"atol": 2e-4, "rtol": 1e-4}, "fp64_reference": True,
        "logical_saved_bytes": 128, "seeds": (17, 29, 43), "zero_input": True,
        "planned_status": "planned", "exclusion_reason": None,
    },
]

FAKE_CORPUS = '''
class Case:
    def __init__(self, row):
        self.case_id = row["case_id"]
        self.logical_save_bytes = row["logical_saved_bytes"]
        self.seeds = tuple(row["seeds"])
        self.zero_input = row["zero_input"]
        self._row = dict(row)
    def as_manifest_row(self):
        return dict(self._row)
    def torch_dtype(self):
        return "float32"

def build_corpus():
    import json, os
    here = os.path.dirname(__file__)
    rows = json.load(open(os.path.join(here, "cases.json")))
    return [Case(r) for r in rows]

def read_manifest(path):
    import json
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return [Case(r) for r in rows]
'''

FAKE_RUNTIME = '''
import json, os

class _Replay:
    def __init__(self, cand):
        self.cand = cand
    def call(self):
        return self.cand.call(self.cand.fixed_input())
    def call_n(self, n):
        pass
    def verify(self):
        return [{"input_id": "seed17", "strategy": self.cand.strategy,
                 "mode": "replay", "pass": True, "reason": None,
                 "max_abs_err": 0.0, "max_rel_err": 0.0}]

class Candidate:
    def __init__(self, case_id, strategy, artifact_dir):
        self.case_id = case_id
        self.strategy = strategy
        self.artifact_dir = artifact_dir
        self.report = type("R", (), {"kernel_count": 2 if strategy == "split" else 1,
                                     "cache_reused": True, "first_call_ms": 3.0})()
    def fixed_input(self):
        return {"x": self.case_id}
    def call(self, x=None):
        return x
    def call_n(self, n):
        pass
    def verify(self):
        return [{"input_id": f"seed{s}", "strategy": self.strategy, "pass": True,
                 "max_abs_err": 0.0, "max_rel_err": 0.0, "tolerance": {}}
                for s in (17, 29, 43)] + [
                {"input_id": "zeros", "strategy": self.strategy, "pass": True,
                 "max_abs_err": 0.0, "max_rel_err": 0.0, "tolerance": {}}]
    def capture_replay(self):
        return _Replay(self)

class CandidateStore:
    def __init__(self, run_path):
        self.run_path = run_path
        self.rows = {}
        p = os.path.join(run_path, "variants.jsonl")
        if os.path.exists(p):
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") in ("valid", "cache_only"):
                    self.rows[(r["case_id"], r["strategy"])] = r
    def status(self, cid, strategy):
        r = self.rows.get((cid, strategy))
        return None if r is None else r.get("status")
    def case_ids(self):
        return sorted({c for (c, _s) in self.rows})
    def load(self, cid, strategy, device=None):
        r = self.rows.get((cid, strategy))
        if r is None:
            raise KeyError((cid, strategy))
        return Candidate(cid, strategy, os.path.join(self.run_path, r["artifact_dir"]))
'''


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fb-adapter-"))
        pkg = self.tmp / "fake_src" / "fusionboundary"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "corpus.py").write_text(FAKE_CORPUS)
        (pkg / "runtime.py").write_text(FAKE_RUNTIME)
        (pkg / "cases.json").write_text(json.dumps(CASES))
        self.cand_run = self.tmp / "cand"
        (self.cand_run / "artifacts").mkdir(parents=True)
        rows = []
        for case in CASES:
            for strategy, status in (("split", "valid"), ("fused", "valid")):
                d = self.cand_run / "artifacts" / case["case_id"] / strategy
                d.mkdir(parents=True, exist_ok=True)
                (d / "output_code.py").write_text(f"# {case['case_id']} {strategy}\n")
                (d / "kernels.json").write_text(json.dumps({
                    "kernels": [{"launch_index": 0, "name": f"k_{strategy}",
                                 "grid": [1], "num_warps": 4, "num_stages": 1,
                                 "source_sha256": "x"}]
                }))
                rows.append({
                    "case_id": case["case_id"], "strategy": strategy,
                    "status": status,
                    "artifact_dir": f"artifacts/{case['case_id']}/{strategy}",
                    "output_code": f"artifacts/{case['case_id']}/{strategy}/output_code.py",
                    "output_code_sha256": sha256_file(d / "output_code.py"),
                    "kernel_count": 2 if strategy == "split" else 1,
                    "kernel_names": [f"k_{strategy}"],
                    "default_maps_to": "split",
                    "compile_ms": 10.0,
                })
        write_jsonl_atomic(self.cand_run / "variants.jsonl", rows)
        write_jsonl_atomic(self.cand_run / "cases.jsonl", CASES)
        self._orig = fp._ensure_importable
        fp._ensure_importable = lambda root: sys.path.insert(0, str(pkg.parent))

    def tearDown(self):
        fp._ensure_importable = self._orig
        for name in list(sys.modules):
            if name.startswith("fusionboundary"):
                del sys.modules[name]
        shutil.rmtree(self.tmp, ignore_errors=True)

    def provider(self):
        return fp.FusionBoundaryProvider(
            self.cand_run, {"device": "cuda", "verify_launch": False})

    def test_catalog_carries_contract_fields_and_status(self):
        p = self.provider()
        catalog = p.catalog()
        self.assertEqual(len(catalog), 2)
        for c in catalog:
            self.assertEqual(c["planned_status"], "planned")
            self.assertEqual(c["strategies"], ["split", "fused"])
            self.assertEqual(c["input_seeds"], [17, 29, 43, 0])
            self.assertEqual(c["logical_bytes"]["t_write"],
                             c["logical_saved_bytes"] // 2)

    def test_missing_variant_marks_case_excluded(self):
        rows = [json.loads(l) for l in (self.cand_run / "variants.jsonl").read_text().splitlines()]
        rows = [r for r in rows if not (r["case_id"] == CASES[0]["case_id"] and r["strategy"] == "fused")]
        write_jsonl_atomic(self.cand_run / "variants.jsonl", rows)
        p = self.provider()
        case = next(c for c in p.catalog() if c["case_id"] == CASES[0]["case_id"])
        self.assertEqual(case["planned_status"], "excluded")
        self.assertIn("fused=None", case["exclusion_reason"])
        self.assertEqual(case["strategies"], ["split"])

    def test_prepare_metadata_and_checks(self):
        p = self.provider()
        case = p.catalog()[0]
        v = p.prepare(case, "split")
        self.assertEqual(v.meta["n_kernels"], 2)
        self.assertTrue(v.meta["preallocated_inputs"])
        self.assertTrue(v.meta["artifact_sha256"])
        self.assertEqual(v.meta["default_maps_to"], "split")
        # measurement input ids are mapped to A's seed ids
        row = v.check("17")
        self.assertTrue(row["pass"])
        self.assertNotIn("input_id", row)
        replay, capture_ms = v._replay()
        self.assertEqual(replay.meta["submit_mode"], "replay")
        self.assertTrue(replay.check("seed17")["pass"])
        self.assertGreaterEqual(capture_ms, 0.0)

    def test_export_provenance_copies_artifacts(self):
        p = self.provider()
        out = self.tmp / "measure"
        rows = p.export_provenance(out, "r1", [c["case_id"] for c in p.catalog()])
        self.assertEqual(len(rows), 4)
        for r in rows:
            path = out / r["artifact_path"]
            self.assertTrue(path.is_dir(), r)
            self.assertFalse(path.is_symlink())
            case = next(c for c in p.catalog() if c["case_id"] == r["case_id"])
            self.assertEqual(r["artifact_hash"], p.prepare(case, r["strategy"]).meta["artifact_sha256"])
            self.assertTrue((path / "output_code.py").exists())
            self.assertTrue(r["artifact_sha256"])
            self.assertEqual(r["static_launch_count"], r["kernel_count"])


if __name__ == "__main__":
    unittest.main()
