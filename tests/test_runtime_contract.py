"""The measurement interface contract: input lifecycle, call semantics, cache proof.

These tests use a stub callable instead of a Triton candidate, so they run on
CPU. The end-to-end load path (real Inductor cache reuse) is exercised by
`python -m fusionboundary verify` on the GPU and by `tests/test_gpu_pilot.py`.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from fusionboundary import corpus, records
from fusionboundary.runtime import Candidate, CandidateStore, LoadReport, cache_dir_snapshot, snapshot_delta


class Stub:
    """Records how many times it was called and returns a fresh tensor."""

    def __init__(self, case):
        self.case = case
        self.calls = 0
        self.last_input = None

    def __call__(self, x):
        self.calls += 1
        self.last_input = x
        return x.clone()


def _candidate(case_id="P1-n1024-f32-contig-t", strategy="split"):
    case = corpus.get_case(case_id)
    report = LoadReport(case_id=case_id, strategy=strategy, status="valid",
                        artifact_dir="/tmp/none", kernels=["k0", "k1"], kernel_count=2)
    stub = Stub(case)
    return Candidate(case, strategy, "/tmp/none", stub, report, "cpu"), stub


def test_fixed_inputs_are_stable_objects():
    cand, _ = _candidate()
    x1 = cand.fixed_input()
    x2 = cand.fixed_input()
    assert x1 is x2
    assert set(cand.inputs) == {"seed17", "seed29", "seed43", "zeros"}
    assert cand.inputs["zeros"].abs().sum().item() == 0
    assert cand.report.kernel_count == 2


def test_call_is_one_invocation_and_allocates_output():
    cand, stub = _candidate()
    before = stub.calls
    y1 = cand.call()
    assert stub.calls == before + 1
    y2 = cand.call()
    assert stub.calls == before + 2
    assert y1 is not y2
    assert torch.equal(y1, y2)


def test_call_n_drops_outputs():
    cand, stub = _candidate()
    before = stub.calls
    assert cand.call_n(5) is None
    assert stub.calls == before + 5


def test_call_latest_keeps_only_one_output():
    cand, _ = _candidate()
    a = cand.call_latest()
    b = cand.call_latest()
    assert b is not a
    assert cand._latest is b


def test_call_with_explicit_input_uses_it():
    cand, stub = _candidate()
    x = cand.fresh_inputs(101)
    cand.call(x)
    assert stub.last_input is x


def test_expected_output_and_bytes():
    cand, _ = _candidate("P1-n65536-f32-contig-t")
    assert cand.expected_output == {"shape": [65536], "dtype": "float32"}
    assert cand.logical_edge_bytes == 2 * 65536 * 4


def test_verify_runs_all_inputs_through_candidate():
    cand, stub = _candidate()
    rows = cand.verify()
    assert len(rows) == 4
    assert all(r["strategy"] == "split" for r in rows)
    assert stub.calls >= 4


def test_store_missing_variant_raises_keyerror(tmp_path):
    store = CandidateStore(str(tmp_path))
    with pytest.raises(KeyError):
        store.load("P1-n1024-f32-contig-t", "split")
    assert store.status("P1-n1024-f32-contig-t", "split") is None


def test_store_indexes_variants(tmp_path):
    records.write_jsonl(str(tmp_path / "variants.jsonl"), [
        {"case_id": "P1-n1024-f32-contig-t", "strategy": "split", "status": "valid",
         "artifact_dir": "artifacts/x"},
        {"case_id": "P1-n1024-f32-contig-t", "strategy": "fused", "status": "unsupported",
         "artifact_dir": "artifacts/y"},
    ])
    store = CandidateStore(str(tmp_path))
    assert store.status("P1-n1024-f32-contig-t", "split") == "valid"
    assert store.status("P1-n1024-f32-contig-t", "fused") is None
    assert store.case_ids() == ["P1-n1024-f32-contig-t"]
    assert set(store.variants("P1-n1024-f32-contig-t")) == {"split"}


def test_cache_snapshot_delta(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    (d / "a.py").write_text("x", encoding="utf-8")
    before = cache_dir_snapshot(str(d))
    after_same = cache_dir_snapshot(str(d))
    assert snapshot_delta(before, after_same)["reused"] is True
    (d / "b.py").write_text("y", encoding="utf-8")
    delta = snapshot_delta(before, cache_dir_snapshot(str(d)))
    assert delta["reused"] is False and delta["added"] == ["b.py"]


def test_load_report_summary_is_json_serialisable():
    cand, _ = _candidate()
    cand.report.cache_delta = {"reused": True, "added": [], "removed": [], "modified": []}
    cand.report.first_call_ms = 1.5
    text = json.dumps(cand.report.summary(), sort_keys=True)
    assert '"cache_reused": true' in text


def test_version_compatibility_ignores_local_build_tag():
    from fusionboundary.runtime import _check_compatible, _release

    assert _release("2.6.0+cu124") == _release("2.6.0") == "2.6.0"
    key = {"versions": {"torch": "2.6.0+cu124", "triton": "3.2.0",
                        "cuda": str(torch.version.cuda)}}
    _check_compatible(key)          # must not raise: same release, same CUDA runtime
    with pytest.raises(RuntimeError, match="different stack"):
        _check_compatible({"versions": {"torch": "2.5.1", "cuda": str(torch.version.cuda)}})
