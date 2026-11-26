"""Corpus invariants: 72 cases, template counts, ids, edge metadata, bytes."""

from __future__ import annotations

import json
import os

import pytest

from fusionboundary import corpus


def test_case_count_and_template_counts():
    cases = corpus.build_corpus()
    assert len(cases) == 72
    counts = {t: sum(1 for c in cases if c.template == t) for t in ("P1", "P2", "R1", "R2")}
    assert counts == {"P1": 12, "P2": 12, "R1": 24, "R2": 24}
    assert corpus.expected_counts()["total"] == 72


def test_ids_unique_and_named_as_contract():
    index = corpus.corpus_index()
    assert len(index) == len(corpus.build_corpus())
    assert "P1-n65536-f32-contig-t" in index
    assert "R1-m256-k512-f32-contig-t" in index
    assert "R2-m4096-k8192-f16-contig-t" in index


def test_grid_matches_work_order():
    index = corpus.corpus_index()
    ns = {index[f"P1-n{n}-f32-contig-t"].shape[0] for n in corpus.ELEMENTWISE_NS}
    assert ns == set(corpus.ELEMENTWISE_NS)
    shapes = {(index[f"R1-m{m}-k{k}-f32-contig-t"].shape) for m in corpus.REDUCTION_MS
              for k in corpus.REDUCTION_KS}
    assert shapes == {(m, k) for m in corpus.REDUCTION_MS for k in corpus.REDUCTION_KS}


def test_edge_metadata():
    p1 = corpus.get_case("P1-n65536-f32-contig-t")
    assert p1.edge == "t"
    assert p1.t_shape == (65536,)
    assert p1.t_dtype == "float32"
    assert p1.logical_save_bytes == 2 * 65536 * 4

    r1_f16 = corpus.get_case("R1-m256-k512-f16-contig-t")
    assert r1_f16.t_dtype == "float32"          # sum accumulates in fp32
    assert r1_f16.logical_save_bytes == 2 * 256 * 4
    r2_f16 = corpus.get_case("R2-m256-k512-f16-contig-t")
    assert r2_f16.t_dtype == "float16"          # amax keeps the input type
    assert r2_f16.logical_save_bytes == 2 * 256 * 2


def test_tolerances_follow_the_table():
    assert corpus.get_case("P1-n1024-f32-contig-t").atol == 1e-6
    assert corpus.get_case("P1-n1024-f16-contig-t").atol == 2e-3
    assert corpus.get_case("R1-m16-k128-f32-contig-t").atol == 2e-4
    assert corpus.get_case("R2-m16-k128-f32-contig-t").atol == 1e-6
    assert corpus.get_case("R2-m16-k128-f16-contig-t").atol == 2e-3


def test_seeds_and_zero_input():
    for case in corpus.build_corpus():
        assert case.seeds == (17, 29, 43)
        assert case.zero_input is True


def test_pilots_are_in_corpus():
    pilots = corpus.pilot_cases()
    assert [c.case_id for c in pilots] == ["P1-n65536-f32-contig-t", "R1-m256-k512-f32-contig-t"]
    assert pilots[0].template == "P1" and pilots[1].template == "R1"


def test_manifest_round_trip(tmp_path):
    path = str(tmp_path / "cases.jsonl")
    summary = corpus.write_manifest(path)
    assert summary["cases"] == 72
    assert summary["by_template"] == {"P1": 12, "P2": 12, "R1": 24, "R2": 24}
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(rows) == 72
    assert {"case_id", "template", "shape", "dtype", "layout", "edge", "seeds",
            "t_shape", "t_dtype", "logical_save_bytes", "planned_status"} <= set(rows[0])
    assert all(r["planned_status"] == "planned" for r in rows)
    back = corpus.read_manifest(path)
    assert [c.case_id for c in back] == [c.case_id for c in corpus.build_corpus()]
    assert back[0].atol == corpus.build_corpus()[0].atol
    assert os.path.getsize(path) > 0


def test_unknown_case_raises():
    with pytest.raises(KeyError):
        corpus.get_case("P1-n1234-f32-contig-t")


def test_manifest_carries_byte_breakdown_and_zero_output():
    row = corpus.get_case("R1-m256-k512-f16-contig-t").as_manifest_row()
    assert row["logical_t_write_bytes"] == 256 * 4
    assert row["logical_t_read_bytes"] == 256 * 4
    assert row["logical_save_bytes"] == row["logical_t_write_bytes"] + row["logical_t_read_bytes"]
    assert row["expected_zero_output"] == 0.0
    assert corpus.get_case("P2-n1024-f32-contig-t").as_manifest_row()["expected_zero_output"] == 0.25


def test_zero_output_claim_matches_eager():
    """The manifest states y on the all-zero input; check it against eager."""
    import torch

    from fusionboundary import templates

    subset = [corpus.get_case(f"{t}-n1024-f32-contig-t") for t in ("P1", "P2")]
    subset += [corpus.get_case(f"{t}-m16-k128-f32-contig-t") for t in ("R1", "R2")]
    subset += [corpus.get_case(f"{t}-m16-k128-f16-contig-t") for t in ("R1", "R2")]
    for case in subset:
        x = torch.zeros(case.shape, dtype=case.torch_dtype())
        y = templates.reference(case, x)
        expected = case.extras["expected_zero_output"]
        assert torch.allclose(y, torch.full_like(y, expected), atol=1e-6), case.case_id
        assert y.dtype == case.t_torch_dtype()
