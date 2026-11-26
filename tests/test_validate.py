"""Input generation, tolerance checks, fp64 diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from fusionboundary import corpus, templates, validate


def test_iter_inputs_fixed_and_device_independent():
    case = corpus.get_case("P1-n1024-f32-contig-t")
    first = dict(validate.iter_inputs(case, "cpu"))
    second = dict(validate.iter_inputs(case, "cpu"))
    assert list(first) == ["seed17", "seed29", "seed43", "zeros"]
    for key in first:
        assert torch.equal(first[key], second[key]), key
    assert first["zeros"].abs().sum().item() == 0
    assert first["seed17"].dtype == torch.float32
    assert first["seed17"].abs().max().item() <= 1.0


def test_iter_inputs_respects_case_dtype():
    case = corpus.get_case("P2-n1024-f16-contig-t")
    inputs = dict(validate.iter_inputs(case, "cpu"))
    assert inputs["seed17"].dtype == torch.float16
    assert inputs["zeros"].dtype == torch.float16


def test_seeds_differ():
    case = corpus.get_case("P1-n1024-f32-contig-t")
    inputs = dict(validate.iter_inputs(case, "cpu"))
    assert not torch.equal(inputs["seed17"], inputs["seed29"])


def test_exact_match_passes():
    ref = torch.linspace(-1, 1, 100)
    ok, reason = validate.compare(ref.clone(), ref, 1e-6, 1e-5)
    assert ok and reason == "ok"
    stats = validate.error_stats(ref.clone(), ref)
    assert stats.max_abs == 0.0 and stats.max_rel == 0.0


def test_tolerance_violation_fails_with_evidence():
    ref = torch.zeros(1000)
    ref[7] = 1.0
    out = ref.clone()
    out[7] = 1.0 + 1e-3
    ok, reason = validate.compare(out, ref, 1e-6, 1e-5)
    assert not ok
    assert "max_abs" in reason
    stats = validate.error_stats(out, ref)
    assert stats.argmax_abs == (7,)
    assert stats.max_abs == pytest.approx(1e-3, rel=1e-4)


def test_rtol_relative_band_admits_large_values():
    ref = torch.full((10,), 1000.0)
    out = ref * (1 + 5e-6)
    assert validate.compare(out, ref, 1e-6, 1e-5)[0]
    assert not validate.compare(out * (1 + 1e-3), ref, 1e-6, 1e-5)[0]


def test_shape_and_dtype_mismatch_fail():
    ref = torch.zeros(4)
    assert not validate.compare(torch.zeros(5), ref, 1e-6, 1e-5)[0]
    assert "shape mismatch" in validate.compare(torch.zeros(5), ref, 1e-6, 1e-5)[1]
    half = torch.zeros(4, dtype=torch.float16)
    ok, reason = validate.compare(half, ref, 1e-6, 1e-5)
    assert not ok and "dtype mismatch" in reason


def test_nonfinite_count_mismatch_fails():
    ref = torch.zeros(4)
    out = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
    ok, reason = validate.compare(out, ref, 1e-6, 1e-5)
    assert not ok and "non-finite" in reason


def test_check_case_against_eager_reference():
    case = corpus.get_case("R1-m16-k128-f32-contig-t")
    rows = validate.check_case(case, lambda x: templates.reference(case, x), "reference", "cpu")
    assert len(rows) == 4
    assert all(r["pass"] for r in rows)
    assert all(r["reference"] == "eager_same_dtype" for r in rows)
    assert rows[0]["tolerance"] == {"atol": 2e-4, "rtol": 1e-4}
    # fp64 diagnostics are recorded for reductions only
    assert rows[0]["fp64"]["accumulation_bound"] == pytest.approx(8 * torch.finfo(torch.float32).eps * 128)
    assert "fp64" not in validate.check_case(
        corpus.get_case("P1-n1024-f32-contig-t"),
        lambda x: templates.reference(corpus.get_case("P1-n1024-f32-contig-t"), x),
        "reference", "cpu")[0]


def test_check_case_detects_wrong_implementation():
    case = corpus.get_case("P1-n1024-f32-contig-t")
    rows = validate.check_case(case, lambda x: x * 1.25, "broken", "cpu")
    assert any(not r["pass"] for r in rows)


def test_reduction_reference64_reduces_error():
    case = corpus.get_case("R1-m256-k512-f32-contig-t")
    x = next(iter(validate.iter_inputs(case, "cpu")))[1]
    ref32 = templates.reference(case, x)
    ref64 = templates.reference64(case, x)
    assert ref64.dtype == torch.float64
    e32 = validate.error_stats(ref32.double(), ref64)
    assert 0.0 < e32.max_abs < 1e-3


def test_naive_bound_matches_definition():
    case = corpus.get_case("R1-m16-k8192-f32-contig-t")
    eps = torch.finfo(torch.float32).eps
    assert validate.naive_reduction_bound(case) == pytest.approx((8192 * eps) / (1 - 8192 * eps))
    assert validate.naive_reduction_bound(corpus.get_case("R2-m16-k8192-f32-contig-t")) is None


def test_templates_t_dtype_rules():
    assert templates.edge_spec(corpus.get_case("R1-m16-k128-f16-contig-t")).t_dtype == "float32"
    assert templates.edge_spec(corpus.get_case("R2-m16-k128-f16-contig-t")).t_dtype == "float16"
    r1 = corpus.get_case("R1-m16-k128-f16-contig-t")
    x = torch.rand(16, 128, dtype=torch.float16)
    assert templates.reference(r1, x).dtype == torch.float32
