"""Forensics against synthetic Inductor artifacts (never experimental evidence)."""

from __future__ import annotations

import os

import pytest

from fusionboundary import corpus, forensics, templates

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic")


@pytest.fixture(scope="module")
def p1_case():
    return corpus.get_case("P1-n65536-f32-contig-t")


@pytest.fixture(scope="module")
def fused(p1_case):
    return forensics.parse_output_code(os.path.join(FIX, "output_code_fused.py"))


@pytest.fixture(scope="module")
def split(p1_case):
    return forensics.parse_output_code(os.path.join(FIX, "output_code_split.py"))


def test_parse_fused_kernel(fused):
    assert len(fused.kernels) == 1
    k = fused.kernels[0]
    assert k.name == "triton_poi_fused_add_mul_relu_0"
    assert k.launch_index == 0
    assert set(k.source_nodes) == {"add", "mul", "relu"}
    assert k.loads == {"in_ptr0"} and k.stores == {"out_ptr0"}
    assert k.writes == ["buf0"] and k.reads == ["arg0_1"]
    assert k.grid == "grid(65536)"
    assert k.grid_dims == 1
    assert fused.allocs["buf0"].shape == (65536,)
    assert fused.allocs["buf0"].dtype == "float32"


def test_fused_edge_evidence(p1_case, fused):
    ev = forensics.edge_evidence(p1_case, fused, "fused", templates.edge_spec(p1_case))
    assert ev["verdict"] == "fused"
    assert ev["t_materialized"] is False
    assert ev["kernel_count"] == 1
    assert len(ev["fused_kernels_crossing_edge"]) == 1
    assert ev["fused_kernels_crossing_edge"][0]["kernel"] == "triton_poi_fused_add_mul_relu_0"
    assert ev["matched_t_flows"] == []


def test_split_edge_evidence(p1_case, split):
    ev = forensics.edge_evidence(p1_case, split, "split", templates.edge_spec(p1_case))
    assert ev["verdict"] == "materialized"
    assert ev["t_materialized"] is True
    assert ev["t_alloc_dtype"] == "float32"
    assert ev["t_alloc_shape"] == [65536]
    flow = ev["matched_t_flows"][0]
    assert flow["buffer"] == "buf1"
    assert flow["writer_kernel"] == "triton_poi_fused_add_mul_0"
    assert flow["reader_kernel"] == "triton_poi_fused_relu_1"
    assert flow["writer_index"] == 0 and flow["reader_index"] == 1
    assert flow["producer_op_match"] and flow["consumer_op_match"]
    assert any("tl.store(out_ptr0" in s["text"] for s in flow["t_write_site"])
    assert any("tl.load(in_ptr0" in s["text"] for s in flow["t_read_site"])
    assert any("triton_poi_fused_add_mul_0.run" in s["text"] for s in flow["wrapper_write_call"])
    assert any("triton_poi_fused_relu_1.run" in s["text"] for s in flow["wrapper_read_call"])
    assert flow["alloc_line"] and "empty_strided" in flow["alloc_line"][0]["text"]


def test_split_has_two_kernels_in_order(split):
    assert [k.name for k in sorted(split.kernels, key=lambda k: k.launch_index)] == [
        "triton_poi_fused_add_mul_0", "triton_poi_fused_relu_1"]


def test_compare_partitions(fused, split):
    diff = forensics.compare_partitions(split, fused)
    assert diff["kernel_count_a"] == 2 and diff["kernel_count_b"] == 1
    assert diff["shared_kernel_count"] == 0
    assert len(diff["only_in_a"]) == 2 and len(diff["only_in_b"]) == 1


def test_scheduler_dump_parse():
    dump = forensics.parse_scheduler_dump(os.path.join(FIX, "ir_post_fusion.txt"))
    assert dump["parsed"] is True
    assert dump["kernel_names"] == ["triton_poi_fused_add_mul_0", "triton_poi_fused_relu_1"]
    assert dump["buffer_writes"]["buf1"] == ["buf1"]
    assert dump["buffer_users"]["buf1"] == ["buf2"]


def test_missing_dump_is_reported_not_raised():
    dump = forensics.parse_scheduler_dump("/nonexistent/ir_post_fusion.txt")
    assert dump["parsed"] is False and "missing file" in dump["notes"]


def test_buffer_flows_identifies_intermediate(fused, split):
    assert forensics.buffer_flows(fused) == []
    flows = forensics.buffer_flows(split)
    assert len(flows) == 1
    assert flows[0]["buffer"] == "buf1"
    assert flows[0]["alloc"]["dtype"] == "float32"
    assert [w["kernel"] for w in flows[0]["writers"]] == ["triton_poi_fused_add_mul_0"]
    assert [r["kernel"] for r in flows[0]["readers"]] == ["triton_poi_fused_relu_1"]


def test_collect_trace_artifacts(tmp_path):
    out = tmp_path / "sampler.py"
    out.write_text("x = 1\n", encoding="utf-8")
    inv = forensics.collect_trace_artifacts(str(tmp_path))
    assert inv["output_code"] is None
    assert inv["files"]


def test_buffer_reuse_alias_path(p1_case):
    """The pattern torch 2.6 emits: t's storage handed to the consumer in place."""
    parsed = forensics.parse_output_code(os.path.join(FIX, "output_code_split_reuse.py"))
    assert parsed.aliases == {"buf1": "buf0"}
    assert sorted(parsed.allocs) == ["buf0"]
    flows = forensics.buffer_flows(parsed)
    assert [f["buffer"] for f in flows] == ["buf0"]
    flow = flows[0]
    assert [w["kernel"] for w in flow["writers"]] == ["triton_poi_fused_add_mul_0",
                                                      "triton_poi_fused_relu_1"]
    assert [r["kernel"] for r in flow["readers"]] == ["triton_poi_fused_relu_1"]
    assert flow["in_place_read"] is True
    ev = forensics.edge_evidence(p1_case, parsed, "split", templates.edge_spec(p1_case))
    assert ev["verdict"] == "materialized"
    matched = ev["matched_t_flows"][0]
    assert matched["writer_kernel"] == "triton_poi_fused_add_mul_0"
    assert matched["reader_kernel"] == "triton_poi_fused_relu_1"
    assert matched["in_place_read"] is True
    assert any("in_out_ptr0" in s["text"] for s in matched["t_read_site"])
    assert any("buf1 = buf0" in s["text"] for s in matched["wrapper_reuse_line"])
    assert ev["t_alloc_shape"] == [65536]


def test_same_kernel_inout_is_not_materialisation():
    """A persistent reduction reading and writing its own output buffer is fused.

    The buffer has t's shape and is read+written, but by one kernel: that must not
    count as `t_materialized` (otherwise every fused reduction would look split).
    """
    case = corpus.get_case("R1-m256-k512-f32-contig-t")
    parsed = forensics.parse_output_code(
        os.path.join(FIX, "output_code_fused_inout_reduction.py"))
    assert parsed.aliases == {"buf1": "buf0"}
    flows = forensics.buffer_flows(parsed)
    assert len(flows) == 1
    flow = flows[0]
    assert [w["kernel"] for w in flow["writers"]] == ["triton_per_fused_relu_sum_0"]
    assert [r["kernel"] for r in flow["readers"]] == ["triton_per_fused_relu_sum_0"]
    assert flow["in_place_read"] is True

    ev = forensics.edge_evidence(case, parsed, "fused", templates.edge_spec(case))
    assert ev["verdict"] == "fused"
    assert ev["t_materialized"] is False
    assert ev["matched_t_flows"] == []
    assert len(ev["fused_kernels_crossing_edge"]) == 1
