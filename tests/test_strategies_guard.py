"""Split guard: the veto must hit exactly the target edge and nothing else."""

from __future__ import annotations

import pytest

from fusionboundary import corpus, templates
from fusionboundary.strategies import FusionGuard, _make_interceptor, _node_names


class FakeNode:
    """Duck-typed scheduler node: enough for the matching logic."""

    def __init__(self, name, writes, reads, aten):
        self.name = name
        self.writes = set(writes)
        self.reads = set(reads)
        self.aten = aten

    def get_name(self):
        return f"{self.name}: {self.aten}"

    def get_outputs(self):
        return sorted(self.writes)

    def get_read_names(self):
        return sorted(self.reads)


@pytest.fixture
def spec():
    return templates.edge_spec(corpus.get_case("P1-n65536-f32-contig-t"))


def _nodes():
    mul = FakeNode("buf0", ["buf0"], ["arg0_1"], "aten.mul.Tensor")
    add = FakeNode("buf1", ["buf1"], ["buf0"], "aten.add.Tensor")
    relu = FakeNode("buf2", ["buf2"], ["buf1"], "aten.relu.default")
    other = FakeNode("buf9", ["buf9"], ["arg0_1"], "aten.clone.default")
    return mul, add, relu, other


def test_node_names_descriptor(spec):
    mul, add, relu, _ = _nodes()
    names = _node_names(add)
    assert any("add" in n for n in names)
    assert "buf1" in names


def test_guard_matches_only_the_target_edge(spec):
    guard = FusionGuard(spec=spec, case_id="c", strategy="split")
    mul, add, relu, other = _nodes()
    assert guard.should_block(add, relu) is not None
    assert guard.should_block(mul, add) is None          # producer side stays fused
    assert guard.should_block(add, other) is None        # no dependency on t
    assert guard.should_block(mul, relu) is None         # no direct dependency


def test_guard_matches_both_argument_orders(spec):
    guard = FusionGuard(spec=spec, case_id="c", strategy="split")
    _mul, add, relu, _other = _nodes()
    assert guard.should_block(add, relu) is not None
    assert guard.should_block(relu, add) is not None


def test_guard_rejects_wrong_op_pattern(spec):
    guard = FusionGuard(spec=spec, case_id="c", strategy="split")
    prod = FakeNode("buf1", ["buf1"], ["arg0_1"], "aten.add.Tensor")
    cons = FakeNode("buf2", ["buf2"], ["buf1"], "aten.clone.default")
    assert guard.should_block(prod, cons) is None


def test_veto_returns_false_and_logs_once(spec):
    guard = FusionGuard(spec=spec, case_id="P1-n65536-f32-contig-t", strategy="split")

    class Sched:
        def can_fuse_vertical(self, a, b):
            return True

    orig = Sched.can_fuse_vertical
    patched = _make_interceptor("can_fuse_vertical", orig, guard)
    _mul, add, relu, other = _nodes()
    sched = Sched()
    assert patched(sched, add, relu) is False
    assert len(guard.log) == 1
    assert guard.log[0]["blocked"] is True
    assert guard.log[0]["gate"] == "can_fuse_vertical"
    assert guard.log[0]["consumer_reads"] == ["buf1"]
    # unrelated pair passes straight through to the original
    assert patched(sched, add, other) is True
    assert len(guard.log) == 1


def test_veto_records_producer_and_consumer_names(spec):
    guard = FusionGuard(spec=spec, case_id="c", strategy="split")

    class Sched:
        def can_fuse_vertical(self, a, b):
            return True

    patched = _make_interceptor("can_fuse_vertical", Sched.can_fuse_vertical, guard)
    _mul, add, relu, _other = _nodes()
    patched(Sched(), add, relu)
    entry = guard.log[0]
    assert any("buf1" in n for n in entry["producer_names"])
    assert any("buf2" in n for n in entry["consumer_names"])


def test_interceptor_restores_original():
    sched_mod = pytest.importorskip("torch._inductor.scheduler")
    from fusionboundary.strategies import split_guard

    case = corpus.get_case("R1-m256-k512-f32-contig-t")
    Scheduler = sched_mod.Scheduler
    before = Scheduler.can_fuse_vertical if hasattr(Scheduler, "can_fuse_vertical") else Scheduler.fuse_nodes
    attr = "can_fuse_vertical" if hasattr(Scheduler, "can_fuse_vertical") else "fuse_nodes"
    with split_guard(case, "split") as guard:
        assert getattr(Scheduler, attr) is not before
        assert guard.patched_symbols == [f"torch._inductor.scheduler.Scheduler.{attr}"]
    assert getattr(Scheduler, attr) is before


def test_buffer_anchored_veto_is_primary(spec):
    """The anchor is the buffer lowering materialised, not the op names."""
    # the fake producer writes buf1 and the consumer reads it
    edges = {"buf1"}
    guard = FusionGuard(spec=spec, case_id="c", strategy="split",
                        edge_buffers=lambda: edges)
    _mul, add, relu, _other = _nodes()
    rec = guard.should_block(add, relu)
    assert rec is not None and rec["match"] == "materialized_buffer"
    assert rec["edge_buffers"] == ["buf1"]


def test_buffer_anchor_ignores_other_buffers(spec):
    """A different materialised buffer must not be vetoed even if op names match."""
    guard = FusionGuard(spec=spec, case_id="c", strategy="split",
                        edge_buffers=lambda: {"buf7"})
    _mul, add, relu, _other = _nodes()
    assert guard.should_block(add, relu) is None


def test_op_name_fallback_without_anchor(spec):
    guard = FusionGuard(spec=spec, case_id="c", strategy="split")
    _mul, add, relu, _other = _nodes()
    rec = guard.should_block(add, relu)
    assert rec is not None and rec["match"] == "op_name"


def test_find_edge_fx_nodes_from_graph():
    """Producer/consumer identification on a synthetic post-grad FX graph."""
    import torch
    import torch.fx as fx

    from fusionboundary.strategies import _find_edge_fx_nodes

    graph = fx.Graph()
    x = graph.placeholder("x")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (x, 1.25))
    add = graph.call_function(torch.ops.aten.add.Tensor, (mul, 0.25))
    relu = graph.call_function(torch.ops.aten.relu.default, (add,))
    graph.output(relu)
    gm = fx.GraphModule({}, graph)

    spec_p1 = templates.edge_spec(corpus.get_case("P1-n1024-f32-contig-t"))
    found = _find_edge_fx_nodes(gm.graph, spec_p1)
    assert found["producer"] == add.name
    assert found["consumer"] == relu.name
    assert found["producer_target"] == "aten.add.Tensor"

    # P2/R2 look for the first mul/add consumer and take its input as producer.
    graph2 = fx.Graph()
    x2 = graph2.placeholder("x")
    tanh = graph2.call_function(torch.ops.aten.tanh.default, (x2,))
    m = graph2.call_function(torch.ops.aten.mul.Tensor, (tanh, 0.75))
    a = graph2.call_function(torch.ops.aten.add.Tensor, (m, 0.25))
    graph2.output(a)
    gm2 = fx.GraphModule({}, graph2)
    spec_p2 = templates.edge_spec(corpus.get_case("P2-n1024-f32-contig-t"))
    found2 = _find_edge_fx_nodes(gm2.graph, spec_p2)
    assert found2["producer"] == tanh.name
    assert found2["consumer"] == m.name
