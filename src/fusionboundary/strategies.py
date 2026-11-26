"""default / split / fused compilation of the same single graph.

Invariants (work order plan/phase1-3090/02-candidates.md):

* every strategy compiles the *same* function object with the *same* Inductor
  config; the only difference is the fusion-admission guard;
* ``default`` installs no guard and is kept as evidence of Inductor's own
  decision. When the default partition already equals one of the two candidate
  partitions, that artifact is reused for that strategy and the mapping is
  recorded instead of compiling a third time;
* ``split`` refuses fusion exactly across the matched ``t`` edge in the
  scheduler, so ``t`` is materialised and the consumer launches as its own
  kernel inside the same compiled wrapper (no extra Python call, no graph
  break);
* ``fused`` allows the merge across the edge.

Diagnostic controls (``diag_clone``, ``diag_graph_break``, ``diag_no_fusion``)
are archived separately with the reason they are not the main candidate.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import forensics, records, templates

MAIN_STRATEGIES = ("default", "split", "fused")
DIAG_STRATEGIES = ("diag_clone", "diag_graph_break", "diag_no_fusion")

#: Inductor knobs pinned for every compilation. Stored verbatim in
#: ``compile_key.json`` and re-applied when loading the archived wrapper.
DEFAULT_OVERRIDES: dict[str, Any] = {
    "max_autotune": False,
    "max_autotune_gemm": False,
    "triton.cudagraphs": False,
    "triton.autotune_pointwise": False,
    "benchmark_harness": False,
}


# --------------------------------------------------------------------------
# config context
# --------------------------------------------------------------------------
#: Overrides that this torch version does not expose (recorded per variant).
_LAST_SKIPPED: list[str] = []


def last_skipped_overrides() -> list[str]:
    return list(_LAST_SKIPPED)


@contextlib.contextmanager
def inductor_environment(trace_dir: str | None = None,
                         overrides: dict[str, Any] | None = None):
    """Pin the Inductor knobs that affect partitioning; restore them after."""
    import torch._inductor.config as cfg

    saved: list[tuple[Any, str, Any]] = []
    skipped: list[str] = []
    wanted: dict[str, Any] = dict(DEFAULT_OVERRIDES)
    if trace_dir is not None:
        wanted["trace.enabled"] = True
        wanted["trace.debug_dir"] = trace_dir
    if overrides:
        wanted.update(overrides)
    try:
        for dotted, value in wanted.items():
            owner: Any = cfg
            parts = dotted.split(".")
            try:
                for p in parts[:-1]:
                    owner = getattr(owner, p)
                old = getattr(owner, parts[-1])
            except AttributeError:
                # A knob removed/renamed in this torch version must be visible in
                # the record, not silently assumed to be the pinned value.
                skipped.append(dotted)
                continue
            saved.append((owner, parts[-1], old))
            setattr(owner, parts[-1], value)
        _LAST_SKIPPED[:] = skipped
        yield cfg
    finally:
        for owner, key, old in reversed(saved):
            setattr(owner, key, old)


# --------------------------------------------------------------------------
# scheduler-level fusion guard
# --------------------------------------------------------------------------
#: Vertical-fusion admission points to try, in order. ``index`` is recorded in
#: every split variant; the driver retries with the next entry (fresh process,
#: fresh cache) when the previous one provably did not fire.
_GATE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("Scheduler", "can_fuse_vertical"),
    ("SchedulerNode", "can_fuse_vertical"),
)


def _iter_subnodes(node: Any):
    if hasattr(node, "get_nodes"):
        try:
            return list(node.get_nodes())
        except Exception:
            pass
    return [node]


def _node_names(node: Any) -> list[str]:
    """Human-readable descriptor strings of a scheduler node (for matching/logs)."""
    names: list[str] = []
    for sub in _iter_subnodes(node):
        for attr in ("get_name", "get_operation_name"):
            fn = getattr(sub, attr, None)
            if callable(fn):
                try:
                    value = fn()
                except Exception:
                    continue
                if value:
                    names.append(str(value))
        origin = getattr(getattr(sub, "node", None), "data", None)
        target = getattr(origin, "target", None)
        if target is not None:
            names.append(str(target))
        names.extend(sorted(str(b) for b in _buffer_names(sub)))
    uniq: list[str] = []
    for n in names:
        if n and n not in uniq:
            uniq.append(n)
    return uniq


_BUF_TOKEN_RE = __import__("re").compile(r"\b(?:buf|arg|reinterpret_tensor)\d*(?:_\d+)?\b")


def _name_of(value: Any) -> str:
    """Best name of a scheduler node / IR node / buffer, tolerant to versions."""
    if isinstance(value, str):
        return value
    for attr in ("name",):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    fn = getattr(value, "get_name", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    text = str(value)
    tokens = _BUF_TOKEN_RE.findall(text)
    return tokens[0] if tokens else text


def _buffer_names(node: Any) -> set[str]:
    """Buffer names a scheduler node writes.

    Every available accessor is unioned instead of taking the first non-empty
    one, because their return types differ across versions (names vs IR nodes).
    """
    out: set[str] = set()
    for sub in _iter_subnodes(node):
        for attr in ("get_outputs", "get_buffer_names", "get_write_names"):
            fn = getattr(sub, attr, None)
            if callable(fn):
                try:
                    value = fn()
                except Exception:
                    continue
                if value:
                    out.update(_name_of(v) for v in value)
    return out


def _read_names(node: Any) -> set[str]:
    out: set[str] = set()
    for sub in _iter_subnodes(node):
        for attr in ("get_read_names", "get_inputs"):
            fn = getattr(sub, attr, None)
            if callable(fn):
                try:
                    value = fn()
                except Exception:
                    continue
                if value:
                    out.update(_name_of(v) for v in value)
        rw = getattr(sub, "read_writes", None)
        if rw is not None:
            try:
                out.update(_name_of(d) for d in rw.reads)
            except Exception:
                pass
    return out


def _text_matches(text: str, needles: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


@dataclass
class FusionGuard:
    """Veto vertical fusion across the matched producer -> consumer edge ``t``."""

    spec: Any
    case_id: str
    strategy: str
    log: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0
    edge_buffers: Callable[[], set[str]] | None = None
    patched_symbols: list[str] = field(default_factory=list)
    gate_doc: str = ""
    gate_index: int = 0

    def describe(self, node: Any) -> dict[str, Any]:
        return {
            "names": _node_names(node),
            "buffers": sorted(_buffer_names(node)),
            "reads": sorted(_read_names(node)),
        }

    def should_block(self, node1: Any, node2: Any) -> dict[str, Any] | None:
        """Return a match record when ``node1 -> node2`` is the target edge.

        Primary test: the buffer that lowering materialised for ``t`` is written
        by the producer and read by the consumer. That anchor comes from the
        lowering hook, so it does not depend on whether this version names
        scheduler nodes after their ATen ops (2.6 names them ``op0``/``op1``).
        Fallback: op-name matching, used when no buffer was materialised.
        """
        self.calls += 1
        edge = set(self.edge_buffers()) if self.edge_buffers is not None else set()
        for prod, cons in ((node1, node2), (node2, node1)):
            shared = _buffer_names(prod) & _read_names(cons)
            if not shared:
                continue
            dprod, dcons = self.describe(prod), self.describe(cons)
            if edge:
                if shared & edge:
                    rec = {"producer": dprod, "consumer": dcons, "match": "materialized_buffer",
                           "edge_buffers": sorted(shared & edge)}
                    return rec
                continue
            prod_text = " ".join(dprod["names"])
            cons_text = " ".join(dcons["names"])
            if _text_matches(prod_text, self.spec.producer_ops) and _text_matches(cons_text, self.spec.consumer_ops):
                return {"producer": dprod, "consumer": dcons, "match": "op_name"}
        return None

    def record(self, match: dict[str, Any], gate: str, blocked: bool) -> None:
        self.log.append({
            "case_id": self.case_id,
            "strategy": self.strategy,
            "gate": gate,
            "blocked": blocked,
            "edge": "t",
            "producer_names": match["producer"]["names"],
            "consumer_names": match["consumer"]["names"],
            "match": match.get("match"),
            "edge_buffers": match.get("edge_buffers"),
            "producer_buffers": match["producer"]["buffers"],
            "consumer_buffers": match["consumer"]["buffers"],
            "consumer_reads": match["consumer"]["reads"],
        })


def _resolve_gate(scheduler_mod: Any, index: int = 0) -> tuple[str, Any, Any, str]:
    """Find the vertical-fusion veto point used by this torch version.

    ``index`` selects from ``_GATE_CANDIDATES``; the CLI retries the next
    candidate in a fresh process/cache when a gate provably does not fire, so a
    version whose admission path moved still yields a split candidate instead of
    a silent no-op.
    """
    if index >= len(_GATE_CANDIDATES):
        raise RuntimeError(f"gate index {index} out of range "
                           f"({len(_GATE_CANDIDATES)} candidates)")
    cls_name, attr = _GATE_CANDIDATES[index]
    cls = getattr(scheduler_mod, cls_name, None)
    if cls is None or not hasattr(cls, attr):
        raise RuntimeError(f"torch._inductor.scheduler.{cls_name}.{attr} not found "
                           f"in this version")
    return attr, cls, getattr(cls, attr), str(getattr(getattr(cls, attr), "__doc__", "") or "")


@contextlib.contextmanager
def split_guard(case, strategy: str = "split", gate_index: int = 0,
                edge_buffers: Callable[[], set[str]] | None = None):  # noqa: ANN001
    """Install the guard for the duration of compilation + first call."""
    import torch._inductor.scheduler as sched

    guard = FusionGuard(spec=templates.edge_spec(case), case_id=case.case_id,
                        strategy=strategy, edge_buffers=edge_buffers)
    attr, cls, original, doc = _resolve_gate(sched, gate_index)
    interceptor = _make_interceptor(attr, original, guard)
    setattr(cls, attr, interceptor)
    guard.patched_symbols.append(f"torch._inductor.scheduler.{cls.__name__}.{attr}")
    guard.gate_doc = doc
    guard.gate_index = gate_index
    try:
        yield guard
    finally:
        setattr(cls, attr, original)


def _make_interceptor(attr: str, original: Callable, guard: FusionGuard) -> Callable:
    import functools

    if attr == "can_fuse_vertical":
        @functools.wraps(original)
        def veto(self, node1, node2, *args, **kwargs):  # noqa: ANN001
            match = guard.should_block(node1, node2)
            if match is not None:
                guard.record(match, attr, blocked=True)
                return False
            return original(self, node1, node2, *args, **kwargs)
        return veto

    @functools.wraps(original)
    def veto_fuse(self, node1, node2, *args, **kwargs):  # noqa: ANN001
        match = guard.should_block(node1, node2)
        if match is not None:
            guard.record(match, attr, blocked=True)
            return None
        return original(self, node1, node2, *args, **kwargs)
    return veto_fuse


# --------------------------------------------------------------------------
# lowering-level materialization request for the target edge
# --------------------------------------------------------------------------
def _find_edge_fx_nodes(graph: Any, spec: Any) -> dict[str, str]:
    """Locate producer/consumer FX nodes of the edge from the post-grad graph.

    The consumer is the first ``call_function`` node whose target matches the
    template's consumer ops. The producer is then the *closest* transitive
    ``call_function`` input whose target matches the template's producer ops;
    for templates whose consumer wraps the edge in another op (N1:
    ``rsqrt(add(mean(...), eps))``) this walks past the wrapper. When nothing
    matches the producer ops the first call_function input is used, which is
    what the pointwise and existing reduction templates already resolve to.
    """
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if not _text_matches(str(node.target), spec.consumer_ops):
            continue
        seen: set[int] = set()
        queue = list(node.all_input_nodes)
        first: Any = None
        while queue:
            candidate = queue.pop(0)
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if candidate.op != "call_function":
                continue
            if first is None:
                first = candidate
            if _text_matches(str(candidate.target), spec.producer_ops):
                return {
                    "producer": candidate.name,
                    "consumer": node.name,
                    "producer_target": str(candidate.target),
                    "consumer_target": str(node.target),
                }
            queue.extend(candidate.all_input_nodes)
        if first is not None:
            return {
                "producer": first.name,
                "consumer": node.name,
                "producer_target": str(first.target),
                "consumer_target": str(node.target),
            }
    return {}


@contextlib.contextmanager
def realize_edge_at_lowering(case, spec, log: list[dict[str, Any]]):  # noqa: ANN001
    """Ask the lowering stage to materialise ``t``.

    TorchInductor inlines a single-use pointwise producer into its consumer
    before the scheduler runs, so no scheduler-level veto can split such an
    edge. Calling ``realize()`` on the producer's TensorBox while its FX node is
    being lowered creates the intermediate buffer; the scheduler veto installed
    in parallel then keeps the consumer from being merged back across the edge.
    """
    from torch._inductor.graph import GraphLowering

    state: dict[str, Any] = {"edge": None, "realized": []}
    original = GraphLowering.run_node

    @functools.wraps(original)
    def patched(self, n):  # noqa: ANN001
        if state["edge"] is None and hasattr(self, "graph"):
            state["edge"] = _find_edge_fx_nodes(self.graph, spec)
        out = original(self, n)
        edge = state["edge"] or {}
        if getattr(n, "name", None) == edge.get("producer"):
            realize = getattr(out, "realize", None)
            entry = {"fx_node": n.name, "target": str(getattr(n, "target", "")),
                     "op": "realize"}
            try:
                buffer_name = realize() if callable(realize) else None
                entry["buffer"] = str(buffer_name) if buffer_name is not None else None
                entry["status"] = "realized" if buffer_name is not None else "not_realizable"
            except Exception as exc:
                entry["status"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            log.append(entry)
            state["realized"].append(entry)
        return out

    GraphLowering.run_node = patched
    try:
        yield state
    finally:
        GraphLowering.run_node = original


@contextlib.contextmanager
def split_context(case, gate_index: int = 0):  # noqa: ANN001
    """The split mechanism: ask lowering to materialise t, then veto its re-fusion.

    Both halves are needed: TorchInductor inlines single-use pointwise producers
    during lowering, and the scheduler may still merge the two resulting nodes
    back into one kernel.
    """
    realize_log: list[dict[str, Any]] = []

    def edge_buffers() -> set[str]:
        return {e["buffer"] for e in realize_log if e.get("buffer")}

    with realize_edge_at_lowering(case, templates.edge_spec(case), realize_log):
        with split_guard(case, "split", gate_index, edge_buffers=edge_buffers) as guard:
            yield guard, realize_log


@contextlib.contextmanager
def fusion_probe():
    """Passive counter: did the scheduler consider (and thus did it perform) the fusion?

    Combined with the final kernel count this says *where* the decision happened:
    one kernel with zero scheduler admissions means the producer was already
    inlined during lowering; one kernel with admissions means the scheduler
    merged the nodes.
    """
    import torch._inductor.scheduler as sched

    state: dict[str, Any] = {"scheduler_pairs": 0}
    Scheduler = getattr(sched, "Scheduler", None)
    if Scheduler is None or not hasattr(Scheduler, "can_fuse_vertical"):
        yield state
        return
    original = Scheduler.can_fuse_vertical

    @functools.wraps(original)
    def counted(self, node1, node2, *args, **kwargs):  # noqa: ANN001
        state["scheduler_pairs"] += 1
        return original(self, node1, node2, *args, **kwargs)

    Scheduler.can_fuse_vertical = counted
    try:
        yield state
    finally:
        Scheduler.can_fuse_vertical = original


# --------------------------------------------------------------------------
# diagnostics: other ways to break the edge (archived, not main candidates)
# --------------------------------------------------------------------------
def make_diag_fn(case, variant: str):  # noqa: ANN001
    """Return (function, overrides, note) for a diagnostic control."""
    prod, cons = templates.PRODUCER_CONSUMER[case.template]
    if variant == "diag_clone":
        def fn(x):
            t = prod(x)
            return cons(t.clone())
        return fn, {}, "explicit clone materialises t; pointwise clones are still fusible, may not split"
    if variant == "diag_graph_break":
        import torch._dynamo as dynamo

        def fn(x):
            t = prod(x)
            dynamo.graph_break()
            return cons(t)
        return fn, {}, "dynamo graph break: two compiled graphs, extra Python call between them"
    if variant == "diag_no_fusion":
        return templates.TEMPLATE_FNS[case.template], {"max_fusion_size": 0}, (
            "global fusion disable (max_fusion_size=0): changes unrelated partitions"
        )
    raise KeyError(variant)


# --------------------------------------------------------------------------
# compile a variant
# --------------------------------------------------------------------------
@dataclass
class CompileResult:
    case_id: str
    strategy: str
    status: str
    artifact_dir: str
    compiled: Any = None
    parsed: forensics.ParsedOutput | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    guard_log: list[dict[str, Any]] = field(default_factory=list)
    compile_ms: float | None = None
    first_call_ms: float | None = None
    second_call_ms: float | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    fx_graph: dict[str, Any] = field(default_factory=dict)
    gate_symbol: str | None = None
    gate_index: int = 0
    realize_log: list[dict[str, Any]] = field(default_factory=list)
    fusion_probe: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "valid"


def _capture_fx(case, fn, strategy: str = "default", gate_index: int = 0) -> dict[str, Any]:  # noqa: ANN001
    """Graph-level evidence: how many graphs dynamo produced and where it broke."""
    import torch._dynamo as dynamo

    try:
        with torch.no_grad():
            if strategy == "split":
                with split_guard(case, strategy, gate_index):
                    explain = dynamo.explain(fn)(torch.zeros(case.shape, dtype=case.torch_dtype(),
                                                            device=_current_device()))
            else:
                explain = dynamo.explain(fn)(torch.zeros(case.shape, dtype=case.torch_dtype(),
                                                        device=_current_device()))
        return {
            "strategy": strategy,
            "graph_count": int(getattr(explain, "graph_count", -1)),
            "graph_break_count": int(getattr(explain, "graph_break_count", -1)),
            "ops_per_graph": [[str(op) for op in ops] for ops in getattr(explain, "ops_per_graph", [])],
            "graph_code": getattr(explain, "graph_code", ""),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"strategy": strategy, "error": f"{type(exc).__name__}: {exc}"}


def _current_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def entry_fn(case, strategy: str):  # noqa: ANN001
    """Return (callable, config overrides, note) for a strategy name."""
    if strategy in DIAG_STRATEGIES:
        return make_diag_fn(case, strategy)
    return templates.TEMPLATE_FNS[case.template], {}, ""


# --------------------------------------------------------------------------
# static resources of the launched kernels (for the association table / x3)
# --------------------------------------------------------------------------
_RESOURCE_FIELDS = ("n_regs", "n_spills", "shared", "num_warps", "num_stages",
                    "block", "warps_per_block")


def _iter_loaded_code_modules():
    """Modules Inductor loaded from its generated code (version tolerant).

    ``PyCodeCache.modules`` is a class attribute, so both the class and the
    module namespace are inspected. Modules loaded from the active code cache
    directory are added as a fallback.
    """
    import sys

    import torch._inductor.codecache as codecache

    seen: set[int] = set()
    holders = [getattr(codecache, "PyCodeCache", None), codecache]
    for holder in holders:
        if holder is None:
            continue
        for attr in ("modules", "cache"):
            container = getattr(holder, attr, None)
            if isinstance(container, dict):
                values = list(container.values())
            elif isinstance(container, (list, tuple)):
                values = list(container)
            else:
                continue
            for mod in values:
                mod = getattr(mod, "__self__", mod)
                if mod is None or id(mod) in seen:
                    continue
                seen.add(id(mod))
                yield mod
    cache_root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    for mod in list(sys.modules.values()):
        if mod is None or id(mod) in seen:
            continue
        path = getattr(mod, "__file__", None) or ""
        if not path:
            continue
        if (cache_root and os.path.abspath(path).startswith(os.path.abspath(cache_root))) \
                or "torchinductor" in path:
            seen.add(id(mod))
            yield mod


def _read_attr(obj: Any, path: str) -> Any:
    """Read a dotted attribute path, supporting ``list[0]`` style indices."""
    cur = obj
    for part in path.split("."):
        if "[" in part and part.endswith("]"):
            name, idx = part[:-1].split("[")
            cur = getattr(cur, name)[int(idx)]
        else:
            cur = getattr(cur, part)
    return cur


_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "n_regs": ("n_regs",),
    "n_spills": ("n_spills",),
    "shared_bytes": ("metadata.shared", "shared"),
    "num_warps": ("metadata.num_warps", "config.num_warps"),
    "num_stages": ("config.num_stages",),
}
_BLOCK_PATHS = ("config.kwargs", "best_config.kwargs", "configs[0].kwargs")


def extract_kernel_resources(kernel_names: list[str]) -> dict[str, Any]:
    """Registers / shared memory / block sizes of the kernels that were launched.

    Read from the loaded Triton compiled-kernel objects (``n_regs``,
    ``n_spills``, ``metadata.shared``, autotuner ``config``), never guessed.
    Every value names the attribute path it came from; a missing value stays
    ``null`` with a reason, per the data contract.
    """
    result: dict[str, Any] = {}
    modules = list(_iter_loaded_code_modules())
    for name in kernel_names:
        entry: dict[str, Any] = {field: None for field in _FIELD_PATHS}
        entry.update({"kernel": name, "block": None, "sources": {}, "reason": None})
        found = None
        for mod in modules:
            candidate = getattr(mod, name, None)
            if candidate is not None:
                found = candidate
                break
        if found is None:
            entry["reason"] = "kernel object not found in the loaded code modules"
            result[name] = entry
            continue
        targets: list[tuple[str, Any]] = [("", found)]
        for expr in ("launchers[0]", "fn"):
            try:
                targets.append((expr + ".", _read_attr(found, expr)))
            except Exception:
                continue
        for field, paths in _FIELD_PATHS.items():
            for prefix, target in targets:
                hit = False
                for path in paths:
                    try:
                        value = _read_attr(target, path)
                    except Exception:
                        continue
                    if isinstance(value, (int, float)):
                        entry[field] = value
                        entry["sources"][field] = prefix + path
                        hit = True
                        break
                if hit:
                    break
        for prefix, target in targets:
            hit = False
            for path in _BLOCK_PATHS:
                try:
                    kwargs = _read_attr(target, path)
                except Exception:
                    continue
                if isinstance(kwargs, dict):
                    block = {k: v for k, v in kwargs.items()
                             if "BLOCK" in k and isinstance(v, int)}
                    if block:
                        entry["block"] = block
                        entry["sources"]["block"] = prefix + path
                        hit = True
                        break
            if hit:
                break
        missing = [f for f in _FIELD_PATHS if entry[f] is None]
        if missing:
            entry["reason"] = ("not exposed by this launch wrapper: "
                               + ", ".join(missing))
        result[name] = entry
        entry.update(_static_occupancy(found))
    return result


def _static_occupancy(kernel: Any) -> dict[str, Any]:
    """Calculate residency from the launched Triton kernel's CUDA function."""
    try:
        (launcher,) = kernel.launchers
        launch = launcher.__globals__
        function = launch["function"]
        warps = int(launch["num_warps"])
        shared = int(launcher.shared)
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        warp_size = props.warp_size
        max_warps = props.max_threads_per_multi_processor // warp_size
        query = ctypes.CDLL("libcuda.so.1").cuOccupancyMaxActiveBlocksPerMultiprocessor
        query.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                          ctypes.c_int, ctypes.c_size_t)
        query.restype = ctypes.c_int
        active = ctypes.c_int()
        rc = query(ctypes.byref(active), function, warps * warp_size, shared)
        if rc != 0 or active.value <= 0:
            raise RuntimeError(f"CUDA occupancy query returned {rc}, blocks={active.value}")
        return {
            "blocks_per_sm": active.value,
            "warps_per_block": warps,
            "sm_max_warps": max_warps,
            "dynamic_shared_bytes": shared,
            "occupancy": active.value * warps / max_warps,
            "occupancy_source": "cuOccupancyMaxActiveBlocksPerMultiprocessor on Triton launcher.function",
        }
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"occupancy": None, "occupancy_reason": str(exc)}


def compile_variant(case, strategy: str, artifact_dir: str, device: str | None = None,  # noqa: ANN001
                    run_trace: bool = True, gate_index: int = 0) -> CompileResult:
    """Compile one (case, strategy) pair and collect code/IR evidence.

    The caller is responsible for process-level isolation (a fresh
    ``TORCHINDUCTOR_CACHE_DIR`` per variant, set before torch is imported); the
    CLI worker does exactly that.
    """
    device = device or _current_device()
    os.makedirs(artifact_dir, exist_ok=True)
    trace_dir = os.path.join(artifact_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)

    result = CompileResult(case_id=case.case_id, strategy=strategy, status="incomplete",
                           artifact_dir=artifact_dir)

    fn, overrides, note = entry_fn(case, strategy)
    if note:
        result.notes.append(note)

    if not torch.cuda.is_available() and device == "cuda":
        result.status = "unsupported"
        result.errors.append("no CUDA device; split/fused Triton candidates need a GPU")
        return result

    # fixed input (seed 17) on device; compiled shapes are static
    from .validate import iter_inputs

    input_id, x = next(iter(iter_inputs(case, device)))
    result.notes.append(f"compile input: {input_id}")

    realize_log: list[dict[str, Any]] = []
    split_cm = (split_context(case, gate_index) if strategy == "split"
                else contextlib.nullcontext())
    try:
        with inductor_environment(trace_dir if run_trace else None, overrides) as cfg:
            result.config = records.inductor_config_snapshot()
            if last_skipped_overrides():
                result.config["skipped_overrides"] = last_skipped_overrides()
            with split_cm as split_state, fusion_probe() as probe:
                guard, realize_log = (split_state if strategy == "split" else (None, []))
                # fullgraph=True makes a silent graph break impossible; only the
                # diagnostic graph-break control is compiled permissively.
                fullgraph = strategy != "diag_graph_break"
                compiled = torch.compile(fn, backend="inductor", dynamic=False,
                                         fullgraph=fullgraph)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = compiled(x)
                torch.cuda.synchronize() if device == "cuda" else None
                t1 = time.perf_counter()
                t2 = time.perf_counter()
                with torch.no_grad():
                    out2 = compiled(x)
                torch.cuda.synchronize() if device == "cuda" else None
                t3 = time.perf_counter()
                result.compiled = compiled
                result.first_call_ms = (t1 - t0) * 1e3
                result.second_call_ms = (t3 - t2) * 1e3
                # The first call is where Inductor lowers/compiles (or resolves the
                # on-disk cache) and launches once; the two cannot be separated in
                # one process, so both are reported with that meaning.
                result.compile_ms = result.first_call_ms
                result.notes.append(f"output shape {tuple(out.shape)} dtype {out.dtype}")
                result.notes.append(f"repeat-call bitwise equal: {bool(torch.equal(out, out2))}")
                result.fusion_probe = dict(probe)
                result.realize_log = realize_log
                if realize_log:
                    result.notes.append(
                        "lowering realize: " + ", ".join(
                            f"{e['fx_node']}->{e.get('buffer')}[{e['status']}]" for e in realize_log))
                if guard is not None:
                    result.guard_log = list(guard.log)
                    result.gate_symbol = (guard.patched_symbols or [None])[0]
                    result.gate_index = guard.gate_index
                    result.notes.append(
                        f"fusion gate {guard.patched_symbols} calls={guard.calls} blocks={len(guard.log)}")
    except Exception as exc:  # pragma: no cover - failure path
        import traceback

        result.status = "compile_error"
        result.errors.append(f"{type(exc).__name__}: {exc}")
        result.notes.append(traceback.format_exc())
        _write_failure(result, artifact_dir)
        return result

    # ---- evidence from the trace directory ------------------------------
    result.trace = forensics.collect_trace_artifacts(trace_dir)
    output_code = result.trace.get("output_code")
    if not output_code:
        result.status = "compile_error"
        result.errors.append("no output_code.py in the Inductor trace directory "
                             "(trace.enabled/debug_dir did not apply)")
        _write_failure(result, artifact_dir)
        return result

    parsed = forensics.parse_output_code(output_code)
    result.parsed = parsed
    # Freeze a copy inside the artifact dir: a later dynamo.explain() in the same
    # process can rewrite the trace dir, and the archived evidence must keep the
    # bytes that were hashed and parsed here.
    frozen_code = os.path.join(artifact_dir, "output_code.py")
    records.copy_file(output_code, frozen_code)
    for key in ("ir_pre_fusion", "ir_post_fusion", "fx_graph"):
        src = result.trace.get(key)
        if src and os.path.exists(src):
            records.copy_file(src, os.path.join(artifact_dir, os.path.basename(src)))
    spec = templates.edge_spec(case)
    evidence = forensics.edge_evidence(case, parsed, strategy, spec)
    evidence["output_code_trace"] = output_code
    evidence["output_code"] = frozen_code
    evidence["partition"] = forensics.partition_signature(parsed)
    evidence["scheduler_dump"] = {
        "pre": forensics.parse_scheduler_dump(result.trace.get("ir_pre_fusion") or ""),
        "post": forensics.parse_scheduler_dump(result.trace.get("ir_post_fusion") or ""),
    }
    evidence["guard_log"] = result.guard_log
    evidence["realize_log"] = result.realize_log
    # Where the fusion decision happened, from the call count plus the kernel count.
    pairs = int((result.fusion_probe or {}).get("scheduler_pairs", 0))
    kernel_count = evidence.get("kernel_count") or 0
    if kernel_count == 1:
        stage = "scheduler" if pairs else "lowering"
    else:
        stage = "not_fused"
    evidence["fusion_decision"] = {
        "stage": stage,
        "scheduler_fusion_pairs_considered": pairs,
        "final_kernel_count": kernel_count,
        "source": ("torch/_inductor/scheduler.py Scheduler.can_fuse_vertical call count "
                   "+ final kernel count from output_code.py"),
    }
    evidence["graph_artifact_count"] = result.trace.get("graph_artifact_count")
    evidence["graph_artifacts"] = result.trace.get("output_code_all")
    if (result.trace.get("graph_artifact_count") or 0) > 1:
        result.notes.append(
            f"{result.trace['graph_artifact_count']} compiled graphs in the trace: the frame "
            "produced more than one graph (evidence is taken from the first)")
    evidence["config"] = result.config
    evidence["kernel_resources"] = extract_kernel_resources([k.name for k in parsed.kernels])
    result.evidence = evidence

    # ---- status ---------------------------------------------------------
    if strategy == "split":
        realized = [e for e in result.realize_log if e.get("status") == "realized"]
        if not realized and not result.guard_log:
            result.status = "unsupported"
            result.errors.append(
                "neither the lowering realize request nor the fusion veto fired; "
                "the split mechanism did not reach the target edge")
        elif not evidence["t_materialized"]:
            result.status = "partition_mismatch"
            result.errors.append(
                "split request fired but no buffer with t's shape was written then read "
                f"(realize={len(realized)}, veto={len(result.guard_log)})")
        else:
            result.status = "valid"
    elif strategy == "fused":
        if evidence["verdict"] == "fused":
            result.status = "valid"
        elif evidence["t_materialized"]:
            # The default policy already materialises t: there is no legal fused
            # partition for this edge under the pinned stack.
            result.status = "unsupported"
            result.errors.append("t is materialized even without a guard: no legal "
                                 "single-kernel partition for this edge")
        else:
            result.status = "partition_mismatch"
            result.errors.append("could not attribute both sides of the edge to one kernel")
    elif strategy == "default":
        # An observation of Inductor's own choice, not a pass/fail claim.
        result.status = "valid" if evidence["verdict"] != "unclear" else "partition_mismatch"
        result.notes.append(f"default verdict: {evidence['verdict']}")
    else:
        result.status = "valid" if evidence["verdict"] != "unclear" else "partition_mismatch"

    _write_success(result, artifact_dir, parsed, evidence)
    return result


def _write_success(result: CompileResult, artifact_dir: str,
                   parsed: forensics.ParsedOutput, evidence: dict[str, Any]) -> None:
    records.write_json(os.path.join(artifact_dir, "compile_result.json"), {
        "case_id": result.case_id,
        "strategy": result.strategy,
        "status": result.status,
        "first_call_ms": result.first_call_ms,
        "second_call_ms": result.second_call_ms,
        "notes": result.notes,
        "errors": result.errors,
    })
    records.write_json(os.path.join(artifact_dir, "edge_evidence.json"), evidence)
    records.write_json(os.path.join(artifact_dir, "kernels.json"), {
        "kernels": [
            {
                "launch_index": k.launch_index,
                "name": k.name,
                "source_nodes": k.source_nodes,
                "original_aten": k.original_aten,
                "params": k.params,
                "reads": k.reads,
                "writes": k.writes,
                "grid": k.grid,
                "grid_dims": k.grid_dims,
                "program_ids": k.program_ids,
                "constexprs": k.constexpr_names,
                "num_warps": k.num_warps,
                "num_stages": k.num_stages,
                "source_sha256": forensics._kernel_code_hash(k),
                "resources": (evidence.get("kernel_resources") or {}).get(k.name),
                "source": k.source,
            }
            for k in sorted(parsed.kernels, key=lambda k: (k.launch_index is None, k.launch_index))
        ],
        "allocs": {name: a.__dict__ for name, a in parsed.allocs.items()},
    })
    # a readable excerpt of the edge evidence for the course report
    decision = evidence.get("fusion_decision") or {}
    lines = [f"# edge evidence: {result.case_id} / {result.strategy}", "",
             f"verdict: **{evidence['verdict']}**", "",
             f"fusion decision stage: **{decision.get('stage')}** "
             f"(scheduler considered {decision.get('scheduler_fusion_pairs_considered')} pair(s), "
             f"{decision.get('final_kernel_count')} kernel(s))", "",
             f"compiled graphs in trace: {evidence.get('graph_artifact_count')}", "",
             f"output_code: `{evidence['output_code']}`",
             f"sha256: `{evidence['output_code_sha256']}`", "",
             "## kernels in launch order"]
    for idx, name in enumerate(evidence["kernels"]):
        lines.append(f"{idx}. `{name}`")
    if evidence["matched_t_flows"]:
        lines += ["", "## t write -> read"]
        for flow in evidence["matched_t_flows"]:
            lines.append(f"- buffer `{flow['buffer']}` shape={flow['alloc']['shape']} dtype={flow['alloc']['dtype']}")
            lines.append(f"  - written by `{flow['writer_kernel']}` (source nodes {flow['writer_source_nodes']})")
            for site in flow["t_write_site"]:
                lines.append(f"    - L{site['line']}: `{site['text']}`")
            lines.append(f"  - read by `{flow['reader_kernel']}` (source nodes {flow['reader_source_nodes']})")
            for site in flow["t_read_site"]:
                lines.append(f"    - L{site['line']}: `{site['text']}`")
            for site in flow["wrapper_write_call"]:
                lines.append(f"  - wrapper L{site['line']}: `{site['text']}`")
            for site in flow["wrapper_read_call"]:
                lines.append(f"  - wrapper L{site['line']}: `{site['text']}`")
            if flow.get("in_place_read"):
                lines.append("  - the consumer reads and writes the same buffer: the "
                             "allocator handed t's storage to the output (in-place reuse)")
            for site in flow.get("wrapper_reuse_line") or []:
                lines.append(f"  - reuse L{site['line']}: `{site['text']}`")
    if evidence["fused_kernels_crossing_edge"]:
        lines += ["", "## kernels containing both sides of the edge"]
        for k in evidence["fused_kernels_crossing_edge"]:
            lines.append(f"- `{k['kernel']}` aten={k['original_aten']} grid={k['grid']} "
                         f"reads={k['reads']} writes={k['writes']}")
    if evidence.get("realize_log"):
        lines += ["", "## lowering realize request"]
        for e in evidence["realize_log"]:
            lines.append(f"- fx node `{e['fx_node']}` ({e['target']}) -> buffer `{e.get('buffer')}` "
                         f"status=`{e['status']}`" + (f" error={e.get('error')}" if e.get("error") else ""))
    if evidence["guard_log"]:
        lines += ["", "## fusion veto log"]
        for hit in evidence["guard_log"]:
            lines.append(f"- gate `{hit['gate']}` blocked: {hit['producer_names']} -> {hit['consumer_names']}")
    res = evidence.get("kernel_resources") or {}
    if res:
        lines += ["", "## static kernel resources (block / registers / shared / warps)",
                  "", "| kernel | block | regs | spills | shared B | warps | stages |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            lines.append(f"| `{name}` | {r.get('block')} | {r.get('n_regs')} | {r.get('n_spills')} "
                         f"| {r.get('shared_bytes')} | {r.get('num_warps')} | {r.get('num_stages')} |")
        lines.append("")
        lines.append("每个数值的来源属性记录在 `edge_evidence.json` 的 `kernel_resources[*].sources`。")
    if result.notes:
        lines += ["", "## notes"] + [f"- {n}" for n in result.notes if n]
    if result.errors:
        lines += ["", "## errors"] + [f"- {e}" for e in result.errors]
    with open(os.path.join(artifact_dir, "evidence.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_failure(result: CompileResult, artifact_dir: str) -> None:
    records.write_json(os.path.join(artifact_dir, "compile_result.json"), {
        "case_id": result.case_id,
        "strategy": result.strategy,
        "status": result.status,
        "first_call_ms": result.first_call_ms,
        "notes": result.notes,
        "errors": result.errors,
    })
