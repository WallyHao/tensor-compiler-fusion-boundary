"""Forensics over Inductor's generated artifacts.

Evidence is taken from the generated code itself (``output_code.py``) and from
the scheduler dumps (``ir_pre_fusion.*`` / ``ir_post_fusion.*``), never inferred
from config flags:

* which Triton kernels exist, in launch order, with their grid and arguments;
* which wrapper buffer each kernel writes and which it reads;
* where the target intermediate ``t`` is allocated, written and read back
  (split), or that one kernel contains both sides of the edge and no
  intermediate buffer of that shape/dtype exists (fused);
* whether the non-target part of the partition is unchanged between two
  strategies.

The parser is AST based; the wrapper/kernel conventions it relies on are
documented in ``docs/artifacts.md`` and asserted by ``tests/test_forensics.py``
against a recorded fixture.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .records import sha256_file


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class BufferAlloc:
    name: str
    shape: tuple[int, ...] | None
    dtype: str | None
    device: str | None
    line: int


@dataclass
class Kernel:
    name: str
    source: str
    source_nodes: list[str] = field(default_factory=list)
    original_aten: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    loads: set[str] = field(default_factory=set)
    stores: set[str] = field(default_factory=set)
    program_ids: list[int] = field(default_factory=list)
    grid_dims: int | None = None
    constexpr_names: list[str] = field(default_factory=list)
    #: wrapper-level facts, filled from the call site
    launch_index: int | None = None
    args: list[str] = field(default_factory=list)
    grid: str | None = None
    num_warps: int | None = None
    num_stages: int | None = None
    writes: list[str] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)


@dataclass
class ParsedOutput:
    path: str
    sha256: str
    kernels: list[Kernel]
    allocs: dict[str, BufferAlloc]
    wrapper_lines: list[str]
    extern_calls: list[dict[str, Any]] = field(default_factory=list)
    parse_notes: list[str] = field(default_factory=list)
    #: wrapper-level `buf1 = buf0` reuse aliases, resolved to the allocated name
    aliases: dict[str, str] = field(default_factory=dict)

    def kernel(self, name: str) -> Kernel | None:
        for k in self.kernels:
            if k.name == name:
                return k
        return None


# --------------------------------------------------------------------------
# output_code.py parsing
# --------------------------------------------------------------------------
_PTR_RE = re.compile(r"\b(in_out_ptr|in_ptr|out_ptr)(\d+)\b")
_SRC_NODES_RE = re.compile(r"Source Nodes:\s*\[([^\]]*)\]")
_ATEN_RE = re.compile(r"Original ATen:\s*\[([^\]]*)\]")


def _assign_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Tuple):
            return None
    return None


def _call_name(call: ast.Call) -> str:
    func = call.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_kernel_source(source: str, kernel_name: str) -> tuple[str, ast.FunctionDef | None, list[str]]:
    """Return (kernel source, parsed def, parse notes)."""
    notes: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return source, None, [f"kernel source does not parse: {exc}"]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == kernel_name:
                return source, node, notes
            if not hasattr(tree, "_fb_first"):
                tree._fb_first = node  # type: ignore[attr-defined]
    first = getattr(tree, "_fb_first", None)
    if first is not None:
        notes.append(f"kernel def named {kernel_name} not found; using first def {first.name}")
        return source, first, notes
    notes.append(f"no FunctionDef found in kernel source for {kernel_name}")
    return source, None, notes


def _scan_kernel_body(fn: ast.FunctionDef) -> tuple[set[str], set[str], list[int], list[str]]:
    loads: set[str] = set()
    stores: set[str] = set()
    pids: list[int] = []
    constexprs: list[str] = []
    for a in fn.args.args:
        ann = a.annotation
        if isinstance(ann, ast.Name) and ann.id == "constexpr":
            constexprs.append(a.arg)
        elif isinstance(ann, ast.Attribute) and getattr(ann.value, "id", "") == "tl":
            constexprs.append(a.arg)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name.endswith("tl.load"):  # type: ignore[truthiness]
            ptr = _ptr_of(node.args[0]) if node.args else None
            if ptr:
                loads.add(ptr)
        elif name.endswith("tl.store"):
            ptr = _ptr_of(node.args[0]) if node.args else None
            if ptr:
                stores.add(ptr)
        elif name.endswith("tl.program_id") and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                pids.append(int(arg.value))
    return loads, stores, pids, constexprs


def _ptr_of(node: ast.AST) -> str | None:
    """Find the first pointer parameter inside a load/store address expression."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and _PTR_RE.fullmatch(sub.id):
            return sub.id
    return None


def _alloc_from_call(call: ast.Call) -> tuple[tuple[int, ...] | None, str | None, str | None]:
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None
    for arg in list(call.args) + [kw.value for kw in call.keywords]:
        if isinstance(arg, (ast.Tuple, ast.List)):
            if shape is not None:
                continue  # the first tuple is the size; later ones are strides
            dims = []
            ok = True
            for elt in arg.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(int(elt.value))
                else:
                    ok = False
                    break
            if ok:
                shape = tuple(dims)
        # dtype may be positional: empty_strided_cuda((..), (..), torch.float32)
        elif dtype is None and isinstance(arg, ast.Attribute) and getattr(arg.value, "id", "") == "torch":
            dtype = arg.attr
        elif dtype is None and isinstance(arg, ast.Name) and arg.id.startswith(("float", "half",
                                                                                "bfloat", "int", "uint", "bool")):
            dtype = arg.id
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "cuda" in arg.value:
            device = arg.value
    for kw in call.keywords:
        if kw.arg == "dtype":
            node = kw.value
            if isinstance(node, ast.Attribute):
                dtype = node.attr
            elif isinstance(node, ast.Name):
                dtype = node.id
        elif kw.arg == "device":
            device = _const_str(kw.value)
    return shape, dtype, device


def parse_output_code(path: str) -> ParsedOutput:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines()
    tree = ast.parse(text)
    notes: list[str] = []

    kernels: list[Kernel] = []
    kernel_by_var: dict[str, Kernel] = {}
    allocs: dict[str, BufferAlloc] = {}
    extern: list[dict[str, Any]] = []

    # 1. kernel factories: `<var> = async_compile.triton('name', '''src''', ...)`
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        fname = _call_name(call)
        var = _assign_name(node)
        if fname.endswith("triton") and len(call.args) >= 2:
            kname = _const_str(call.args[0])
            ksrc = _const_str(call.args[1])
            if not kname or ksrc is None:
                continue
            src, fn, knotes = _extract_kernel_source(ksrc, kname)
            notes.extend(knotes)
            k = Kernel(name=kname, source=src)
            # decorator comment lines above the assignment carry Source Nodes
            j = node.lineno - 1
            while j >= 1:
                prev = lines[j - 1]
                if prev.strip().startswith("#"):
                    m = _SRC_NODES_RE.search(prev)
                    if m:
                        k.source_nodes = [s.strip() for s in m.group(1).split(",") if s.strip()]
                    m = _ATEN_RE.search(prev)
                    if m:
                        k.original_aten = [s.strip() for s in m.group(1).split(",") if s.strip()]
                    j -= 1
                    continue
                break
            if fn is not None:
                k.params = [a.arg for a in fn.args.args]
                loads, stores, pids, constexprs = _scan_kernel_body(fn)
                k.loads, k.stores, k.program_ids, k.constexpr_names = loads, stores, pids, constexprs
                k.grid_dims = (max(pids) + 1) if pids else None
            kernels.append(k)
            if var:
                kernel_by_var[var] = k
        elif fname.endswith("triton_helpers") or fname.startswith("extern_kernels"):
            extern.append({"var": var, "call": fname, "line": node.lineno})

    # 2. wrapper: buffer allocations and launch order
    wrapper = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("call", "run", "_run"):
            wrapper = node
            break
    if wrapper is None:
        notes.append("no wrapper function (call/run) found")
        return ParsedOutput(path, sha256_file(path), kernels, allocs, lines, extern, notes)

    # Alias assignments created by Inductor's memory reuse: `buf1 = buf0  # reuse`
    alias: dict[str, str] = {}
    for node in ast.walk(wrapper):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name)):
            alias[node.targets[0].id] = node.value.id

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    launch = 0
    for node in ast.walk(wrapper):
        # buffer allocation: buf0 = empty_strided(...)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fname = _call_name(node.value)
            var = _assign_name(node)
            leaf = fname.split(".")[-1]
            if var and leaf.startswith("empty"):
                shape, dtype, device = _alloc_from_call(node.value)
                allocs[var] = BufferAlloc(var, shape, dtype, device, node.lineno)
            elif var and fname.startswith("extern_kernels"):
                extern.append({"var": var, "call": fname, "line": node.lineno})
        # kernel launch: triton_x.run(args..., grid=..., stream=...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run":
            base = node.func.value
            if isinstance(base, ast.Name) and base.id in kernel_by_var:
                k = kernel_by_var[base.id]
                k.launch_index = launch
                launch += 1
                k.args = [resolve(a.id) if isinstance(a, ast.Name) else ast.dump(a)
                          for a in node.args]
                for kw in node.keywords:
                    if kw.arg == "grid":
                        k.grid = ast.unparse(kw.value)
                    elif kw.arg == "num_warps" and isinstance(kw.value, ast.Constant):
                        k.num_warps = int(kw.value.value)
                    elif kw.arg == "num_stages" and isinstance(kw.value, ast.Constant):
                        k.num_stages = int(kw.value.value)
                # map param -> wrapper value, then ptr -> buffer names
                for param, argname in zip(k.params, k.args):
                    m = _PTR_RE.fullmatch(param)
                    if not m:
                        continue
                    if param.startswith("out_ptr") and argname:
                        if argname not in k.writes:
                            k.writes.append(argname)
                    if param.startswith(("in_ptr", "in_out_ptr")) and argname:
                        if param.startswith("in_out_ptr"):
                            if argname not in k.reads:
                                k.reads.append(argname)
                        elif argname not in k.reads:
                            k.reads.append(argname)
    # pointer params that are both loaded and stored are in-out buffers
    for k in kernels:
        for param, argname in zip(k.params, k.args):
            if param.startswith("in_out_ptr") and argname and argname not in k.writes:
                k.writes.append(argname)
    parsed = ParsedOutput(path, sha256_file(path), kernels, allocs, lines, extern, notes)
    parsed.aliases = {k: resolve(k) for k in alias}
    return parsed


# --------------------------------------------------------------------------
# IR dump parsing (supporting evidence, tolerant to format changes)
# --------------------------------------------------------------------------
_SCHED_RE = re.compile(r"^\[(?P<idx>\d+)\]:\s*(?P<name>\S+)\s*$")
_BUF_WRITES_RE = re.compile(r"^(?P<buf>[A-Za-z0-9_]+)\.writes = \[(?P<v>.*)\]")
_BUF_USERS_RE = re.compile(r"^(?P<buf>[A-Za-z0-9_]+)\.users = \[(?P<v>.*)\]")
_MEMDEP_RE = re.compile(r"MemoryDep\('(?P<buf>[A-Za-z0-9_]+)'")
_KERNEL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _looks_like_kernel_header(line: str) -> bool:
    """A bare line that is a generated kernel name (``triton_...``/``cpp_...``)."""
    if not line or " " in line or "=" in line or line.startswith("#"):
        return False
    if not _KERNEL_NAME_RE.match(line):
        return False
    return line.startswith(("triton_", "cpp_", "extern_", "multi_"))


def parse_scheduler_dump(path: str) -> dict[str, Any]:
    """Best-effort structure of an ir_post_fusion dump.

    Returns ``{"kernels": [{"scheduler_name", "buffers_written", "buffer_users"}],
    "buffer_writes": {...}, "buffer_users": {...}, "parsed": bool}``.
    Format differences across versions are reported instead of raising.
    """
    import json

    result: dict[str, Any] = {"parsed": False, "notes": [], "kernel_names": [],
                              "buffer_writes": {}, "buffer_users": {}, "raw_line_count": 0}
    if not os.path.exists(path):
        result["notes"].append("missing file")
        return result
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    result["raw_line_count"] = len(lines)
    for line in lines:
        s = line.strip()
        if s.startswith("{") and "scheduler_name" in s:
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "scheduler_name" in obj:
                result["kernel_names"].append(obj.get("scheduler_name"))
                result["parsed"] = True
            continue
        m = _SCHED_RE.match(s)
        if m:
            result["kernel_names"].append(m.group("name"))
            result["parsed"] = True
            continue
        if _looks_like_kernel_header(s):
            result["kernel_names"].append(s)
            result["parsed"] = True
            continue
        m = _BUF_WRITES_RE.match(s)
        if m:
            deps = _MEMDEP_RE.findall(m.group("v"))
            result["buffer_writes"][m.group("buf")] = deps
            result["parsed"] = True
            continue
        m = _BUF_USERS_RE.match(s)
        if m:
            deps = re.findall(r"buf\d+|arg\d+_\d+", m.group("v"))
            result["buffer_users"][m.group("buf")] = sorted(set(deps))
            result["parsed"] = True
    if not result["kernel_names"]:
        result["notes"].append("no scheduler node headers matched")
    return result


# --------------------------------------------------------------------------
# evidence extraction
# --------------------------------------------------------------------------
def _op_matches(names: list[str], needles: tuple[str, ...]) -> bool:
    joined = " ".join(names).lower()
    return any(n.lower() in joined for n in needles)


def buffer_flows(parsed: ParsedOutput) -> list[dict[str, Any]]:
    """Intermediate buffers with a writer kernel and one or more reader kernels."""
    flows: list[dict[str, Any]] = []
    writers: dict[str, list[Kernel]] = {}
    readers: dict[str, list[Kernel]] = {}
    for k in parsed.kernels:
        for b in k.writes:
            writers.setdefault(b, []).append(k)
        for b in k.reads:
            readers.setdefault(b, []).append(k)
    for buf, ws in writers.items():
        rs = readers.get(buf, [])
        if not rs:
            continue
        writer_names = {k.name for k in ws}
        flows.append({
            "buffer": buf,
            "alloc": parsed.allocs.get(buf).__dict__ if buf in parsed.allocs else None,
            "writers": [{"kernel": k.name, "launch_index": k.launch_index} for k in ws],
            "readers": [{"kernel": k.name, "launch_index": k.launch_index} for k in rs],
            # True when a reading kernel also stores into the same buffer
            # (Inductor reuses the last user's storage: `buf1 = buf0  # reuse`).
            "in_place_read": any(r.name in writer_names for r in rs),
            "aliases": {k: v for k, v in parsed.aliases.items() if v == buf},
        })
    return flows


def _excerpt(text_lines: list[str], line_no: int, before: int = 0, after: int = 0) -> dict[str, Any]:
    lo = max(1, line_no - before)
    hi = min(len(text_lines), line_no + after)
    return {"line": line_no, "text": "\n".join(text_lines[lo - 1:hi])}


def _kernel_store_excerpt(k: Kernel, ptr: str, limit: int = 6) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(k.source.splitlines(), start=1):
        if f"tl.store({ptr}" in line.replace(" ", "") or f"tl.store({ptr} " in line:
            out.append({"kernel": k.name, "line": i, "text": line.strip()})
            if len(out) >= limit:
                break
    return out


def _kernel_load_excerpt(k: Kernel, ptr: str, limit: int = 6) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(k.source.splitlines(), start=1):
        stripped = line.replace(" ", "")
        if f"tl.load({ptr}" in stripped:
            out.append({"kernel": k.name, "line": i, "text": line.strip()})
            if len(out) >= limit:
                break
    return out


def _wrapper_excerpt(parsed: ParsedOutput, pattern: str, limit: int = 8) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(parsed.wrapper_lines, start=1):
        if re.search(pattern, line):
            out.append({"line": i, "text": line.strip()})
            if len(out) >= limit:
                break
    return out


def edge_evidence(case, parsed: ParsedOutput, strategy: str, spec) -> dict[str, Any]:  # noqa: ANN001
    """Decide, from generated code, how the edge ``t`` was compiled."""
    t_shape = tuple(spec.t_shape)
    t_dtype = spec.t_dtype

    candidate_allocs = [
        a for a in parsed.allocs.values()
        if a.shape == t_shape and (a.dtype is None or a.dtype == t_dtype)
    ]
    flows = buffer_flows(parsed)
    matched_flows = []
    for flow in flows:
        buf = flow["buffer"]
        alloc = parsed.allocs.get(buf)
        if alloc is None or alloc.shape != t_shape:
            continue
        kw = None
        for w in flow["writers"]:
            cand = parsed.kernel(w["kernel"])
            if cand is not None and _op_matches(cand.source_nodes + cand.original_aten,
                                                spec.producer_ops):
                kw = cand
                break
        kr = None
        for r in flow["readers"]:
            cand = parsed.kernel(r["kernel"])
            if cand is not None and _op_matches(cand.source_nodes + cand.original_aten,
                                                spec.consumer_ops):
                kr = cand
                break
        if kw is None or kr is None or kw is kr:
            continue
        matched_flows.append({
            "buffer": buf,
            "alloc": alloc.__dict__,
            "writer_kernel": kw.name,
            "reader_kernel": kr.name,
            "writer_index": kw.launch_index,
            "reader_index": kr.launch_index,
            "writer_source_nodes": kw.source_nodes,
            "writer_aten": kw.original_aten,
            "reader_source_nodes": kr.source_nodes,
            "reader_aten": kr.original_aten,
            "producer_op_match": True,
            "consumer_op_match": True,
            "in_place_read": flow.get("in_place_read"),
            "aliases": flow.get("aliases"),
            "t_write_site": _kernel_store_excerpt(kw, _ptr_for_arg(kw, buf)),
            "t_read_site": _kernel_load_excerpt(kr, _ptr_for_arg(kr, buf)),
            "wrapper_write_call": _wrapper_excerpt(parsed, rf"{re.escape(kw.name)}\.run"),
            "wrapper_read_call": _wrapper_excerpt(parsed, rf"{re.escape(kr.name)}\.run"),
            # `buf1 = buf0; del buf0  # reuse` — storage hand-over to the consumer
            "wrapper_reuse_line": [
                hit for alias_name in (flow.get("aliases") or {})
                for hit in _wrapper_excerpt(parsed, rf"^\s*{re.escape(alias_name)}\s*=\s*{re.escape(buf)}\b", 2)
            ],
            "alloc_line": _wrapper_excerpt(parsed, rf"^\s*{re.escape(buf)}\s*="),
        })

    fused_kernels = []
    for k in parsed.kernels:
        ops = k.source_nodes + k.original_aten
        if _op_matches(ops, spec.producer_ops) and _op_matches(ops, spec.consumer_ops):
            fused_kernels.append({
                "kernel": k.name,
                "launch_index": k.launch_index,
                "source_nodes": k.source_nodes,
                "original_aten": k.original_aten,
                "reads": k.reads,
                "writes": k.writes,
                "grid": k.grid,
                "grid_dims": k.grid_dims,
                "num_warps": k.num_warps,
            })

    if strategy == "fused" or (strategy == "default" and not matched_flows):
        verdict = "fused" if fused_kernels else "unclear"
    elif strategy == "split":
        verdict = "materialized" if matched_flows else "unclear"
    else:  # diagnostic variants: report what happened, no expectation
        verdict = "materialized" if matched_flows else ("fused" if fused_kernels else "unclear")

    detail: dict[str, Any] = {
        "case_id": case.case_id,
        "strategy": strategy,
        "edge": case.edge,
        "t_shape": list(t_shape),
        "t_dtype": t_dtype,
        "output_code": parsed.path,
        "output_code_sha256": parsed.sha256,
        "kernel_count": len(parsed.kernels),
        "kernels": [k.name for k in sorted(parsed.kernels, key=lambda k: (k.launch_index is None, k.launch_index))],
        "tearoff_buffers_with_t_shape": [a.name for a in candidate_allocs],
        "intermediate_flows": flows,
        "matched_t_flows": matched_flows,
        "fused_kernels_crossing_edge": fused_kernels,
        "verdict": verdict,
        "parse_notes": parsed.parse_notes,
    }
    detail["t_allocated"] = bool(candidate_allocs)
    detail["t_materialized"] = bool(matched_flows)
    if matched_flows:
        alloc = matched_flows[0]["alloc"]
        detail["t_alloc_dtype"] = alloc.get("dtype")
        detail["t_alloc_shape"] = list(alloc["shape"]) if alloc.get("shape") else None
        detail["t_alloc_buffer"] = matched_flows[0]["buffer"]
    else:
        detail["t_alloc_dtype"] = candidate_allocs[0].dtype if candidate_allocs else None
        detail["t_alloc_shape"] = (list(candidate_allocs[0].shape)
                                   if candidate_allocs and candidate_allocs[0].shape else None)
        detail["t_alloc_buffer"] = candidate_allocs[0].name if candidate_allocs else None
    return detail


def _ptr_for_arg(kernel: Kernel, argname: str) -> str:
    for param, arg in zip(kernel.params, kernel.args):
        if arg == argname and _PTR_RE.fullmatch(param):
            return param
    # fall back to the conventional name
    return "out_ptr0" if argname not in kernel.reads else "in_ptr0"


def partition_signature(parsed: ParsedOutput) -> list[dict[str, Any]]:
    """Comparable per-kernel identity used to detect unrelated partition changes."""
    sig = []
    for k in sorted(parsed.kernels, key=lambda k: (k.launch_index is None, k.launch_index)):
        sig.append({
            "launch_index": k.launch_index,
            "name": k.name,
            "source_nodes": k.source_nodes,
            "original_aten": k.original_aten,
            "reads": k.reads,
            "writes": k.writes,
            "grid": k.grid,
            "program_id_dims": k.grid_dims,
            "num_warps": k.num_warps,
            "num_stages": k.num_stages,
            "code_sha256": _kernel_code_hash(k),
        })
    return sig


def _kernel_code_hash(k: Kernel) -> str:
    from .records import sha256_text

    return sha256_text(k.source)


def compare_partitions(a: ParsedOutput, b: ParsedOutput) -> dict[str, Any]:
    """Diff two partitions, ignoring the kernels that touch the target edge.

    Returns the shared (unchanged) kernels by code hash, the kernels only in
    ``a`` and only in ``b``. Used to check that a control change did not move
    unrelated partitions (work order: `无关分区变化`).
    """
    sa, sb = partition_signature(a), partition_signature(b)
    ha = {x["code_sha256"]: x for x in sa}
    hb = {x["code_sha256"]: x for x in sb}
    shared = [dict(name_a=ha[h]["name"], name_b=hb[h]["name"], launch_index=ha[h]["launch_index"])
              for h in ha.keys() & hb.keys()]
    return {
        "shared_kernel_count": len(shared),
        "shared": shared,
        "only_in_a": [x for h, x in ha.items() if h not in hb],
        "only_in_b": [x for h, x in hb.items() if h not in ha],
        "kernel_count_a": len(sa),
        "kernel_count_b": len(sb),
    }


def collect_trace_artifacts(trace_dir: str) -> dict[str, Any]:
    """Inventory of the Inductor trace directory written by config.trace."""
    inv: dict[str, Any] = {"trace_dir": trace_dir, "files": {}, "output_code": None,
                           "output_code_all": [], "graph_artifact_count": 0,
                           "ir_pre_fusion": None, "ir_post_fusion": None, "fx_graph": None}
    if not os.path.isdir(trace_dir):
        return inv
    for root, _dirs, files in os.walk(trace_dir):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, trace_dir)
            inv["files"][rel] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
            if f == "output_code.py":
                # One output_code.py per compiled graph: more than one means the
                # frame produced several graphs (graph break / multiple regions).
                inv["output_code_all"].append(p)
                if inv["output_code"] is None:
                    inv["output_code"] = p
            elif f in ("ir_pre_fusion.txt", "ir_pre_fusion.py"):
                inv["ir_pre_fusion"] = p
            elif f in ("ir_post_fusion.txt", "ir_post_fusion.py"):
                inv["ir_post_fusion"] = p
            elif f in ("fx_graph.py", "fx_graph.txt", "fx_graph_readable.py", "fx_graph_runnable.py"):
                inv["fx_graph"] = p
    inv["output_code_all"].sort()
    inv["graph_artifact_count"] = len(inv["output_code_all"])
    if inv["output_code_all"]:
        inv["output_code"] = inv["output_code_all"][0]
    return inv
