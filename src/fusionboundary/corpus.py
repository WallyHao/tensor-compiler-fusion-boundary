"""The 72-case corpus for phase 1.

One case = (template, shape, dtype, contiguous layout, named edge ``t``).
Grid and counts come from plan/phase1-3090/02-candidates.md:

* elementwise (P1, P2): N in {1024, 8192, 65536, 524288, 4194304, 16777216}
* reduction   (R1, R2): M in {16, 256, 4096}, K in {128, 512, 2048, 8192}
* dtype: float32, float16
  -> 2*6*2 = 24 elementwise + 2*3*4*2 = 48 reduction = 72 cases.

case_id  is stable and human readable, e.g. ``P1-n65536-f32-contig-t`` or
``R1-m256-k512-f32-contig-t`` (suggested form in plan/phase1-3090/04-data.md).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

ELEMENTWISE_NS = (1024, 8192, 65536, 524288, 4194304, 16777216)
REDUCTION_MS = (16, 256, 4096)
REDUCTION_KS = (128, 512, 2048, 8192)
DTYPES = ("float32", "float16")
REDUCTION_TEMPLATES = ("R1", "R2", "N1")
LAYOUTS = ("contig", "transpose")

#: touple of (atol, rtol) per plan/phase1-3090/02-candidates.md "数值验证".
TOL_ELEM_F32 = (1e-6, 1e-5)      # fp32 elementwise and fp32 amax
TOL_SUM_F32 = (2e-4, 1e-4)       # fp32 accumulation (sum)
TOL_F16 = (2e-3, 2e-3)           # every other fp16 output

TEMPLATE_INFO: dict[str, dict[str, Any]] = {
    "P1": {
        "kind": "elementwise",
        "producer": "t = x * 1.25 + 0.25",
        "consumer": "y = relu(t)",
        "t_source": "vec",
        "t_dtype_rule": "same_as_input",
        "out_dtype_rule": "same_as_input",
    },
    "P2": {
        "kind": "elementwise",
        "producer": "t = tanh(x)",
        "consumer": "y = t * 0.75 + 0.25",
        "t_source": "vec",
        "t_dtype_rule": "same_as_input",
        "out_dtype_rule": "same_as_input",
    },
    "R1": {
        "kind": "reduction",
        "producer": "t = sum(x, dim=1, dtype=float32)",
        "consumer": "y = relu(t)",
        "t_source": "row",
        "t_dtype_rule": "always_float32",
        "out_dtype_rule": "follow_t",
    },
    "R2": {
        "kind": "reduction",
        "producer": "t = amax(x, dim=1)",
        "consumer": "y = t * 0.75 + 0.25",
        "t_source": "row",
        "t_dtype_rule": "same_as_input",
        "out_dtype_rule": "follow_t",
    },
    # Followup (plan/followup-3090): RMSNorm row-mean -> inverse-RMS-factor edge.
    "N1": {
        "kind": "reduction",
        "producer": "t = mean(x.square(), dim=1)",
        "consumer": "y = rsqrt(t + 1e-5)",
        "t_source": "row",
        "t_dtype_rule": "always_float32",
        "out_dtype_rule": "follow_t",
    },
}

_DTYPE_TAG = {"float32": "f32", "float16": "f16"}
_DTYPE_ITEMSIZE = {"float32": 4, "float16": 2}


def tolerance_for(template: str, dtype: str) -> tuple[float, float]:
    """Tolerance table from the work order. R1 accumulates in fp32 for both dtypes."""
    if template in ("R1", "N1"):
        return TOL_SUM_F32
    if dtype == "float32":
        return TOL_ELEM_F32
    return TOL_F16


def input_stride(shape: tuple[int, ...], layout: str) -> tuple[int, ...]:
    """Physical stride of a case input.

    ``contig`` is the natural row-major stride; ``transpose`` is a logical
    ``(M, K)`` view over a contiguous ``(K, M)`` buffer, i.e. stride ``(1, M)``,
    so the reduction over logical dim 1 walks the fast axis of the storage.
    """
    if layout == "transpose" and len(shape) == 2:
        m, _k = shape
        return (1, m)
    stride: list[int] = []
    acc = 1
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


@dataclass(frozen=True)
class Case:
    case_id: str
    template: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    edge: str
    producer: str
    consumer: str
    t_shape: tuple[int, ...]
    t_dtype: str
    out_shape: tuple[int, ...]
    out_dtype: str
    logical_save_bytes: int
    seeds: tuple[int, ...]
    zero_input: bool
    group: str
    atol: float
    rtol: float
    fp64_reference: bool
    planned_status: str
    exclusion_reason: str | None
    extras: dict[str, Any] = field(default_factory=dict)

    # ---- derived helpers -------------------------------------------------
    @property
    def t_numel(self) -> int:
        n = 1
        for d in self.t_shape:
            n *= d
        return n

    @property
    def t_itemsize(self) -> int:
        return _DTYPE_ITEMSIZE[self.t_dtype]

    def torch_dtype(self) -> "Any":
        import torch

        return getattr(torch, self.dtype)

    def t_torch_dtype(self) -> "Any":
        import torch

        return getattr(torch, self.t_dtype)

    def as_manifest_row(self) -> dict[str, Any]:
        """Row shape of cases.jsonl (plan/phase1-3090/04-data.md)."""
        row = {
            "case_id": self.case_id,
            "template": self.template,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "edge": self.edge,
            "producer": self.producer,
            "consumer": self.consumer,
            "t_shape": list(self.t_shape),
            "t_dtype": self.t_dtype,
            "out_shape": list(self.out_shape),
            "out_dtype": self.out_dtype,
            "logical_save_bytes": self.logical_save_bytes,
            "logical_save_formula": "2 * numel(t) * itemsize(t)  # one write + one read of t",
            "seeds": list(self.seeds),
            "zero_input": self.zero_input,
            "tolerance": {"atol": self.atol, "rtol": self.rtol},
            "fp64_reference": self.fp64_reference,
            "group": self.group,
            "planned_status": self.planned_status,
            "exclusion_reason": self.exclusion_reason,
        }
        row.update(self.extras)
        return row

    @staticmethod
    def from_row(row: dict[str, Any]) -> "Case":
        known = {
            "case_id", "template", "kind", "shape", "dtype", "layout", "edge",
            "producer", "consumer", "t_shape", "t_dtype", "out_shape", "out_dtype",
            "logical_save_bytes", "seeds", "zero_input", "group", "planned_status",
            "exclusion_reason", "fp64_reference", "tolerance", "logical_save_formula",
        }
        template = row["template"]
        info = TEMPLATE_INFO.get(template, {})
        dtype = row["dtype"]
        kind = row.get("kind") or info.get("kind") or (
            "reduction" if template in REDUCTION_TEMPLATES else "elementwise")
        tol = row.get("tolerance", {})
        default_atol, default_rtol = tolerance_for(template, dtype)
        t_shape = tuple(row["t_shape"])
        t_itemsize = _DTYPE_ITEMSIZE.get(row["t_dtype"], 4)
        t_numel = 1
        for dim in t_shape:
            t_numel *= dim
        logical_save_bytes = row.get(
            "logical_save_bytes",
            row.get("logical_saved_bytes", 2 * t_numel * t_itemsize))
        return Case(
            case_id=row["case_id"],
            template=template,
            kind=kind,
            shape=tuple(row["shape"]),
            dtype=dtype,
            layout=row.get("layout", "contig"),
            edge=row.get("edge", "t"),
            producer=row.get("producer") or info.get("producer", ""),
            consumer=row.get("consumer") or info.get("consumer", ""),
            t_shape=t_shape,
            t_dtype=row["t_dtype"],
            out_shape=tuple(row.get("out_shape") or t_shape),
            out_dtype=row.get("out_dtype") or row["t_dtype"],
            logical_save_bytes=int(logical_save_bytes),
            seeds=tuple(row.get("seeds", row.get("input_seeds", (17, 29, 43)))),
            zero_input=bool(row.get("zero_input", True)),
            group=row.get("group") or template,
            atol=float(tol.get("atol", default_atol)),
            rtol=float(tol.get("rtol", default_rtol)),
            fp64_reference=bool(row.get("fp64_reference", kind == "reduction")),
            planned_status=row.get("planned_status", "planned"),
            exclusion_reason=row.get("exclusion_reason"),
            extras={k: v for k, v in row.items() if k not in known},
        )


def _make_case(template: str, shape: tuple[int, ...], dtype: str,
               layout: str = "contig", group: str | None = None) -> Case:
    info = TEMPLATE_INFO[template]
    kind = info["kind"]
    tag = _DTYPE_TAG[dtype]
    if kind == "elementwise":
        (n,) = shape
        cid = f"{template}-n{n}-{tag}-{layout}-t"
        t_shape: tuple[int, ...] = (n,)
        out_shape: tuple[int, ...] = (n,)
        t_dtype = dtype
        out_dtype = dtype
        extras: dict[str, Any] = {"n": n, "grid_axes": {"N": n}}
    else:
        m, k = shape
        cid = f"{template}-m{m}-k{k}-{tag}-{layout}-t"
        t_shape = (m,)
        out_shape = (m,)
        t_dtype = "float32" if info["t_dtype_rule"] == "always_float32" else dtype
        out_dtype = t_dtype
        extras = {"m": m, "k": k, "reduction_axis": 1, "grid_axes": {"M": m, "K": k}}
    atol, rtol = tolerance_for(template, dtype)
    t_bytes = 1
    for d in t_shape:
        t_bytes *= d
    t_bytes *= _DTYPE_ITEMSIZE[t_dtype]
    # Value of y on the all-zero input (t is then a constant, so y is exact);
    # used as an independent semantic self-check of the manifest. For N1 the
    # all-zero row mean is 0, so y = rsqrt(0 + 1e-5).
    zero_out = {"P1": 0.25, "P2": 0.25, "R1": 0.0, "R2": 0.25,
                "N1": 1.0 / math.sqrt(1e-5)}[template]
    extras["expected_zero_output"] = zero_out
    extras["logical_t_write_bytes"] = t_bytes
    extras["logical_t_read_bytes"] = t_bytes
    return Case(
        case_id=cid,
        template=template,
        kind=kind,
        shape=shape,
        dtype=dtype,
        layout=layout,
        edge="t",
        producer=info["producer"],
        consumer=info["consumer"],
        t_shape=t_shape,
        t_dtype=t_dtype,
        out_shape=out_shape,
        out_dtype=out_dtype,
        logical_save_bytes=2 * t_bytes,
        seeds=(17, 29, 43),
        zero_input=True,
        group=group if group is not None else template,
        atol=atol,
        rtol=rtol,
        fp64_reference=(kind == "reduction"),
        planned_status="planned",
        exclusion_reason=None,
        extras=extras,
    )


def build_corpus() -> list[Case]:
    cases: list[Case] = []
    for template in ("P1", "P2"):
        for n in ELEMENTWISE_NS:
            for dtype in DTYPES:
                cases.append(_make_case(template, (n,), dtype))
    for template in ("R1", "R2"):
        for m in REDUCTION_MS:
            for k in REDUCTION_KS:
                for dtype in DTYPES:
                    cases.append(_make_case(template, (m, k), dtype))
    return cases


def corpus_index(cases: list[Case] | None = None) -> dict[str, Case]:
    cases = cases if cases is not None else build_corpus()
    index = {c.case_id: c for c in cases}
    if len(index) != len(cases):
        raise AssertionError("duplicate case_id in corpus")
    return index


def get_case(case_id: str) -> Case:
    index = corpus_index()
    if case_id in index:
        return index[case_id]
    followup = followup_index()
    if case_id in followup:
        return followup[case_id]
    raise KeyError(
        f"unknown case_id {case_id!r}; see `python -m fusionboundary manifest` "
        "or `python -m fusionboundary manifest --followup`"
    )


def pilot_cases() -> list[Case]:
    """The two pilots fixed by the work order."""
    index = corpus_index()
    return [index["P1-n65536-f32-contig-t"], index["R1-m256-k512-f32-contig-t"]]


def expected_counts() -> dict[str, int]:
    return {"P1": 12, "P2": 12, "R1": 24, "R2": 24, "total": 72}


def iter_cases(templates: tuple[str, ...] | None = None,
               dtypes: tuple[str, ...] | None = None) -> Iterator[Case]:
    for case in build_corpus():
        if templates and case.template not in templates:
            continue
        if dtypes and case.dtype not in dtypes:
            continue
        yield case


def write_manifest(path: str) -> dict[str, Any]:
    """Write cases.jsonl (+ a JSON copy) and return a summary dict."""
    import hashlib
    import os

    cases = build_corpus()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.as_manifest_row(), sort_keys=True) + "\n")
    summary = {
        "cases": len(cases),
        "by_template": {t: sum(1 for c in cases if c.template == t) for t in ("P1", "P2", "R1", "R2")},
        "by_dtype": {d: sum(1 for c in cases if c.dtype == d) for d in DTYPES},
        "sha256": _sha256_file(path),
    }
    return summary


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: str) -> list[Case]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(Case.from_row(json.loads(line)))
    return rows


# --------------------------------------------------------------------------
# Followup corpus (plan/followup-3090): 16 new configurations + 4 same-round
# controls. Kept separate from ``build_corpus`` so the phase-1 behaviour
# (72 cases, ``--all``, existing case IDs) is unchanged.
# --------------------------------------------------------------------------
FOLLOWUP_GROUPS = ("anchor", "long", "layout", "rmsnorm")

#: The four same-round controls, re-measured under their original case IDs.
_FOLLOWUP_ANCHOR: tuple[tuple[str, tuple[int, int]], ...] = (
    ("R1", (16, 8192)),
    ("R1", (256, 8192)),
    ("R2", (16, 8192)),
    ("R2", (256, 8192)),
)
#: K extended to 16384 / 32768 on the four controls.
_FOLLOWUP_LONG: tuple[tuple[str, tuple[int, int]], ...] = tuple(
    (template, (m, k)) for template in ("R1", "R2") for m in (16, 256)
    for k in (16384, 32768)
)
#: Same shapes as the R templates but a transposed input view (stride (1, M)).
_FOLLOWUP_LAYOUT: tuple[tuple[str, tuple[int, int]], ...] = _FOLLOWUP_ANCHOR
#: RMSNorm inverse-RMS-factor edge, new template N1.
_FOLLOWUP_RMSNORM: tuple[tuple[str, tuple[int, int]], ...] = tuple(
    ("N1", (m, k)) for m in (16, 256) for k in (4096, 8192)
)

#: The three ``pilot=yes`` configurations of the work order.
FOLLOWUP_PILOT_IDS: tuple[str, ...] = (
    "R1-m16-k16384-f32-contig-t",
    "R2-m16-k8192-f32-transpose-t",
    "N1-m16-k8192-f32-contig-t",
)

_FOLLOWUP_CONTROLS: dict[str, str | None] = {
    "R1-m16-k16384-f32-contig-t": "R1-m16-k8192-f32-contig-t",
    "R1-m16-k32768-f32-contig-t": "R1-m16-k8192-f32-contig-t",
    "R1-m256-k16384-f32-contig-t": "R1-m256-k8192-f32-contig-t",
    "R1-m256-k32768-f32-contig-t": "R1-m256-k8192-f32-contig-t",
    "R2-m16-k16384-f32-contig-t": "R2-m16-k8192-f32-contig-t",
    "R2-m16-k32768-f32-contig-t": "R2-m16-k8192-f32-contig-t",
    "R2-m256-k16384-f32-contig-t": "R2-m256-k8192-f32-contig-t",
    "R2-m256-k32768-f32-contig-t": "R2-m256-k8192-f32-contig-t",
    "R1-m16-k8192-f32-transpose-t": "R1-m16-k8192-f32-contig-t",
    "R1-m256-k8192-f32-transpose-t": "R1-m256-k8192-f32-contig-t",
    "R2-m16-k8192-f32-transpose-t": "R2-m16-k8192-f32-contig-t",
    "R2-m256-k8192-f32-transpose-t": "R2-m256-k8192-f32-contig-t",
}


def followup_cases() -> list[Case]:
    """The 20 followup configurations (cases.csv), in work-order order."""
    built: list[Case] = []
    for template, shape in _FOLLOWUP_ANCHOR:
        built.append(_make_case(template, shape, "float32", "contig", "anchor"))
    for template, shape in _FOLLOWUP_LONG:
        built.append(_make_case(template, shape, "float32", "contig", "long"))
    for template, shape in _FOLLOWUP_LAYOUT:
        built.append(_make_case(template, shape, "float32", "transpose", "layout"))
    for template, shape in _FOLLOWUP_RMSNORM:
        built.append(_make_case(template, shape, "float32", "contig", "rmsnorm"))
    cases: list[Case] = []
    for case in built:
        extras = dict(case.extras)
        extras["control_id"] = _FOLLOWUP_CONTROLS.get(case.case_id)
        extras["pilot"] = case.case_id in FOLLOWUP_PILOT_IDS
        extras["source"] = "followup-3090"
        extras["input_shape"] = list(case.shape)
        extras["input_stride"] = list(input_stride(case.shape, case.layout))
        cases.append(replace(case, extras=extras))
    return cases


def followup_index(cases: list[Case] | None = None) -> dict[str, Case]:
    cases = cases if cases is not None else followup_cases()
    index = {c.case_id: c for c in cases}
    if len(index) != len(cases):
        raise AssertionError("duplicate case_id in followup corpus")
    return index


def followup_pilot_cases() -> list[Case]:
    index = followup_index()
    return [index[cid] for cid in FOLLOWUP_PILOT_IDS]


def followup_expected_counts() -> dict[str, int]:
    cases = followup_cases()
    counts = {g: sum(1 for c in cases if c.group == g) for g in FOLLOWUP_GROUPS}
    template_counts = {t: sum(1 for c in cases if c.template == t)
                       for t in sorted({c.template for c in cases})}
    counts.update(template_counts)
    counts["total"] = len(cases)
    return counts


def write_followup_manifest(path: str) -> dict[str, Any]:
    """Write the 20-case followup cases.jsonl and return a summary dict."""
    import os

    cases = followup_cases()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.as_manifest_row(), sort_keys=True) + "\n")
    return {
        "cases": len(cases),
        "by_group": {g: sum(1 for c in cases if c.group == g) for g in FOLLOWUP_GROUPS},
        "by_template": {t: sum(1 for c in cases if c.template == t)
                        for t in sorted({c.template for c in cases})},
        "pilot_cases": list(FOLLOWUP_PILOT_IDS),
        "sha256": _sha256_file(path),
    }
