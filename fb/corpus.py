"""Case catalog and manifest.

The canonical first-phase corpus is 72 cases:
  P1/P2: N in {1024, 8192, 65536, 524288, 4194304, 16777216} x {f32, f16} = 24
  R1/R2: M in {16, 256, 4096} x K in {128, 512, 2048, 8192} x {f32, f16} = 48

A (compilation) owns the authoritative catalog; this module emits the reference
catalog and validates a delivered catalog against the data contract.
"""
from __future__ import annotations

import re
from typing import Any

from .util import sha256_json

NS = (1024, 8192, 65536, 524288, 4194304, 16777216)
MS = (16, 256, 4096)
KS = (128, 512, 2048, 8192)
DTYPES = ("float32", "float16")
DTYPE_TAG = {"float32": "f32", "float16": "f16"}
DTYPE_BYTES = {"float32": 4, "float16": 2}

TEMPLATES = {
    "P1": {
        "producer": "x * 1.25 + 0.25",
        "consumer": "relu(t)",
        "edge": "t",
        "t_expr": "x * 1.25 + 0.25",
    },
    "P2": {
        "producer": "tanh(x)",
        "consumer": "t * 0.75 + 0.25",
        "edge": "t",
        "t_expr": "tanh(x)",
    },
    "R1": {
        "producer": "sum(x, dim=1, dtype=float32)",
        "consumer": "relu(t)",
        "edge": "t",
        "t_expr": "sum(x, dim=1, dtype=float32)",
    },
    "R2": {
        "producer": "amax(x, dim=1)",
        "consumer": "t * 0.75 + 0.25",
        "edge": "t",
        "t_expr": "amax(x, dim=1)",
    },
}

PILOT_CASES = ("P1-n65536-f32-contig-t", "R1-m256-k512-f32-contig-t")

REQUIRED_CASE_FIELDS = (
    "case_id",
    "template",
    "shape",
    "dtype",
    "layout",
    "edge",
    "t_shape",
    "t_dtype",
    "logical_bytes",
    "input_seeds",
    "strategies",
    "planned_status",
)


def _case(template: str, shape: dict[str, int], dtype: str) -> dict[str, Any]:
    tty = DTYPE_TAG[dtype]
    if template in ("P1", "P2"):
        n = shape["N"]
        case_id = f"{template}-n{n}-{tty}-contig-t"
        t_shape = [n]
        t_dtype = dtype
        bytes_per_t = DTYPE_BYTES[dtype]
        logical_bytes = {"t_write": n * bytes_per_t, "t_read": n * bytes_per_t}
    else:
        m, k = shape["M"], shape["K"]
        case_id = f"{template}-m{m}-k{k}-{tty}-contig-t"
        t_dtype = "float32" if template == "R1" else dtype
        t_shape = [m]
        bytes_per_t = DTYPE_BYTES[t_dtype]
        logical_bytes = {"t_write": m * bytes_per_t, "t_read": m * bytes_per_t}
    logical_saved = logical_bytes["t_write"] + logical_bytes["t_read"]
    return {
        "case_id": case_id,
        "template": template,
        "shape": shape,
        "dtype": dtype,
        "layout": "contig",
        "edge": "t",
        "t_shape": t_shape,
        "t_dtype": t_dtype,
        "logical_bytes": logical_bytes,
        "logical_saved_bytes": logical_saved,
        "input_seeds": [17, 29, 43, 0],
        "zero_input_id": "zero",
        "strategies": ["split", "fused"],
        "planned_status": "planned",
        "exclusion_reason": None,
        "producer_expr": TEMPLATES[template]["producer"],
        "consumer_expr": TEMPLATES[template]["consumer"],
    }


def build_catalog() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for template in ("P1", "P2"):
        for n in NS:
            for dtype in DTYPES:
                cases.append(_case(template, {"N": n}, dtype))
    for template in ("R1", "R2"):
        for m in MS:
            for k in KS:
                for dtype in DTYPES:
                    cases.append(_case(template, {"M": m, "K": k}, dtype))
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)) == 72, (len(ids), len(set(ids)))
    return cases


def catalog_hash(cases: list[dict]) -> str:
    return sha256_json(cases)


def template_counts(cases: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        out[c["template"]] = out.get(c["template"], 0) + 1
    return out


def validate_catalog(cases: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for i, c in enumerate(cases):
        where = c.get("case_id", f"#{i}")
        for f in REQUIRED_CASE_FIELDS:
            if f not in c:
                problems.append(f"{where}: missing field {f}")
        cid = c.get("case_id")
        if cid in seen:
            problems.append(f"{where}: duplicate case_id")
        seen.add(cid)
        if c.get("layout") != "contig":
            problems.append(f"{where}: layout must be contig")
        if c.get("edge") != "t":
            problems.append(f"{where}: edge must be t")
        if not isinstance(c.get("input_seeds"), list) or not c.get("input_seeds"):
            problems.append(f"{where}: input_seeds must be a non-empty list")
        strategies = c.get("strategies")
        excluded = c.get("planned_status") == "excluded"
        if not isinstance(strategies, list) or (
            not excluded
            and set(strategies) not in ({"split", "fused"}, {"split", "fused", "default"})
        ):
            problems.append(f"{where}: strategies must include split/fused")
        if c.get("dtype") not in DTYPES:
            problems.append(f"{where}: unsupported dtype {c.get('dtype')}")
        shape = c.get("shape")
        if c.get("template") in ("P1", "P2"):
            if isinstance(shape, dict):
                ok = "N" in shape
            else:
                ok = isinstance(shape, (list, tuple)) and len(shape) == 1
            if not ok:
                problems.append(f"{where}: P template needs N in shape")
        elif c.get("template") in ("R1", "R2"):
            if isinstance(shape, dict):
                ok = {"M", "K"} <= set(shape)
            else:
                ok = isinstance(shape, (list, tuple)) and len(shape) == 2
            if not ok:
                problems.append(f"{where}: R template needs M and K in shape")
        elif c.get("template") not in TEMPLATES:
            problems.append(f"{where}: unknown template {c.get('template')}")
        cid_ok = re.fullmatch(r"(P[12]|R[12])-(n|m)\d+(-k\d+)?-(f32|f16)-contig-t", str(cid))
        if not cid_ok:
            problems.append(f"{where}: case_id does not follow the documented pattern")
    return problems
