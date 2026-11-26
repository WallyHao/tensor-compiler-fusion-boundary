"""Reference candidate provider (development only, not official evidence).

Builds split/fused candidates with torch.compile. Marked ``unverified``:
no artifact hash, allocations inside the call, no static kernel count. Formal
runs therefore require ``--allow-unverified-candidates`` and run.json records
that the candidates are reference-only.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .backend import PreparedVariant
from .corpus import build_catalog

TOL = {
    ("P1", "float32"): (1e-6, 1e-5),
    ("P2", "float32"): (1e-6, 1e-5),
    ("R1", "float32"): (2e-4, 1e-4),
    ("R2", "float32"): (1e-6, 1e-5),
    ("P1", "float16"): (2e-3, 2e-3),
    ("P2", "float16"): (2e-3, 2e-3),
    ("R1", "float16"): (2e-3, 2e-3),
    ("R2", "float16"): (2e-3, 2e-3),
}


def eager_ref(template: str, x):
    import torch

    if template == "P1":
        return torch.relu(x * 1.25 + 0.25)
    if template == "P2":
        return torch.tanh(x) * 0.75 + 0.25
    if template == "R1":
        return torch.relu(torch.sum(x, dim=1, dtype=torch.float32))
    if template == "R2":
        return torch.amax(x, dim=1) * 0.75 + 0.25
    raise ValueError(template)


def split_exprs(template: str):
    import torch

    if template == "P1":
        return (lambda z: z * 1.25 + 0.25), (lambda t: torch.relu(t))
    if template == "P2":
        return (lambda z: torch.tanh(z)), (lambda t: t * 0.75 + 0.25)
    if template == "R1":
        return (lambda z: torch.sum(z, dim=1, dtype=torch.float32)), (
            lambda t: torch.relu(t)
        )
    if template == "R2":
        return (lambda z: torch.amax(z, dim=1)), (lambda t: t * 0.75 + 0.25)
    raise ValueError(template)


class ReferenceProvider:
    def __init__(self, options: dict | None = None):
        self.options = options or {}
        self.cases = build_catalog()
        self._by_id = {c["case_id"]: c for c in self.cases}
        self._inputs: dict[tuple[str, int], Any] = {}
        self._variants: dict[tuple[str, str], PreparedVariant] = {}
        self._torch = None

    @property
    def torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def catalog(self) -> list[dict]:
        return self.cases

    def _dtype(self, case: dict):
        return getattr(self.torch, case["dtype"])

    def _make_input(self, case: dict, seed: int):
        torch = self.torch
        shape = (
            (case["shape"]["N"],)
            if "N" in case["shape"]
            else (case["shape"]["M"], case["shape"]["K"])
        )
        dtype = self._dtype(case)
        if seed == 0:
            return torch.zeros(shape, dtype=dtype, device="cuda")
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        x = torch.rand(shape, generator=g, device="cuda", dtype=torch.float32)
        return (x * 2.0 - 1.0).to(dtype)

    def _timing_input(self, case: dict):
        key = (case["case_id"], 17)
        if key not in self._inputs:
            self._inputs[key] = self._make_input(case, 17)
        return self._inputs[key]

    def _compile_fn(self, case: dict, strategy: str) -> Callable:
        torch = self.torch
        f1, f2 = split_exprs(case["template"])
        if strategy == "split":
            producer = torch.compile(f1, dynamic=False, fullgraph=True)
            consumer = torch.compile(f2, dynamic=False, fullgraph=True)

            def fn(z):
                return consumer(producer(z))

        elif strategy == "fused":

            def whole(z):
                return f2(f1(z))

            fn = torch.compile(whole, dynamic=False, fullgraph=True)
        else:
            raise ValueError(f"unknown strategy {strategy}")
        return fn

    def prepare(self, case: dict, strategy: str) -> PreparedVariant:
        key = (case["case_id"], strategy)
        if key in self._variants:
            return self._variants[key]
        torch = self.torch
        try:
            import torch._inductor.config as ind_cfg

            ind_cfg.triton.cudagraphs = False
        except Exception:
            pass
        x = self._timing_input(case)
        t0 = time.perf_counter()
        fn = self._compile_fn(case, strategy)
        with torch.inference_mode():
            fn(x)
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000.0

        meta = {
            "provider": "reference",
            "strategy": strategy,
            "artifact_sha256": None,
            "artifacts": [],
            "n_kernels": None,
            "preallocated": False,
            "preallocated_inputs": True,
            "compile_ms": compile_ms,
            "submit_mode": "direct",
            "notes": "reference provider: torch.compile graph; allocations inside call",
            "dtype": str(self._dtype(case)),
        }

        prepared = PreparedVariant(
            case_id=case["case_id"],
            strategy=strategy,
            meta=meta,
            _call=lambda: fn(x),
            _check=lambda input_id: self.check(case, strategy, fn, input_id),
        )
        self._variants[key] = prepared
        return prepared

    def check(self, case: dict, strategy: str, fn: Callable, input_id: str) -> dict:
        torch = self.torch
        seed = 0 if input_id == "zero" else int(input_id)
        x = self._make_input(case, seed)
        with torch.inference_mode():
            out = fn(x)
            ref = eager_ref(case["template"], x)
        atol, rtol = TOL[(case["template"], case["dtype"])]
        out_f = out.float()
        ref_f = ref.float()
        diff = (out_f - ref_f).abs()
        denom = ref_f.abs().clamp_min(1e-12)
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
        finite = int(torch.isfinite(out_f).sum().item())
        scale = max(float(ref_f.abs().max().item()), 1.0)
        ok = bool(
            out.shape == ref.shape
            and out.dtype == ref.dtype
            and max_abs <= atol + rtol * scale
            and finite == out_f.numel()
        )
        return {
            "reference": "eager",
            "tolerance": {"atol": atol, "rtol": rtol},
            "output_shape": list(out.shape),
            "output_dtype": str(out.dtype),
            "nonfinite_count": out_f.numel() - finite,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "fp64_reference": (
                "checked_by_tolerance_only" if case["template"] in ("R1", "R2") else "not_applicable"
            ),
            "pass": ok,
            "reason": None if ok else "tolerance_or_shape",
        }

    def close(self) -> None:
        for v in self._variants.values():
            v.close()
        self._variants.clear()
        self._inputs.clear()


def make_provider(root: Path, options: dict) -> ReferenceProvider:
    return ReferenceProvider(options)
