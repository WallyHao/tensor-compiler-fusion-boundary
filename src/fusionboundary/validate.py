"""Fixed-seed inputs, eager/fp64 references, tolerance checks (checks.jsonl rows)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from . import templates


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _expected_stride(case) -> tuple[int, ...]:  # noqa: ANN001
    from .corpus import input_stride

    return tuple(input_stride(tuple(case.shape), getattr(case, "layout", "contig")))


def _place_input(logical: torch.Tensor, case) -> torch.Tensor:  # noqa: ANN001
    """Materialise logical values into the case's physical input layout."""
    dt = case.torch_dtype()
    if getattr(case, "layout", "contig") == "transpose":
        m, k = case.shape
        buf = torch.empty((k, m), dtype=dt)
        buf.copy_(logical.to(dt).t())
        return buf.t()
    return logical.to(dt)


def assert_input(case, x: torch.Tensor, input_id: str) -> None:  # noqa: ANN001
    """Guard the invariant every strategy and both submit modes rely on."""
    if tuple(x.shape) != tuple(case.shape):
        raise AssertionError(
            f"{case.case_id}/{input_id}: input shape {tuple(x.shape)} != {tuple(case.shape)}")
    if x.dtype != case.torch_dtype():
        raise AssertionError(
            f"{case.case_id}/{input_id}: input dtype {x.dtype} != {case.torch_dtype()}")
    want = _expected_stride(case)
    if tuple(x.stride()) != want:
        raise AssertionError(
            f"{case.case_id}/{input_id}: input stride {tuple(x.stride())} != {want}")


def iter_inputs(case, device: str | torch.device = "cpu") -> Iterator[tuple[str, torch.Tensor]]:  # noqa: ANN001
    """Yield (input_id, tensor) pairs: one per seed, then the all-zero input.

    Inputs are generated on CPU with a per-seed CPU generator, cast to the case
    dtype, then moved to ``device``. This makes the bit pattern independent of
    the device and of global RNG state, which the measurement owner relies on
    to reuse one fixed input across strategies and across processes.

    ``layout == "transpose"`` generates the same *logical* values from the same
    seed and copies them into a contiguous ``(K, M)`` buffer viewed as
    ``(M, K)`` with stride ``(1, M)``. Shape, stride and dtype are asserted for
    every yielded input.
    """
    for seed in case.seeds:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        base = torch.rand(case.shape, generator=gen, dtype=torch.float32)
        x = _place_input(base * 2.0 - 1.0, case)
        input_id = f"seed{seed}"
        assert_input(case, x, input_id)
        yield input_id, x.to(device)
    if case.zero_input:
        x = _place_input(torch.zeros(case.shape, dtype=torch.float32), case)
        assert_input(case, x, "zeros")
        yield "zeros", x.to(device)


@dataclass
class ErrorStats:
    max_abs: float
    max_rel: float
    argmax_abs: tuple[int, ...] | None
    nonfinite_out: int
    nonfinite_ref: int
    shape_ok: bool
    dtype_ok: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "argmax_abs": list(self.argmax_abs) if self.argmax_abs is not None else None,
            "nonfinite_out": self.nonfinite_out,
            "nonfinite_ref": self.nonfinite_ref,
            "shape_ok": self.shape_ok,
            "dtype_ok": self.dtype_ok,
        }


def error_stats(out: torch.Tensor, ref: torch.Tensor) -> ErrorStats:
    shape_ok = tuple(out.shape) == tuple(ref.shape)
    dtype_ok = out.dtype == ref.dtype
    out_f = out.detach()
    ref_f = ref.detach().to(out_f.device)
    if not shape_ok:
        return ErrorStats(math.inf, math.inf, None, int((~torch.isfinite(out_f)).sum()),
                          int((~torch.isfinite(ref_f)).sum()), False, dtype_ok)
    diff = (out_f.double() - ref_f.double()).abs()
    ref_abs = ref_f.double().abs()
    # Relative error denominator: |ref|, floored at the dtype eps * 8 so that
    # values that should be exactly zero (relu of a negative t) do not blow up.
    floor = max(torch.finfo(ref_f.dtype).eps * 8.0, 1e-30)
    rel = diff / torch.clamp(ref_abs, min=floor)
    nonfinite_out = int((~torch.isfinite(out_f)).sum())
    nonfinite_ref = int((~torch.isfinite(ref_f)).sum())
    if diff.numel() == 0:
        return ErrorStats(0.0, 0.0, None, nonfinite_out, nonfinite_ref, shape_ok, dtype_ok)
    flat = int(diff.argmax())
    idx = torch.unravel_index(torch.tensor(flat), diff.shape)
    return ErrorStats(
        max_abs=float(diff.max()),
        max_rel=float(rel.max()),
        argmax_abs=tuple(int(i) for i in idx),
        nonfinite_out=nonfinite_out,
        nonfinite_ref=nonfinite_ref,
        shape_ok=shape_ok,
        dtype_ok=dtype_ok,
    )


def compare(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> tuple[bool, str]:
    stats = error_stats(out, ref)
    if not stats.shape_ok:
        return False, f"shape mismatch out={tuple(out.shape)} ref={tuple(ref.shape)}"
    if not stats.dtype_ok:
        return False, f"dtype mismatch out={out.dtype} ref={ref.dtype}"
    if stats.nonfinite_out != stats.nonfinite_ref:
        return False, f"non-finite count mismatch out={stats.nonfinite_out} ref={stats.nonfinite_ref}"
    ok = torch.allclose(out.to(ref.dtype), ref, atol=atol, rtol=rtol, equal_nan=False)
    if not ok:
        return False, (f"allclose failed atol={atol} rtol={rtol} "
                       f"max_abs={stats.max_abs:.6g} max_rel={stats.max_rel:.6g}")
    return True, "ok"


def check_case(case, run_fn, strategy: str, device: str | torch.device = "cpu") -> list[dict[str, Any]]:  # noqa: ANN001
    """Run every fixed input through ``run_fn`` and compare with eager + fp64.

    ``run_fn(x) -> Tensor`` is the compiled candidate (or eager for a dry run).
    Returns one dict per input, shaped like checks.jsonl.
    """
    rows: list[dict[str, Any]] = []
    for input_id, x in iter_inputs(case, device):
        with torch.no_grad():
            out = run_fn(x)
            ref = templates.reference(case, x)
            ok, reason = compare(out, ref, case.atol, case.rtol)
            stats = error_stats(out, ref)
            row: dict[str, Any] = {
                "case_id": case.case_id,
                "strategy": strategy,
                "input_id": input_id,
                "mode": "direct",
                "reference": "eager_same_dtype",
                "tolerance": {"atol": case.atol, "rtol": case.rtol},
                "out_shape": list(out.shape),
                "out_dtype": str(out.dtype).replace("torch.", ""),
                "ref_shape": list(ref.shape),
                "ref_dtype": str(ref.dtype).replace("torch.", ""),
                "nonfinite_out": stats.nonfinite_out,
                "nonfinite_ref": stats.nonfinite_ref,
                "max_abs_err": stats.max_abs,
                "max_rel_err": stats.max_rel,
                "argmax_abs_err": list(stats.argmax_abs) if stats.argmax_abs is not None else None,
                "pass": bool(ok),
                "reason": reason,
                "input_shape": list(x.shape),
                "input_stride": list(x.stride()),
                "input_dtype": str(x.dtype).replace("torch.", ""),
            }
            if case.fp64_reference:
                ref64 = templates.reference64(case, x)
                s64 = error_stats(out, ref64)
                row["fp64"] = {
                    "max_abs_err": s64.max_abs,
                    "max_rel_err": s64.max_rel,
                    "argmax_abs_err": list(s64.argmax_abs) if s64.argmax_abs is not None else None,
                    # Statistical accumulation bound for the reduction length; a
                    # diagnostic, not a gate (the gate is the eager tolerance table).
                    "accumulation_bound": (
                        8.0 * torch.finfo(torch.float32).eps * case.shape[1]
                        if case.kind == "reduction" else None
                    ),
                }
        rows.append(row)
    return rows


def naive_reduction_bound(case) -> float | None:  # noqa: ANN001
    """Worst-case bound for a sequential fp32 sum over a row of length K."""
    if case.kind != "reduction" or case.template != "R1":
        return None
    k = case.shape[1]
    # gamma_n = n*eps/(1-n*eps)
    eps = torch.finfo(torch.float32).eps
    return (k * eps) / (1.0 - k * eps) if k * eps < 1 else math.inf
