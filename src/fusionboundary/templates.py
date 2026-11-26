"""Template expressions: one graph entry, one reference, one edge specification.

Every strategy compiles the *same* function object from ``TEMPLATE_FNS`` with the
same Inductor config; the strategies differ only in the fusion admission guard
see``strategies.py``). The reference expressions are plain eager PyTorch, i.e.
exactly the operators listed in plan/phase1-3090/02-candidates.md.

Grid in this file is only the shape grid; the case manifest lives in ``corpus``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

# Fixed scalars of the templates (exactly representable in fp16 and fp32).
P1_A, P1_B = 1.25, 0.25
P2_A, P2_B = 0.75, 0.25
R2_A, R2_B = 0.75, 0.25
# Followup RMSNorm template: inverse-RMS factor edge.
N1_EPS = 1e-5


# --------------------------------------------------------------------------
# Single-graph entries (identical for default / split / fused)
# --------------------------------------------------------------------------
def p1(x: torch.Tensor) -> torch.Tensor:
    t = x * P1_A + P1_B          # producer
    return torch.relu(t)          # consumer


def p2(x: torch.Tensor) -> torch.Tensor:
    t = torch.tanh(x)             # producer
    return t * P2_A + P2_B        # consumer


def r1(x: torch.Tensor) -> torch.Tensor:
    t = torch.sum(x, dim=1, dtype=torch.float32)   # producer (t is always fp32)
    return torch.relu(t)                            # consumer


def r2(x: torch.Tensor) -> torch.Tensor:
    t = torch.amax(x, dim=1)      # producer (t keeps input dtype)
    return t * R2_A + R2_B        # consumer


def n1(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm fragment: row mean of squares -> inverse RMS factor."""
    t = torch.mean(x.square(), dim=1)   # producer (t is fp32)
    return torch.rsqrt(t + N1_EPS)      # consumer


TEMPLATE_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "P1": p1,
    "P2": p2,
    "R1": r1,
    "R2": r2,
    "N1": n1,
}


# --------------------------------------------------------------------------
# Producer / consumer halves (diagnostics only: graph break, explicit clone)
# --------------------------------------------------------------------------
def p1_producer(x):  # noqa: ANN001
    return x * P1_A + P1_B


def p1_consumer(t):  # noqa: ANN001
    return torch.relu(t)


def p2_producer(x):  # noqa: ANN001
    return torch.tanh(x)


def p2_consumer(t):  # noqa: ANN001
    return t * P2_A + P2_B


def r1_producer(x):  # noqa: ANN001
    return torch.sum(x, dim=1, dtype=torch.float32)


def r1_consumer(t):  # noqa: ANN001
    return torch.relu(t)


def r2_producer(x):  # noqa: ANN001
    return torch.amax(x, dim=1)


def r2_consumer(t):  # noqa: ANN001
    return t * R2_A + R2_B


def n1_producer(x):  # noqa: ANN001
    return torch.mean(x.square(), dim=1)


def n1_consumer(t):  # noqa: ANN001
    return torch.rsqrt(t + N1_EPS)


PRODUCER_CONSUMER: dict[str, tuple[Callable, Callable]] = {
    "P1": (p1_producer, p1_consumer),
    "P2": (p2_producer, p2_consumer),
    "R1": (r1_producer, r1_consumer),
    "R2": (r2_producer, r2_consumer),
    "N1": (n1_producer, n1_consumer),
}


# --------------------------------------------------------------------------
# Edge specification used by the split guard and by the forensics match
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeSpec:
    """How to recognise the producer -> consumer edge ``t``.

    ``producer_ops`` / ``consumer_ops`` are substrings of the aten/buffer names
    Inductor derives from the FX graph (e.g. ``mul``, ``add``, ``relu``,
    ``sum_1``, ``amax``). ``t_dtype`` is the dtype the materialised intermediate
    must have; ``t_shape`` the shape it must have. Forensics checks both against
    the generated buffer allocation.
    """

    template: str
    producer_ops: tuple[str, ...]
    consumer_ops: tuple[str, ...]
    t_shape: tuple[int, ...]
    t_dtype: str


def edge_spec(case) -> EdgeSpec:  # noqa: ANN001 - corpus.Case
    producer_ops, consumer_ops = {
        "P1": (("mul", "add"), ("relu",)),
        "P2": (("tanh",), ("mul", "add")),
        "R1": (("sum", "reduce"), ("relu",)),
        "R2": (("amax", "max", "reduce"), ("mul", "add")),
        # t = mean(x.square(), dim=1) -> y = rsqrt(t + 1e-5). The producer may
        # be lowered as aten.mean or as sum/div; the consumer's `add` bridges
        # the two, so the producer search walks past it (strategies.
        # _find_edge_fx_nodes picks the closest input matching producer_ops).
        "N1": (("mean", "div", "sum", "mul"), ("rsqrt",)),
    }[case.template]
    return EdgeSpec(
        template=case.template,
        producer_ops=producer_ops,
        consumer_ops=consumer_ops,
        t_shape=tuple(case.t_shape),
        t_dtype=case.t_dtype,
    )


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------
def reference(case, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    """Eager execution of the same expression on the same device and dtype."""
    return TEMPLATE_FNS[case.template](x)


def reference64(case, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    """float64 reference used to quantify reduction / rounding error."""
    xd = x.double()
    if case.template == "P1":
        return torch.relu(xd * P1_A + P1_B)
    if case.template == "P2":
        return torch.tanh(xd) * P2_A + P2_B
    if case.template == "R1":
        return torch.relu(torch.sum(xd, dim=1))
    if case.template == "R2":
        return torch.amax(xd, dim=1) * R2_A + R2_B
    if case.template == "N1":
        return torch.rsqrt(torch.mean(xd.square(), dim=1) + N1_EPS)
    raise KeyError(case.template)


def producer_value(case, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    """Eager value of ``t`` (all-zero seeds make t == constants; useful for docs)."""
    return PRODUCER_CONSUMER[case.template][0](x)


def make_inputs(case, device: str | torch.device = "cpu"):  # noqa: ANN201
    """Fixed input for a case on a device: seeds 17/29/43 in [-1,1] plus zeros."""
    from .validate import iter_inputs

    return list(iter_inputs(case, device))
