"""Tiny target for an optional Nsight Compute run (invoked by ncu)."""
from __future__ import annotations

import argparse

from .backend import CudaBackend
from .provider import load_provider


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--strategy", required=True, choices=["split", "fused", "default"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ns = ap.parse_args()
    provider = load_provider(ns.provider)
    case = next(c for c in provider.catalog() if c["case_id"] == ns.case)
    backend = CudaBackend(ns.device)
    v = provider.prepare(case, ns.strategy)
    backend.warmup(v, 5)
    for _ in range(ns.iters):
        v.call()
    backend.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
