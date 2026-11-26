"""Candidate provider contract (interface with compilation owner A).

A provider is a directory with ``provider.py`` exposing::

    def make_provider(root: pathlib.Path, options: dict) -> Provider

``Provider`` provides:

    catalog() -> list[dict]                 # data-contract case rows
    prepare(case: dict, strategy: str) -> PreparedVariant
    close() -> None

``PreparedVariant`` is ``fb.backend.PreparedVariant``; ``meta`` must carry:

    artifact_sha256: str        hash of the compiled artifact set (required
                                for formal runs; null means unverified)
    n_kernels: int              static launch count for one call
    preallocated: bool          true iff input/output addresses are fixed
    compile_ms: float
    artifacts: list[str]        paths relative to the provider root
    kernel_list: list[dict]     optional per-kernel grid/block/resources

``variant.meta`` may also include ``default_equivalent`` (strategy name) and
``default_partition`` ("same" | "distinct") for the default-vs-split/fused map.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


class ProviderError(RuntimeError):
    pass


def _load_module_from_dir(root: Path):
    path = root / "provider.py"
    if not path.exists():
        raise ProviderError(f"provider.py not found under {root}")
    name = f"fb_provider_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProviderError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_provider(spec: str, options: dict | None = None):
    """``spec`` is ``reference``, ``fusionboundary:<candidate-run>`` or a
    directory holding either ``provider.py`` (custom) or ``variants.jsonl``
    (Agent A's candidate run)."""
    if spec == "reference":
        from .reference_provider import make_provider

        return make_provider(Path("."), options or {})
    if spec.startswith("fusionboundary:"):
        from .fusionboundary_provider import FusionBoundaryProvider

        return FusionBoundaryProvider(spec.split(":", 1)[1], options or {})
    root = Path(spec)
    if (root / "variants.jsonl").exists() and not (root / "provider.py").exists():
        from .fusionboundary_provider import FusionBoundaryProvider

        return FusionBoundaryProvider(root, options or {})
    mod = _load_module_from_dir(root)
    if not hasattr(mod, "make_provider"):
        raise ProviderError(f"{root}/provider.py has no make_provider()")
    return mod.make_provider(root, options or {})


REQUIRED_META = ("artifact_sha256", "n_kernels")


def variant_contract_problems(meta: dict, formal: bool) -> list[str]:
    problems: list[str] = []
    for key in REQUIRED_META:
        if key not in meta:
            problems.append(f"missing meta.{key}")
    if formal:
        if not meta.get("artifact_sha256"):
            problems.append("artifact_sha256 is null")
        if not (meta.get("preallocated_inputs") is True or meta.get("preallocated") is True):
            problems.append("input addresses are not declared fixed")
        if not isinstance(meta.get("n_kernels"), int) or meta.get("n_kernels", 0) < 1:
            problems.append("n_kernels is not a positive int")
        if not meta.get("artifacts"):
            problems.append("no artifact paths recorded")
    return problems
