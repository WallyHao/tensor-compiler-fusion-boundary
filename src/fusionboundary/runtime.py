"""Stable load/call interface for the measurement owner (Agent B).

Usage
-----
>>> from fusionboundary.runtime import CandidateStore
>>> store = CandidateStore("runs/pilot-20261125")     # indexes variants.jsonl
>>> split = store.load("R1-m256-k512-f32-contig-t", "split")
>>> fused = store.load("R1-m256-k512-f32-contig-t", "fused")
>>> split.report.summary()
{'case_id': ..., 'strategy': 'split', 'kernels': 2, 'cache_reused': True, ...}
>>> x = split.fixed_input()          # device tensor, address-stable, owned by the handle
>>> y = split.call(x)                # exactly one call; no compile, no autotune
>>> y2 = fused.call(fused.fixed_input())

Lifecycle contract
------------------
* ``load()`` verifies and loads the archived Inductor wrapper, then warms it
  once. Triton may load or compile the embedded kernel sources at this stage.
  The graph is not run through Dynamo or the fusion scheduler again.
* ``fixed_input()`` returns the handle-owned tensor for seed 17 (or the zeros
  input via ``inputs[input_id]``). Same content and same device address on
  every access. Do not mutate it; it is shared by all checks and by replay.
* ``call(x)`` performs one steady-state call: the wrapper's own buffer
  allocations and kernel launches only. It returns a freshly allocated output
  tensor. Drop the previous output before the next call if you need stable
  addresses (CUDA Graph capture); ``call_latest()`` keeps exactly one output.
* ``fresh_inputs(seed)`` allocates new inputs; use it only outside timed
  regions (it changes allocator state).
* ``capture_replay()`` captures one whole call into a CUDA graph and returns a
  handle whose ``call()`` is a single ``graph.replay()``; the replay uses the
  same fixed input and a private output buffer.

"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch

from . import records, strategies, templates


@contextlib.contextmanager
def _artifact_cache(path: str):
    old = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = path
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = old


def cache_dir_snapshot(path: str) -> dict[str, Any]:
    snapshot: dict[str, tuple[int, int]] = {}
    if not os.path.isdir(path):
        return snapshot
    for root, _dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            snapshot[os.path.relpath(p, path)] = (int(st.st_mtime_ns), st.st_size)
    return snapshot


def snapshot_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {"added": added, "removed": removed, "modified": modified,
            "reused": not (added or removed or modified)}


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
@dataclass
class LoadReport:
    case_id: str
    strategy: str
    status: str
    artifact_dir: str
    kernels: list[str] = field(default_factory=list)
    kernel_count: int = 0
    cache_dir: str | None = None
    cache_delta: dict[str, Any] = field(default_factory=dict)
    first_call_ms: float | None = None
    triggered: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    evidence_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    cache_activation: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_reused(self) -> bool:
        return bool(self.cache_delta.get("reused"))

    def summary(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "strategy": self.strategy,
            "status": self.status,
            "kernels": self.kernel_count,
            "kernel_names": self.kernels,
            "cache_reused": self.cache_reused,
            "cache_added_files": len(self.cache_delta.get("added", [])),
            "cache_added_paths": self.cache_delta.get("added", [])[:20],
            "cache_modified_paths": self.cache_delta.get("modified", [])[:20],
            "cache_activation": self.cache_activation,
            "first_call_ms": self.first_call_ms,
            "triggered": self.triggered,
            "evidence": self.evidence_path,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# candidate handle
# --------------------------------------------------------------------------
class Candidate:
    """One compiled (case, strategy) candidate. See module docstring."""

    def __init__(self, case, strategy: str, artifact_dir: str, compiled,  # noqa: ANN001
                 report: LoadReport, device: str):
        self.case = case
        self.strategy = strategy
        self.artifact_dir = artifact_dir
        self._compiled = compiled
        self.report = report
        self.device = device
        self._inputs: dict[str, torch.Tensor] = {}
        self._latest: torch.Tensor | None = None
        self.calls = 0
        self._prepare_fixed_inputs()

    # -- inputs -----------------------------------------------------------
    def _prepare_fixed_inputs(self) -> None:
        from .validate import iter_inputs

        for input_id, x in iter_inputs(self.case, self.device):
            self._inputs[input_id] = x

    @property
    def inputs(self) -> dict[str, torch.Tensor]:
        return dict(self._inputs)

    def fixed_input(self, input_id: str = "seed17") -> torch.Tensor:
        return self._inputs[input_id]

    def fresh_inputs(self, seed: int) -> torch.Tensor:
        """New allocation outside the timed region (changes allocator state)."""
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        base = torch.rand(self.case.shape, generator=gen, dtype=torch.float32)
        return ((base * 2.0 - 1.0).to(self.case.torch_dtype())).to(self.device)

    # -- calls ------------------------------------------------------------
    def call(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Exactly one call. No compile, no autotune, no input allocation."""
        if x is None:
            x = self.fixed_input()
        return self._compiled(x)

    def call_latest(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """One call keeping exactly one live output (stable allocator footprint)."""
        self._latest = None
        out = self.call(x)
        self._latest = out
        return out

    def call_n(self, n: int, x: torch.Tensor | None = None) -> None:
        """n back-to-back calls, dropping each output (no host sync, no alloc churn)."""
        if x is None:
            x = self.fixed_input()
        fn = self._compiled
        for _ in range(int(n)):
            out = fn(x)
            del out

    def __call__(self, x: torch.Tensor | None = None) -> torch.Tensor:
        self.calls += 1
        return self.call(x)

    # -- validation -------------------------------------------------------
    def verify(self) -> list[dict[str, Any]]:
        from .validate import check_case

        return check_case(self.case, lambda t: self.call(t), self.strategy, self.device)

    # -- replay -----------------------------------------------------------
    def capture_replay(self, warmup: int = 3) -> "ReplayCandidate":
        """Capture one whole call into a CUDA Graph (mechanism control mode)."""
        if self.device != "cuda":
            raise RuntimeError("replay capture requires cuda")
        x = self.fixed_input()
        for _ in range(int(warmup)):
            self.call_latest(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        out_static = None
        with torch.cuda.graph(graph):
            out_static = self.call(x)
        torch.cuda.synchronize()
        return ReplayCandidate(self, graph, x, out_static)

    # -- introspection ----------------------------------------------------
    @property
    def logical_edge_bytes(self) -> int:
        return self.case.logical_save_bytes

    @property
    def expected_output(self) -> dict[str, Any]:
        return {"shape": list(self.case.out_shape), "dtype": self.case.out_dtype}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"<Candidate {self.case.case_id} {self.strategy} "
                f"kernels={self.report.kernel_count} cache_reused={self.report.cache_reused}>")


class ReplayCandidate:
    """CUDA-graph replay of one full call of an already compiled candidate."""

    def __init__(self, parent: Candidate, graph, x: torch.Tensor, out: torch.Tensor):
        self.parent = parent
        self.graph = graph
        self.x = x
        self.out = out
        self.case = parent.case
        self.strategy = parent.strategy
        self.report = parent.report

    def call(self) -> torch.Tensor:
        self.graph.replay()
        return self.out

    def call_n(self, n: int) -> None:
        for _ in range(int(n)):
            self.graph.replay()

    def verify(self) -> list[dict[str, Any]]:
        """Replay once after capture and compare against eager on the same input."""
        from .validate import compare

        self.graph.replay()
        ref = templates.reference(self.case, self.x)
        ok, reason = compare(self.out, ref, self.case.atol, self.case.rtol)
        return [{
            "case_id": self.case.case_id,
            "strategy": self.strategy,
            "input_id": "seed17",
            "mode": "replay",
            "pass": bool(ok),
            "reason": reason,
        }]


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
class CandidateStore:
    """Resolve case_id + strategy to a compiled candidate recorded in a run."""

    def __init__(self, run_path: str):
        self.run_path = os.path.abspath(run_path)
        self._variants: dict[tuple[str, str], dict[str, Any]] = {}
        path = os.path.join(self.run_path, "variants.jsonl")
        if os.path.exists(path):
            for row in records.read_jsonl(path):
                if row.get("status") in ("valid", "cache_only"):
                    self._variants[(row["case_id"], row["strategy"])] = row
        self._cache: dict[tuple[str, str, str], Candidate] = {}

    # -- lookup -----------------------------------------------------------
    def status(self, case_id: str, strategy: str) -> str | None:
        row = self._variants.get((case_id, strategy))
        return None if row is None else row.get("status")

    def variants(self, case_id: str) -> dict[str, dict[str, Any]]:
        return {s: r for (c, s), r in self._variants.items() if c == case_id}

    def case_ids(self) -> list[str]:
        return sorted({c for (c, _s) in self._variants})

    def release_case(self, case_id: str) -> None:
        for key in [key for key in self._cache if key[0] == case_id]:
            del self._cache[key]

    # -- load -------------------------------------------------------------
    def load(self, case_id: str, strategy: str, device: str | None = None) -> Candidate:
        device = device or _default_device()
        key = (case_id, strategy, device)
        if key in self._cache:
            return self._cache[key]
        row = self._variants.get((case_id, strategy))
        if row is None:
            raise KeyError(f"no valid variant for {case_id}/{strategy} in {self.run_path}")
        artifact = Path(self.run_path) / row["artifact_dir"]
        compile_key = json.loads((artifact / "compile_key.json").read_text(encoding="utf-8"))
        _check_compatible(compile_key)
        if compile_key.get("device") != torch.device(device).type:
            raise RuntimeError(f"{case_id}/{strategy}: candidate targets {compile_key.get('device')}, got {device}")

        expected_gpu = compile_key.get("gpu") or {}
        if torch.device(device).type == "cuda" and expected_gpu.get("uuid") not in (None, "unknown"):
            actual_uuid = str(getattr(torch.cuda.get_device_properties(device), "uuid", "unknown"))
            if expected_gpu["uuid"] != actual_uuid:
                raise RuntimeError(f"{case_id}/{strategy}: recompile the candidate on the target GPU")

        from torch._inductor.codecache import PyCodeCache
        from .corpus import get_case

        case = get_case(case_id)
        source = (artifact / "output_code.py").read_bytes()
        expected_hash = row.get("output_code_sha256")
        if not expected_hash or hashlib.sha256(source).hexdigest() != expected_hash:
            raise RuntimeError(f"{case_id}/{strategy}: archived output_code.py hash mismatch")
        kernels = json.loads((artifact / "kernels.json").read_text(encoding="utf-8"))["kernels"]
        cache_dir = os.path.join(self.run_path, "cache", case_id, strategy)
        report = LoadReport(
            case_id=case_id, strategy=strategy, status=row["status"],
            artifact_dir=str(artifact), cache_dir=cache_dir,
            kernels=[k["name"] for k in kernels],
            kernel_count=len(kernels),
            config=compile_key.get("inductor_config", {}),
            evidence_path=str(artifact / "evidence.md"),
        )

        start = time.perf_counter()
        before = cache_dir_snapshot(cache_dir)
        with _artifact_cache(cache_dir), strategies.inductor_environment(
            None, compile_key.get("config_overrides", {})
        ):
            module = PyCodeCache.load(source.decode("utf-8"))

            def compiled(x):
                # The generated wrapper takes a mutable argument list and returns
                # a tuple. The corpus has exactly one input and one output.
                (out,) = module.call([x])
                return out

            with torch.no_grad():
                compiled(_first_input(case, device))
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
        report.first_call_ms = (time.perf_counter() - start) * 1e3
        report.cache_delta = snapshot_delta(before, cache_dir_snapshot(cache_dir))
        report.triggered = ["load_archived_wrapper", "first_call"]
        if not report.cache_reused:
            report.triggered.append("cache_files_changed")
        artifact_dir = str(artifact)
        cand = Candidate(case, strategy, artifact_dir, compiled, report, device)
        self._cache[key] = cand
        return cand

    def load_pair(self, case_id: str, device: str | None = None) -> dict[str, Candidate]:
        return {s: self.load(case_id, s, device) for s in ("split", "fused")
                if self.status(case_id, s) is not None}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _first_input(case, device: str) -> torch.Tensor:  # noqa: ANN001
    from .validate import iter_inputs

    return next(iter(iter_inputs(case, device)))[1]


def _release(version: str | None) -> str:
    """Release part of a version string: `2.6.0+cu124` and `2.6.0` are the same release.

    The local build tag only says which wheel was installed; the CUDA runtime
    version is compared separately below, so the two do not need to match
    character for character.
    """
    return (version or "").split("+")[0]


def _check_compatible(compile_key: dict[str, Any]) -> None:
    problems: list[str] = []
    want = compile_key.get("versions", {})
    if want.get("torch") and _release(want["torch"]) != _release(torch.__version__):
        problems.append(f"torch {torch.__version__} != recorded {want['torch']}")
    try:
        import triton

        if want.get("triton") and _release(want["triton"]) != _release(triton.__version__):
            problems.append(f"triton {triton.__version__} != recorded {want['triton']}")
    except Exception:  # pragma: no cover
        pass
    if want.get("cuda") and str(torch.version.cuda) != str(want["cuda"]):
        problems.append(f"cuda runtime {torch.version.cuda} != recorded {want['cuda']}")
    if problems:
        raise RuntimeError("candidate was compiled with a different stack: "
                           + "; ".join(problems)
                           + ". Recompile with the current environment.")
