"""Run directory layout, JSONL writers, hashes and environment snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Iterable


# --------------------------------------------------------------------------
# basic helpers
# --------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tail_text(path: str, max_bytes: int = 64 * 1024) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", "replace")


def copy_file(src: str, dst: str) -> str:
    import shutil

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


# --------------------------------------------------------------------------
# repository / environment snapshots
# --------------------------------------------------------------------------
def git_state(repo_root: str | None = None) -> dict[str, Any]:
    repo_root = repo_root or os.getcwd()

    def run(*args: str) -> str:
        try:
            return subprocess.run(("git",) + args, cwd=repo_root, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception as exc:  # pragma: no cover - git absent
            return f"<error: {exc}>"

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        "diff_stat": run("diff", "--stat"),
    }


def gpu_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"available": False}
    try:
        import torch

        if not torch.cuda.is_available():
            snap["reason"] = "torch.cuda.is_available() is False"
            return snap
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        snap.update(
            available=True,
            index=index,
            name=torch.cuda.get_device_name(index),
            uuid=str(getattr(props, "uuid", "unknown")),
            capability=f"{props.major}.{props.minor}",
            total_memory_bytes=int(props.total_memory),
            multi_processor_count=int(props.multi_processor_count),
            l2_cache_bytes=int(getattr(props, "L2_cache_size", 0)) or None,
            cuda_runtime=str(torch.version.cuda),
            cudnn=str(torch.backends.cudnn.version()),
        )
    except Exception as exc:  # pragma: no cover
        snap["reason"] = f"probe failed: {exc}"
        return snap
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        snap["nvidia_smi"] = out
    except Exception as exc:  # pragma: no cover
        snap["nvidia_smi"] = f"<error: {exc}>"
    return snap


def env_snapshot() -> dict[str, Any]:
    import torch

    env: dict[str, Any] = {
        "timestamp": utc_now(),
        "python": sys_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "inductor_source": source_of("torch._inductor.scheduler"),
        "dynamo_source": source_of("torch._dynamo"),
        "gpu": gpu_snapshot(),
    }
    try:
        import triton

        env["triton_version"] = triton.__version__
        env["triton_source"] = os.path.dirname(triton.__file__)
    except Exception as exc:  # pragma: no cover
        env["triton_version"] = f"<error: {exc}>"
    env["git"] = git_state()
    return env


def sys_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def source_of(module_name: str) -> dict[str, Any]:
    """File path + sha256 of a module's source, and its version if any."""
    import importlib
    from importlib import util as importlib_util

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover
        return {"module": module_name, "error": str(exc)}
    path = getattr(module, "__file__", None)
    info: dict[str, Any] = {
        "module": module_name,
        "file": path,
        "file_sha256": sha256_file(path) if path and os.path.exists(path) else None,
    }
    return info


def inductor_config_snapshot() -> dict[str, Any]:
    """Actual values of the Inductor/config knobs that change partitioning.

    Values are read back from ``torch._inductor.config`` at compile time, not
    assumed, so a version bump that changes a default is visible in the record.
    """
    import torch._inductor.config as cfg

    keys = [
        ("max_autotune", lambda: cfg.max_autotune),
        ("max_autotune_gemm", lambda: cfg.max_autotune_gemm),
        ("triton.cudagraphs", lambda: cfg.triton.cudagraphs),
        ("triton.autotune_pointwise", lambda: cfg.triton.autotune_pointwise),
        ("triton.autotune_cublasLt", lambda: cfg.triton.autotune_cublasLt),
        ("triton.persistent_reductions", lambda: cfg.triton.persistent_reductions),
        ("max_fusion_size", lambda: cfg.max_fusion_size),
        ("realize_reads_threshold", lambda: cfg.realize_reads_threshold),
        ("realize_acc_reads_threshold", lambda: cfg.realize_acc_reads_threshold),
        ("benchmark_harness", lambda: cfg.benchmark_harness),
        ("deterministic", lambda: cfg.deterministic),
        ("fallback_random", lambda: cfg.fallback_random),
        ("freezing", lambda: cfg.freezing),
        ("size_asserts", lambda: cfg.size_asserts),
        ("shape_padding", lambda: cfg.shape_padding),
        ("debug", lambda: cfg.debug),
        ("debug_graph_code", lambda: cfg.debug_graph_code),
        ("trace.enabled", lambda: cfg.trace.enabled),
    ]
    snapshot: dict[str, Any] = {}
    for name, getter in keys:
        try:
            snapshot[name] = getter()
        except Exception as exc:  # pragma: no cover
            snapshot[name] = f"<unavailable: {exc}>"
    return snapshot


# --------------------------------------------------------------------------
# run layout
# --------------------------------------------------------------------------
class RunDir:
    """runs/<run_id>/ layout from plan/phase1-3090/04-data.md."""

    def __init__(self, root: str, run_id: str):
        self.root = os.path.abspath(root)
        self.run_id = run_id
        self.path = os.path.join(self.root, run_id)
        for sub in ("artifacts", "logs", "evidence", "freeze"):
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)

    # --- table paths
    @property
    def run_json(self) -> str:
        return os.path.join(self.path, "run.json")

    @property
    def cases_jsonl(self) -> str:
        return os.path.join(self.path, "cases.jsonl")

    @property
    def variants_jsonl(self) -> str:
        return os.path.join(self.path, "variants.jsonl")

    @property
    def checks_jsonl(self) -> str:
        return os.path.join(self.path, "checks.jsonl")

    @property
    def profiles_jsonl(self) -> str:
        return os.path.join(self.path, "profiles.jsonl")

    @property
    def timings_jsonl(self) -> str:
        return os.path.join(self.path, "timings.jsonl")

    def artifact_dir(self, case_id: str, strategy: str) -> str:
        d = os.path.join(self.path, "artifacts", case_id, strategy)
        os.makedirs(d, exist_ok=True)
        return d

    def log_path(self, name: str) -> str:
        return os.path.join(self.path, "logs", name)

    def evidence_path(self, name: str) -> str:
        return os.path.join(self.path, "evidence", name)
