"""Environment capture, device verification and lock export."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .rundir import RunContext
from .util import (
    atomic_write_json,
    atomic_write_text,
    git_state,
    read_json,
    sha256_file,
    sha256_json,
    sha256_text,
    utcnow,
)

QUERY_FIELDS = [
    "name",
    "uuid",
    "compute_cap",
    "memory.total",
    "memory.used",
    "driver_version",
    "vbios_version",
    "power.limit",
    "power.max_limit",
    "clocks.max.sm",
    "clocks.max.mem",
    "pcie.link.gen.max",
    "pcie.link.width.max",
    "serial",
    "pci.bus_id",
]

CONFIG_READS = [
    ("torch.is_grad_enabled", lambda t: t.is_grad_enabled()),
    ("torch.are_deterministic_algorithms_enabled", lambda t: t.are_deterministic_algorithms_enabled()),
    ("torch.get_float32_matmul_precision", lambda t: t.get_float32_matmul_precision()),
    ("torch.backends.cuda.matmul.allow_tf32", lambda t: t.backends.cuda.matmul.allow_tf32),
    ("torch.backends.cudnn.allow_tf32", lambda t: t.backends.cudnn.allow_tf32),
    ("torch.backends.cudnn.benchmark", lambda t: t.backends.cudnn.benchmark),
    ("torch.backends.cudnn.deterministic", lambda t: t.backends.cudnn.deterministic),
    ("torch.get_autocast_dtype('cuda')", lambda t: str(t.get_autocast_dtype("cuda"))),
]


def _sh(cmd: list[str], cwd: str | None = None) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:
        return {"cmd": cmd, "error": str(exc)}


def _freeze(python: str) -> str:
    """Package freeze without requiring pip in the venv (uv-first)."""
    candidates = [
        ["uv", "pip", "freeze", "--python", python],
        [python, "-m", "pip", "freeze"],
    ]
    for cmd in candidates:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout
    try:
        from importlib import metadata

        lines = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in metadata.distributions()
            if d.metadata.get("Name")
        )
        return "\n".join(lines) + ("\n" if lines else "")
    except Exception:
        return ""


def _pkg(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def collect_environment(device_index: int = 0, repo_root: str | None = None) -> dict:
    env: dict[str, Any] = {"captured_utc": utcnow(), "sys": {}, "gpu": {}, "python": {}, "torch": {}, "cuda_config": {}, "raw": {}}
    env["sys"] = {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pid": os.getpid(),
    }
    osrel = Path("/etc/os-release")
    env["sys"]["os_release"] = (
        dict(
            line.split("=", 1)
            for line in osrel.read_text().splitlines()
            if "=" in line
        )
        if osrel.exists()
        else None
    )
    env["raw"]["nvidia_smi_L"] = _sh(["nvidia-smi", "-L"])
    env["raw"]["nvidia_smi_q"] = _sh(["nvidia-smi", "-q"])
    env["raw"]["nvidia_smi_query_gpu"] = _sh(
        [
            "nvidia-smi",
            f"--id={device_index}",
            f"--query-gpu={','.join(QUERY_FIELDS)}",
            "--format=csv,noheader",
        ]
    )
    env["raw"]["nvidia_smi_compute_apps"] = _sh(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"]
    )
    env["raw"]["lscpu"] = _sh(["lscpu"])
    env["raw"]["findmnt_run_root"] = _sh(["findmnt", "-T", os.getcwd()])
    env["raw"]["uname"] = _sh(["uname", "-a"])
    env["raw"]["free"] = _sh(["free", "-b"])

    # Parse the machine-readable GPU line.
    line = env["raw"]["nvidia_smi_query_gpu"].get("stdout", "").strip().splitlines()
    if line:
        import csv
        import io

        vals = next(csv.reader(io.StringIO(line[0])))
        if len(vals) == len(QUERY_FIELDS):
            env["gpu"] = dict(zip(QUERY_FIELDS, [v.strip() for v in vals]))
    env["gpu"]["visible_devices_env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    env["gpu"]["device_index"] = device_index

    env["python"]["packages"] = {
        name: _pkg(name)
        for name in ("torch", "triton", "numpy", "pynvml", "nvidia-ml-py", "setuptools")
    }
    env["python"]["pip_freeze"] = _freeze(sys.executable)

    try:
        import torch

        env["torch"] = {
            "version": torch.__version__,
            "git_version": getattr(torch.version, "git_version", None),
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "arch_list": torch.cuda.get_arch_list(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_index)
            uuid = getattr(props, "uuid", None)
            env["torch"]["device"] = {
                "name": props.name,
                "uuid": f"GPU-{uuid}" if uuid else None,
                "capability": [props.major, props.minor],
                "sm_count": props.multi_processor_count,
                "total_memory_bytes": props.total_memory,
                "l2_cache_bytes": getattr(props, "L2_cache_size", None),
                "regs_per_multiprocessor": getattr(props, "regs_per_multiprocessor", None),
                "max_threads_per_multi_processor": getattr(props, "max_threads_per_multi_processor", None),
                "shared_memory_per_block_optin": getattr(props, "shared_memory_per_block_optin", None),
            }
            torch.cuda.init()
            env["cuda_config"] = {
                name: _safe_get(fn, torch) for name, fn in CONFIG_READS
            }
            env["cuda_config"]["current_stream"] = str(torch.cuda.current_stream())
            try:
                env["cuda_config"]["amp_autocast_enabled"] = torch.is_autocast_enabled("cuda")
            except Exception as exc:
                env["cuda_config"]["amp_autocast_enabled"] = {"error": str(exc)}
            try:
                from torch.cuda.memory import get_allocator_backend

                env["cuda_config"]["allocator_backend"] = get_allocator_backend()
            except Exception as exc:
                env["cuda_config"]["allocator_backend"] = {"error": str(exc)}
            try:
                import torch._inductor.config as ind

                env["cuda_config"]["inductor"] = {
                    "triton.cudagraphs": ind.triton.cudagraphs,
                    "triton.cudagraph_trees": getattr(ind.triton, "cudagraph_trees", None),
                    "max_autotune": ind.max_autotune,
                    "coordinate_descent_tuning": ind.coordinate_descent_tuning,
                    "fx_graph_cache": getattr(ind, "fx_graph_cache", None),
                }
            except Exception as exc:
                env["cuda_config"]["inductor"] = {"error": str(exc)}
            env["cuda_config"]["env"] = {
                k: v
                for k, v in os.environ.items()
                if re.match(r"^(TORCH|TRITON|CUDA|CUBLAS|CUDNN|PYTORCH|NVIDIA|NCCL)", k)
            }
            env["cuda_config"]["cache_dirs"] = {
                "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
                "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
            }
    except Exception as exc:
        env["torch"] = {"error": f"{type(exc).__name__}: {exc}"}

    root = Path(repo_root or os.getcwd())
    env["git"] = git_state(root)
    req = root / "requirements.lock"
    env["repo"] = {
        "root": str(root),
        "requirements_lock": str(req) if req.exists() else None,
        "requirements_lock_sha256": sha256_file(req) if req.exists() else None,
        "requirements_lock_text": req.read_text(encoding="utf-8") if req.exists() else None,
    }
    disk = _sh(["df", "-B1", "."])
    env["sys"]["df_bytes"] = disk.get("stdout")
    env["mount"] = _sh(["findmnt", "-T", os.getcwd(), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS", "-n"]).get("stdout", "").strip()
    return env


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _safe_get(fn, torch):
    try:
        return fn(torch)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def save_environment(ctx: RunContext, env: dict) -> dict:
    out = ctx.run_dir / "environment"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "environment.json", env)
    for name, payload in env.get("raw", {}).items():
        if isinstance(payload, dict):
            text = payload.get("stdout") or ""
            if payload.get("stderr"):
                text += "\n--- stderr ---\n" + payload["stderr"]
            if payload.get("error"):
                text += "\n--- error ---\n" + payload["error"]
            atomic_write_text(out / f"{name}.txt", text)
    freeze = env.get("python", {}).get("pip_freeze")
    if freeze:
        atomic_write_text(out / "requirements.freeze.txt", freeze)
    lock = {
        "captured_utc": env.get("captured_utc"),
        "python": env.get("sys", {}).get("python"),
        "packages": env.get("python", {}).get("packages"),
        "torch": env.get("torch", {}).get("version"),
        "torch_git": env.get("torch", {}).get("git_version"),
        "torch_cuda_build": env.get("torch", {}).get("cuda_build"),
        "cudnn": env.get("torch", {}).get("cudnn"),
        "triton": env.get("python", {}).get("packages", {}).get("triton"),
        "driver": env.get("gpu", {}).get("driver_version"),
        "gpu": env.get("gpu"),
        "git": {
            "commit": env.get("git", {}).get("commit"),
            "dirty": env.get("git", {}).get("dirty"),
            "diff_sha256": env.get("git", {}).get("diff_sha256"),
        },
        "cuda_config": env.get("cuda_config"),
        "freeze_sha256": sha256_text(freeze or ""),
        "requirements_lock": env.get("repo", {}).get("requirements_lock"),
        "requirements_lock_sha256": env.get("repo", {}).get("requirements_lock_sha256"),
    }
    atomic_write_json(out / "environment.lock.json", lock)
    if env.get("git", {}).get("diff_text"):
        atomic_write_text(out / "git.diff", env["git"]["diff_text"])
    ctx.update(environment={"path": "environment/environment.json", "lock_sha256": sha256_file(out / "environment.lock.json"), "gpu": env.get("gpu")})
    return lock


def verify_device(env: dict, expect: str | None, expect_uuid: str | None) -> dict:
    problems: list[str] = []
    gpu = env.get("gpu", {})
    tdev = env.get("torch", {}).get("device", {}) if isinstance(env.get("torch"), dict) else {}
    names = [gpu.get("name"), tdev.get("name")]
    uuids = [gpu.get("uuid"), tdev.get("uuid")]
    if expect:
        if not any(n and expect.lower() in str(n).lower() for n in names):
            problems.append(f"GPU name(s) {names} do not match expected {expect!r}")
    if expect_uuid:
        if not any(u and u == expect_uuid for u in uuids):
            problems.append(f"GPU uuid(s) {uuids} do not match expected {expect_uuid!r}")
    if not env.get("torch", {}).get("cuda_available"):
        problems.append("torch.cuda.is_available() is false")
    return {
        "expected_name_substring": expect,
        "expected_uuid": expect_uuid,
        "observed_names": names,
        "observed_uuids": uuids,
        "pass": not problems,
        "problems": problems,
        "checked_utc": utcnow(),
    }
