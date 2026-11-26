"""GPU telemetry sampling and per-block contamination assessment."""
from __future__ import annotations

import csv
import io
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .util import utcnow

GPU_QUERY_FIELDS = [
    "timestamp",
    "name",
    "uuid",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "clocks_throttle_reasons.active",
    "utilization.gpu",
    "memory.used",
]
APPS_QUERY = "pid,process_name,used_memory"


def run_nvidia_smi(args: list[str], timeout: float = 5.0) -> str:
    p = subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout
    )
    if p.returncode != 0:
        raise RuntimeError(f"nvidia-smi {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


class TelemetrySampler:
    def __init__(self, ctx=None, interval_s: float = 1.0, device_index: int = 0):
        self.ctx = ctx
        self.interval_s = interval_s
        self.device_index = device_index
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.source = "nvidia-smi"
        self.available = False
        self.error: str | None = None

    def _sample_once(self) -> list[dict]:
        ts = time.time()
        iso = utcnow()
        out = run_nvidia_smi(
            [
                f"--id={self.device_index}",
                f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ]
        ).strip()
        rows: list[dict] = []
        if out:
            values = next(csv.reader(io.StringIO(out)))
            if len(values) != len(GPU_QUERY_FIELDS):
                raise RuntimeError(f"unexpected nvidia-smi gpu row: {out!r}")
            row = {
                "kind": "gpu",
                "ts_utc": iso,
                "ts_unix": ts,
                "source": self.source,
            }
            row.update(dict(zip(GPU_QUERY_FIELDS, [v.strip() for v in values])))
            rows.append(row)
        apps_out = run_nvidia_smi(
            [
                f"--id={self.device_index}",
                f"--query-compute-apps={APPS_QUERY}",
                "--format=csv,noheader,nounits",
            ]
        ).strip()
        procs = []
        for line in apps_out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",", 2)]
            try:
                pid = int(parts[0])
            except ValueError:
                pid = -1
            procs.append(
                {
                    "pid": pid,
                    "process_name": parts[1] if len(parts) > 1 else None,
                    "used_memory": parts[2] if len(parts) > 2 else None,
                }
            )
        rows.append(
            {
                "kind": "apps",
                "ts_utc": iso,
                "ts_unix": ts,
                "source": self.source,
                "processes": procs,
            }
        )
        return rows

    def start(self) -> "TelemetrySampler":
        try:
            rows = self._sample_once()
            self.rows.extend(rows)
            if self.ctx is not None:
                self.ctx.append("telemetry", rows)
            self.available = True
        except Exception as exc:  # missing telemetry must not block timing
            self.error = str(exc)
            self.available = False
            return self
        self._thread = threading.Thread(target=self._loop, name="fb-telemetry", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                rows = self._sample_once()
            except Exception as exc:
                self.error = str(exc)
                continue
            self.rows.extend(rows)
            if self.ctx is not None:
                try:
                    self.ctx.append("telemetry", rows)
                except Exception as exc:
                    self.error = str(exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.ctx is not None:
            try:
                self.ctx.close_writers("telemetry")
            except Exception:
                pass

    # ------------------------------------------------------------------
    def assess(self, t0: float, t1: float, our_pids: set[int]) -> dict[str, Any]:
        if not self.available:
            return {
                "quality": "ok",
                "reasons": ["telemetry_unavailable"],
                "detail": self.error,
                "sample_count": 0,
            }
        window = [r for r in self.rows if t0 - 0.001 <= r.get("ts_unix", 0) <= t1 + 0.001]
        gpu_all = [r for r in self.rows if r["kind"] == "gpu"]
        app_all = [r for r in self.rows if r["kind"] == "apps"]
        before_gpu = [r for r in gpu_all if r.get("ts_unix", 0) < t0]
        after_gpu = [r for r in gpu_all if r.get("ts_unix", 0) > t1]
        before_app = [r for r in app_all if r.get("ts_unix", 0) < t0]
        after_app = [r for r in app_all if r.get("ts_unix", 0) > t1]
        reasons: list[str] = []
        detail: dict[str, Any] = {}
        # A ~10 ms block cannot be sampled by a 1 Hz loop. Only samples close to
        # the window are used as evidence; stale samples would report the
        # idle<->load transition that happened once before the block.
        fresh = max(1.5 * self.interval_s, 1.5)
        before = before_gpu[-1] if before_gpu and t0 - before_gpu[-1]["ts_unix"] <= fresh else None
        after = after_gpu[0] if after_gpu and after_gpu[0]["ts_unix"] - t1 <= fresh else None
        inside = [r for r in window if r["kind"] == "gpu"]
        gpu_rows = [r for r in ([before] if before else []) + inside + ([after] if after else [])]
        if not gpu_rows:
            reasons.append("telemetry_gap")
        app_rows = ([before_app[-1]] if before_app and t0 - before_app[-1]["ts_unix"] <= fresh else []) + [
            r for r in window if r["kind"] == "apps"
        ] + ([after_app[0]] if after_app and after_app[0]["ts_unix"] - t1 <= fresh else [])
        # External CUDA processes.
        external: set[tuple[int, str | None]] = set()
        for r in app_rows:
            for proc in r.get("processes", []):
                if proc["pid"] > 0 and proc["pid"] not in our_pids:
                    external.add((proc["pid"], proc.get("process_name")))
        if external:
            reasons.append("external_cuda_process")
            detail["external_processes"] = sorted(
                ({"pid": p, "name": n} for p, n in external), key=lambda d: d["pid"]
            )
        # Throttle reasons: bit 0 is GpuIdle and flips on every idle<->load
        # boundary, so only the real slow-down bits count as a state change.
        def _active(mask: str | None) -> int | None:
            try:
                return int(str(mask), 16) & ~0x1
            except (TypeError, ValueError):
                return None

        masks = [(r.get("clocks_throttle_reasons.active") or "").strip() for r in gpu_rows]
        active = [m for m in (_active(x) for x in masks) if m is not None]
        if len(active) >= 2 and len(set(active)) > 1:
            reasons.append("throttle_state_change")
            detail["throttle_masks"] = masks
            detail["throttle_active_bits"] = active

        def _num(row, key):
            try:
                return float(row.get(key))
            except (TypeError, ValueError):
                return None

        temps = [v for v in (_num(r, "temperature.gpu") for r in gpu_rows) if v is not None]
        powers = [v for v in (_num(r, "power.draw") for r in gpu_rows) if v is not None]
        detail.update(
            {
                "samples": len(window),
                "bracketed": bool(before or after),
                "temp_c_max": max(temps) if temps else None,
                "power_w_max": max(powers) if powers else None,
            }
        )
        quality = "contaminated" if reasons and reasons != ["telemetry_gap"] else "ok"
        return {
            "quality": quality,
            "reasons": reasons,
            "detail": detail,
            "sample_count": len(window),
        }

    def settle(self) -> None:
        """Wait for one fresh sample after a contamination-flagged attempt."""
        time.sleep(min(2.0, self.interval_s + 0.2))


def list_external_gpu_processes(our_pids: set[int]) -> list[dict]:
    """Compute-app rows on the visible device excluding our own PIDs."""
    try:
        out = run_nvidia_smi(
            ["--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"]
        )
    except Exception:
        return []
    external = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",", 2)]
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid > 0 and pid not in our_pids:
            external.append({"pid": pid,
                             "process_name": parts[1] if len(parts) > 1 else None,
                             "used_memory": parts[2] if len(parts) > 2 else None})
    return external


def our_process_ids() -> set[int]:
    """Our PID plus direct children that may hold the CUDA context."""
    pids = {os.getpid()}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            if ppid == os.getpid():
                pids.add(int(entry.name))
    except OSError:
        pass
    return pids
