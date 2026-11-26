"""Run directory, tables, protocol hashing and resume checks."""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

from .protocol import Protocol
from .util import (
    JsonlWriter,
    atomic_write_json,
    ensure_dir,
    read_json,
    read_jsonl,
    sha256_json,
    utcnow,
)

TABLES = (
    "cases",
    "variants",
    "checks",
    "calibration",
    "timings",
    "telemetry",
    "profiles",
    "results",
)


class RunError(RuntimeError):
    pass


class RunContext:
    def __init__(self, run_dir: Path, protocol: Protocol, purpose: str,
                 synthetic: bool = False):
        self.run_dir = Path(run_dir)
        self.protocol = protocol
        self.purpose = purpose
        self.synthetic = synthetic
        self.run_id = self.run_dir.name
        self.run_json_path = self.run_dir / "run.json"
        self._writers: dict[str, JsonlWriter] = {}
        self._meta: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._log_path = self.run_dir / "logs" / "runner.log"

    # ---------------- lifecycle ----------------
    @classmethod
    def create(
        cls,
        run_root: str | Path,
        run_id: str,
        protocol: Protocol,
        purpose: str,
        *,
        argv: list[str] | None = None,
        corpus_hash: str | None = None,
        parent_run: str | None = None,
        synthetic: bool = False,
        cloud: dict | None = None,
        owner: str | None = None,
    ) -> "RunContext":
        ctx = cls(Path(run_root) / run_id, protocol, purpose, synthetic)
        ensure_dir(ctx.run_dir)
        for sub in ("state", "blocks", "artifacts", "logs", "environment", "freeze"):
            ensure_dir(ctx.run_dir / sub)
        if ctx.run_json_path.exists():
            existing = read_json(ctx.run_json_path)
            if existing.get("protocol_hash") != sha256_json(protocol.to_dict()):
                raise RunError(
                    "run directory exists with a different protocol hash; "
                    "use a new --run-id or the same protocol"
                )
            ctx._meta = existing
            ctx._meta.setdefault("resumed_utcs", []).append(utcnow())
            ctx._meta["status"] = "running"
            ctx._meta["resume_count"] = int(existing.get("resume_count", 0)) + 1
            ctx._flush()
            return ctx
        ctx._meta = {
            "run_id": run_id,
            "purpose": purpose,
            "parent_run": parent_run,
            "synthetic": synthetic,
            "protocol": protocol.to_dict(),
            "protocol_hash": sha256_json(protocol.to_dict()),
            "corpus_hash": corpus_hash,
            "code": {},
            "command_argv": argv or sys.argv,
            "cloud": cloud or {},
            "started_utc": utcnow(),
            "finished_utc": None,
            "seed": protocol.seed,
            "owner": owner,
            "host": os.uname().nodename,
            "status": "running",
            "resume_count": 0,
            "resumed_utcs": [],
            "run_dir": str(ctx.run_dir.resolve()),
        }
        ctx._flush()
        return ctx

    def load_resume(self, run_dir: str | Path, protocol: Protocol) -> "RunContext":
        run_dir = Path(run_dir)
        if not (run_dir / "run.json").exists():
            raise RunError(f"{run_dir}/run.json not found; cannot resume")
        meta = read_json(run_dir / "run.json")
        if meta.get("protocol_hash") != sha256_json(protocol.to_dict()):
            raise RunError(
                f"protocol hash mismatch for {run_dir}: run={meta.get('protocol_hash')} "
                f"current={sha256_json(protocol.to_dict())}"
            )
        self.__init__(run_dir, protocol, meta.get("purpose", "full"),
                      bool(meta.get("synthetic", False)))
        self._meta = meta
        self._meta.setdefault("resumed_utcs", []).append(utcnow())
        self._meta["status"] = "running"
        self._meta["resume_count"] = int(meta.get("resume_count", 0)) + 1
        return self

    @staticmethod
    def protocol_from_dict(proto_dict: dict) -> Protocol:
        tuple_fields = (
            "input_seeds", "strategies", "modes", "pattern_blocks", "launch_k",
            "bw_sizes_bytes", "telemetry_gpu_fields",
        )
        conv = {k: tuple(proto_dict[k]) for k in tuple_fields if k in proto_dict}
        return Protocol(**{**proto_dict, **conv})

    @classmethod
    def open_existing(cls, run_dir: str | Path) -> "RunContext":
        run_dir = Path(run_dir)
        meta = read_json(run_dir / "run.json")
        proto = cls.protocol_from_dict(meta["protocol"])
        ctx = cls(run_dir, proto, meta.get("purpose", "full"),
                  bool(meta.get("synthetic", False)))
        ctx._meta = meta
        return ctx

    def reopen(self, run_dir: str | Path) -> "RunContext":
        return RunContext.open_existing(run_dir)

    def update(self, **fields) -> None:
        with self._lock:
            self._meta.update(fields)
            self._flush()

    def finish(self, status: str) -> None:
        self.update(status=status, finished_utc=utcnow())
        for w in self._writers.values():
            w.close()
        self._writers.clear()

    def _flush(self) -> None:
        atomic_write_json(self.run_json_path, self._meta)

    @property
    def meta(self) -> dict:
        return self._meta

    # ---------------- tables ----------------
    def writer(self, table: str) -> JsonlWriter:
        if table not in TABLES:
            raise RunError(f"unknown table {table}")
        if table not in self._writers:
            self._writers[table] = JsonlWriter(self.run_dir / f"{table}.jsonl")
        return self._writers[table]

    def append(self, table: str, rows: dict | Iterable[dict]) -> None:
        if isinstance(rows, dict):
            rows = [rows]
        rows = [dict(r, synthetic=self.synthetic) for r in rows]
        self.writer(table).append_many(rows)

    def read_table(self, table: str) -> list[dict]:
        self.close_writers(table)
        return read_jsonl(self.run_dir / f"{table}.jsonl")

    def close_writers(self, table: str | None = None) -> None:
        if table is None:
            for name, w in list(self._writers.items()):
                w.close()
            self._writers.clear()
        elif table in self._writers:
            self._writers.pop(table).close()

    # ---------------- logging ----------------
    def log(self, msg: str) -> None:
        line = f"{utcnow()} {msg}"
        ensure_dir(self._log_path.parent)
        with self._lock, open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        print(line, file=sys.stderr, flush=True)

    # ---------------- atomic block files ----------------
    def block_path(self, case_id: str, mode: str, block: int, attempt: int,
                   tag: str | None = None) -> Path:
        suffix = f"__{tag}" if tag else ""
        return self.run_dir / "blocks" / f"{case_id}__{mode}{suffix}__b{block:02d}__a{attempt}.json"

    def commit_block(self, case_id: str, mode: str, block: int, attempt: int,
                     rows: list[dict], pair_status: str,
                     tag: str | None = None) -> None:
        payload = {
            "case_id": case_id,
            "mode": mode,
            "block": block,
            "attempt": attempt,
            "pair_status": pair_status,
            "tag": tag,
            "committed_utc": utcnow(),
            "rows": [dict(r, synthetic=self.synthetic) for r in rows],
        }
        atomic_write_json(self.block_path(case_id, mode, block, attempt, tag), payload)
        self.append("timings", rows)

    def state_path(self, case_id: str, mode: str) -> Path:
        return self.run_dir / "state" / f"{case_id}__{mode}.json"

    def save_state(self, state: dict) -> None:
        atomic_write_json(self.state_path(state["case_id"], state["mode"]), state)

    def load_state(self, case_id: str, mode: str) -> dict | None:
        p = self.state_path(case_id, mode)
        if not p.exists():
            return None
        return read_json(p)

    def recover_timings(self) -> None:
        """Append any rows present in block files but missing from timings.jsonl."""
        timings_path = self.run_dir / "timings.jsonl"
        rows = read_jsonl(timings_path) if timings_path.exists() else []
        have = {
            (r["case_id"], r["mode"], r["block"], r["attempt"], r["strategy"])
            for r in rows
        }
        missing: list[dict] = []
        for p in sorted((self.run_dir / "blocks").glob("*.json")):
            payload = read_json(p)
            for r in payload.get("rows", []):
                key = (r["case_id"], r["mode"], r["block"], r["attempt"], r["strategy"])
                if key not in have:
                    missing.append(r)
                    have.add(key)
        if missing:
            self.append("timings", missing)
            self.log(f"recovered {len(missing)} timing rows from block files")


def default_cloud_meta(args: argparse.Namespace) -> dict:
    return {
        "platform": getattr(args, "cloud_platform", None),
        "instance_spec": getattr(args, "instance_spec", None),
        "image_id": getattr(args, "image_id", None),
        "persistent_root": getattr(args, "run_root", None),
    }
