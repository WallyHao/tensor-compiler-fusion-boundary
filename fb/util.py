"""Small utilities: UTC time, hashing, atomic files, JSONL."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable


def utcnow() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def parse_utc(s: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: malformed JSONL row: {exc}") from exc
    return rows


class JsonlWriter:
    """Append-only JSONL with fsync; safe to call from one thread at a time."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, row: dict, fsync: bool = True) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            if fsync:
                os.fsync(self._fh.fileno())

    def append_many(self, rows: Iterable[dict], fsync: bool = True) -> None:
        with self._lock:
            for row in rows:
                self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()
            if fsync:
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_jsonl_atomic(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(path, text)


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return ys[mid] if n % 2 else 0.5 * (ys[mid - 1] + ys[mid])


def percentiles(xs: list[float], ps: tuple[float, ...] = (25, 75)) -> dict[str, float]:
    if not xs:
        return {f"p{int(p)}": None for p in ps}
    ys = sorted(xs)
    out: dict[str, float] = {}
    for p in ps:
        if len(ys) == 1:
            out[f"p{int(p)}"] = ys[0]
            continue
        rank = (len(ys) - 1) * p / 100.0
        lo = int(rank)
        hi = min(lo + 1, len(ys) - 1)
        frac = rank - lo
        out[f"p{int(p)}"] = ys[lo] * (1 - frac) + ys[hi] * frac
    return out


def linear_fit(xs: list[float], ys: list[float]) -> dict:
    """Least-squares y = slope*x + intercept with residuals."""
    n = len(xs)
    if n < 2:
        raise ValueError("linear_fit needs at least two points")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("degenerate x in linear_fit")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in residuals)
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "slope": slope,
        "intercept": intercept,
        "residual_std": (ss_res / max(1, n - 2)) ** 0.5,
        "residuals": residuals,
        "r2": r2,
        "n_points": n,
    }


def git_state(root: str | os.PathLike) -> dict:
    import subprocess

    root = str(root)

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, cwd=root, capture_output=True, text=True, check=False
            ).stdout.strip()
        except OSError:
            return None

    rev = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    diff = run("git", "diff")
    return {
        "commit": rev or None,
        "dirty": bool(status),
        "status_porcelain": status,
        "diff_sha256": sha256_text(diff) if diff else None,
        "diff_text": diff,
    }
