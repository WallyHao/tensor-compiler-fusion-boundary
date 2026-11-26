"""Deterministic file I/O and audit helpers.

No third-party dependencies: the frozen package must rebuild on a bare Python
interpreter.  Every row read from JSONL keeps its source file/line so that each
derived number can be traced back to the raw record.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def issue(
    severity: str,
    code: str,
    message: str,
    *,
    case_id: str | None = None,
    mode: str | None = None,
    block: int | None = None,
    attempt: int | None = None,
    strategy: str | None = None,
    file: str | None = None,
    line: int | None = None,
    detail: Any = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
        "case_id": case_id,
        "mode": mode,
        "block": block,
        "attempt": attempt,
        "strategy": strategy,
        "file": file,
        "line": line,
    }
    if detail is not None:
        row["detail"] = detail
    return row


def load_json(path: Path, issues: list[dict[str, Any]], *, required: bool = True) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        if required:
            issues.append(issue("error", "missing_file", f"required file is missing: {path.name}", file=path.name))
        return {}
    except json.JSONDecodeError as exc:
        issues.append(
            issue("error", "malformed_json", f"{path.name}: {exc}", file=path.name, line=exc.lineno)
        )
        return {}
    if not isinstance(data, dict):
        issues.append(issue("error", "malformed_json", f"{path.name}: top level is not an object", file=path.name))
        return {}
    data["_src"] = {"file": path.name, "line": 1}
    return data


def load_jsonl(path: Path, issues: list[dict[str, Any]], *, required: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    issues.append(
                        issue(
                            "error",
                            "malformed_json",
                            f"{path.name}:{lineno}: {exc.msg}",
                            file=path.name,
                            line=lineno,
                        )
                    )
                    continue
                if not isinstance(row, dict):
                    issues.append(
                        issue(
                            "error",
                            "malformed_json",
                            f"{path.name}:{lineno}: row is not an object",
                            file=path.name,
                            line=lineno,
                        )
                    )
                    continue
                row["_src"] = {"file": path.name, "line": lineno}
                rows.append(row)
    except FileNotFoundError:
        if required:
            issues.append(issue("error", "missing_file", f"required file is missing: {path.name}", file=path.name))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def normalize_hash(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text or None


def hash_matches(path: Path, expected: Any) -> tuple[bool, str | None]:
    actual = sha256_file(path)
    return actual is not None and actual == normalize_hash(expected), actual


def source_of(row: dict[str, Any]) -> tuple[str | None, int | None]:
    src = row.get("_src") or {}
    return src.get("file"), src.get("line")


def strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def file_hashes(root: Path, names: Iterable[str]) -> dict[str, str | None]:
    return {name: sha256_file(root / name) for name in names}
