"""Adapter: Agent A's ``fusionboundary.runtime.CandidateStore`` -> provider.

The measurement tool only relies on the documented interface in
``docs/interface-for-b.md``: ``CandidateStore.load``, ``Candidate.call`` /
``call_n`` / ``verify`` / ``capture_replay`` and the ``variants.jsonl`` row.
The archived wrapper is loaded before timing. A missing variant excludes the case.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .backend import PreparedVariant
from .util import sha256_file, sha256_json

ARTIFACT_FILES = (
    "output_code.py",
    "kernels.json",
    "compile_key.json",
    "edge_evidence.json",
    "evidence.md",
    "variant_row.json",
)


def _ensure_importable(repo_root: Path) -> None:
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _input_id_for_measurement(input_id: str) -> str:
    if input_id == "zero":
        return "zeros"
    if input_id.startswith("seed"):
        return input_id
    return f"seed{int(input_id)}"


class FusionBoundaryProvider:
    provider = "fusionboundary"

    def __init__(self, run_path: str | Path, options: dict | None = None):
        self.options = options or {}
        self.run_path = Path(run_path).resolve()
        _ensure_importable(Path(__file__).resolve().parent.parent)
        from fusionboundary.corpus import build_corpus
        from fusionboundary.runtime import CandidateStore

        self.verify_launch = bool(self.options.get("verify_launch", True))
        dev = str(self.options.get("device", "cuda"))
        # A's Candidate.capture_replay() insists on the literal "cuda".
        self.device = "cuda" if dev.startswith("cuda") else dev
        self.store = CandidateStore(str(self.run_path))
        raw_rows = []
        vpath = self.run_path / "variants.jsonl"
        if vpath.exists():
            for line in vpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw_rows.append(json.loads(line))
        self._raw: dict[tuple[str, str], dict] = {
            (r["case_id"], r["strategy"]): r for r in raw_rows
        }
        corpus = {c.case_id: c for c in build_corpus()}
        manifest_path = self.run_path / "cases.jsonl"
        if manifest_path.exists():
            cids = []
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    cids.append(json.loads(line)["case_id"])
        else:
            cids = sorted({c for (c, _s) in self._raw} | set(self.store.case_ids()))
        self._cases = {}
        for cid in cids:
            case = corpus.get(cid)
            if case is None:
                continue
            statuses = {
                s: self._raw.get((cid, s), {}).get("status") for s in ("split", "fused")
            }
            incompatibility = next(
                (flag for flag in ("unrelated_partition_change", "partitions_identical_split_fused")
                 if any(self._raw.get((cid, s), {}).get(flag) for s in ("split", "fused"))),
                None,
            )
            comparable = incompatibility is None
            available = [s for s in ("split", "fused")
                         if statuses.get(s) in ("valid", "cache_only")] if comparable else []
            row = case.as_manifest_row()
            row["logical_bytes"] = {
                "t_write": int(case.logical_save_bytes // 2),
                "t_read": int(case.logical_save_bytes // 2),
            }
            row["logical_saved_bytes"] = int(case.logical_save_bytes)
            row["input_seeds"] = list(case.seeds) + ([0] if case.zero_input else [])
            row["zero_input_id"] = "zeros"
            row["strategies"] = available
            row["planned_status"] = "planned" if len(available) == 2 else "excluded"
            if not comparable:
                row["exclusion_reason"] = incompatibility
            elif len(available) == 2:
                row["exclusion_reason"] = None
            else:
                failures = [f"{s}={statuses.get(s)}" for s in ("split", "fused") if s not in available]
                row["exclusion_reason"] = "missing_or_invalid_variant:" + ",".join(failures)
            row["source"] = {
                "candidate_run": str(self.run_path),
                "variant_status": statuses,
                "kernel_count": {
                    s: self._raw.get((cid, s), {}).get("kernel_count") for s in statuses
                },
                "output_code_sha256": {
                    s: self._raw.get((cid, s), {}).get("output_code_sha256") for s in statuses
                },
                "default_maps_to": next(
                    (self._raw.get((cid, s), {}).get("default_maps_to")
                     for s in ("split", "fused")
                     if self._raw.get((cid, s), {}).get("default_maps_to")), None
                ),
            }
            self._cases[cid] = row
        self._prepared: dict[tuple[str, str], PreparedVariant] = {}
        self._checks: dict[tuple[str, str], dict[str, dict]] = {}

    def catalog(self) -> list[dict]:
        return [self._cases[c] for c in sorted(self._cases)]

    def export_provenance(self, measurement_run: Path, run_id: str,
                          case_ids: list[str]) -> list[dict]:
        """Materialize A's variant rows and artifact links for this run.

        The measurement run keeps A's ``variants.jsonl`` rows (including failed
        variants) so downstream analysis sees the full candidate coverage, and
        copies each artifact directory under ``artifacts/candidates/artifacts``. ``run.json.candidates.source_run`` records where
        the originals live.
        """
        base = Path(measurement_run) / "artifacts" / "candidates" / "artifacts"
        rows: list[dict] = []
        for cid in case_ids:
            if cid not in self._cases:
                continue
            default_map = next(
                (self._raw.get((cid, s), {}).get("default_maps_to")
                 for s in ("split", "fused", "default")
                 if self._raw.get((cid, s), {}).get("default_maps_to")),
                None,
            )
            for strategy in ("split", "fused", "default"):
                raw = self._raw.get((cid, strategy))
                if raw is None:
                    continue
                artifact_rel = raw.get("artifact_dir")
                target = (self.run_path / artifact_rel).resolve() if artifact_rel else None
                link_rel = None
                output_rel = None
                if target is not None and target.exists():
                    link = base / cid / strategy
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if link.is_symlink():
                        link.unlink()
                    if not link.exists():
                        shutil.copytree(target, link)
                    if self._artifact_hashes(link) != self._artifact_hashes(target):
                        raise RuntimeError(f"candidate artifact changed since this run began: {cid}/{strategy}")
                    link_rel = str(link.relative_to(Path(measurement_run)))
                    if (target / "output_code.py").exists():
                        output_rel = f"{link_rel}/output_code.py"
                file_hashes = self._artifact_hashes(target) if target is not None and target.exists() else {}
                compiled_kernels = (json.loads((target / "kernels.json").read_text(encoding="utf-8"))
                                    .get("kernels", []) if "kernels.json" in file_hashes else [])
                # A single fused kernel or a two-kernel split covers the
                # entire region. Leave multi-stage occupancy unavailable.
                kernel_features = [
                    {"name": k["name"], "region": True, **(k.get("resources") or {})}
                    for k in compiled_kernels
                ] if len(compiled_kernels) == (2 if strategy == "split" else 1) else []
                row = {
                    "run_id": run_id,
                    "case_id": cid,
                    "strategy": strategy,
                    "status": raw.get("status"),
                    "artifact_path": link_rel,
                    "artifact_dir": link_rel,
                    "artifact_hash": sha256_json(file_hashes) if file_hashes else None,
                    "artifact_sha256": sha256_json(file_hashes) if file_hashes else None,
                    "artifact_files": file_hashes,
                    "output_code": output_rel,
                    "output_code_sha256": raw.get("output_code_sha256"),
                    "static_launch_count": raw.get("kernel_count"),
                    "n_kernels": raw.get("kernel_count"),
                    "kernel_count": raw.get("kernel_count"),
                    "kernel_names": raw.get("kernel_names"),
                    "kernels": kernel_features or raw.get("kernel_names"),
                    "default_maps_to": raw.get("default_maps_to") or default_map,
                    "verified_default_decision": raw.get("default_maps_to") or default_map,
                    "t_materialized": raw.get("t_materialized"),
                    "t_alloc_dtype": raw.get("t_alloc_dtype"),
                    "t_alloc_shape": raw.get("t_alloc_shape"),
                    "edge_verdict": raw.get("edge_verdict"),
                    "compile_ms": raw.get("compile_ms"),
                    "first_call_ms": raw.get("first_call_ms"),
                    "checks_total": raw.get("checks_total"),
                    "checks_failed": raw.get("checks_failed"),
                    "errors": raw.get("errors"),
                    "guard_blocks": raw.get("guard_blocks"),
                    "graph_count": raw.get("graph_count"),
                    "graph_break_count": raw.get("graph_break_count"),
                    "source": "fusionboundary",
                    "candidate_run": str(self.run_path),
                    "preallocated_inputs": True,
                    "synthetic": False,
                    "ts_utc": __import__("fb.util", fromlist=["utcnow"]).utcnow(),
                }
                rows.append(row)
        return rows

    def _artifact_hashes(self, artifact_dir: Path) -> dict[str, str]:
        out = {}
        for name in ARTIFACT_FILES:
            p = artifact_dir / name
            if p.exists():
                out[name] = sha256_file(p)
        return out

    def _observe_kernels(self, call, calls: int = 3) -> list[str]:
        import torch

        for _ in range(2):
            call()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(calls):
                call()
            torch.cuda.synchronize()
        names = []
        for evt in prof.events():
            if "CUDA" not in str(getattr(evt, "device_type", "")):
                continue
            name = getattr(evt, "name", None)
            if not name:
                continue
            low = name.lower()
            if "memcpy" in low or "memset" in low:
                continue
            names.append(name)
        return names

    def _verify_launch(self, cid: str, strategy: str, cand, expected: list[str] | None) -> list[str]:
        observed = self._observe_kernels(lambda: cand.call(cand.fixed_input()))
        if not expected or Counter(observed) != Counter(expected * 3):
            raise RuntimeError(
                f"{cid}/{strategy}: loaded candidate does not execute its recorded "
                f"kernels; expected three calls of {expected}, observed {observed}."
            )
        return sorted(set(observed))

    def _check_rows(self, cid: str, strategy: str, cand) -> dict[str, dict]:
        key = (cid, strategy)
        if key not in self._checks:
            rows = cand.verify()
            self._checks[key] = {r["input_id"]: r for r in rows if "input_id" in r}
        return self._checks[key]

    def prepare(self, case: dict, strategy: str) -> PreparedVariant:
        cid = case["case_id"]
        key = (cid, strategy)
        if key in self._prepared:
            return self._prepared[key]
        status = self.store.status(cid, strategy)
        if status is None:
            raise KeyError(
                f"no loadable {cid}/{strategy} in {self.run_path} "
                f"(status={self._raw.get(key, {}).get('status')})"
            )
        cand = self.store.load(cid, strategy, device=self.device)
        row = self._raw.get(key, {})
        expected_kernels = row.get("kernel_names") or []
        observed = (self._verify_launch(cid, strategy, cand, expected_kernels)
                    if self.verify_launch else [])
        artifact_dir = Path(cand.artifact_dir)
        hashes = self._artifact_hashes(artifact_dir)
        if not hashes:
            raise FileNotFoundError(f"no artifact files under {artifact_dir}")
        artifact_sha = sha256_json(hashes)
        if row.get("output_code_sha256") and hashes.get("output_code.py") and \
                row["output_code_sha256"] != hashes["output_code.py"]:
            raise RuntimeError(f"{cid}/{strategy}: output_code.py hash mismatch")
        kernels = []
        kpath = artifact_dir / "kernels.json"
        if kpath.exists():
            data = json.loads(kpath.read_text(encoding="utf-8"))
            for k in data.get("kernels", []):
                kernels.append({
                    "index": k.get("launch_index"),
                    "name": k.get("name"),
                    "grid": k.get("grid_dims") or k.get("grid"),
                    "num_warps": k.get("num_warps"),
                    "num_stages": k.get("num_stages"),
                    "source_sha256": k.get("source_sha256"),
                })
        n_kernels = row.get("kernel_count")
        if not isinstance(n_kernels, int) or n_kernels < 1:
            n_kernels = cand.report.kernel_count or (len(kernels) or None)
        meta = {
            "provider": "fusionboundary",
            "strategy": strategy,
            "artifact_sha256": artifact_sha,
            "artifacts": [str((artifact_dir / n)) for n in ARTIFACT_FILES
                          if (artifact_dir / n).exists()],
            "n_kernels": n_kernels,
            "preallocated_inputs": True,
            "preallocated": False,
            "compile_ms": row.get("compile_ms"),
            "load_ms": cand.report.first_call_ms,
            "cache_reused": cand.report.cache_reused,
            "submit_mode": "direct",
            "kernel_list": kernels,
            "source_variant_status": status,
            "output_code_sha256": row.get("output_code_sha256"),
            "kernel_names": row.get("kernel_names"),
            "t_materialized": row.get("t_materialized"),
            "edge_verdict": row.get("edge_verdict"),
            "default_maps_to": row.get("default_maps_to"),
            "artifact_hashes": hashes,
            "artifact_dir": str(artifact_dir),
            "candidate_run": str(self.run_path),
            "expected_kernel_names": sorted(set(expected_kernels)),
            "observed_kernel_names": observed,
            "launch_verified": bool(self.verify_launch and expected_kernels),
            "notes": "loaded from candidate store; no compile in the timing process",
        }
        default_maps_to = row.get("default_maps_to")
        if default_maps_to in ("split", "fused"):
            meta["default_equivalent"] = default_maps_to
            meta["default_partition"] = "same"
        elif default_maps_to == "distinct":
            meta["default_partition"] = "distinct"

        def make_replay() -> tuple[PreparedVariant, float]:
            t0 = time.perf_counter()
            rc = cand.capture_replay()
            capture_ms = (time.perf_counter() - t0) * 1000.0
            replay_names = self._observe_kernels(rc.call) if self.verify_launch else []
            # PyTorch's CUDA Graph replay prologue may fill the int64 RNG
            # offset. Preserve those launches as replay overhead in the record.
            offset_fills = [name for name in replay_names
                            if name.startswith("void at::native::vectorized_elementwise_kernel<")
                            and "at::native::FillFunctor<long>" in name]
            graph_kernels = Counter(replay_names) - Counter(offset_fills)
            replay_verified = bool(expected_kernels and graph_kernels == Counter(expected_kernels * 3))
            if self.verify_launch and not replay_verified:
                raise RuntimeError(f"{cid}/{strategy}: replay kernels differ from recorded kernels: {replay_names}")
            rmeta = dict(meta)
            rmeta.update(submit_mode="replay", base_strategy=strategy,
                         observed_kernel_names=sorted(set(replay_names)),
                         replay_prologue_launches=len(offset_fills),
                         launch_verified=replay_verified,
                         replay_launch_verified=replay_verified)
            rv = PreparedVariant(
                case_id=cid,
                strategy=strategy,
                meta=rmeta,
                _call=rc.call,
                _check=lambda input_id: self._replay_check(cid, strategy, rc, input_id),
                _call_n=lambda n: rc.call_n(n),
            )
            return rv, capture_ms

        prepared = PreparedVariant(
            case_id=cid,
            strategy=strategy,
            meta=meta,
            _call=lambda: cand.call(cand.fixed_input()),
            _check=lambda input_id: self._check_one(cid, strategy, cand, input_id),
            _call_n=lambda n: cand.call_n(n),
            _replay=make_replay,
        )
        self._prepared[key] = prepared
        return prepared

    def _check_one(self, cid: str, strategy: str, cand, input_id: str) -> dict:
        rows = self._check_rows(cid, strategy, cand)
        mapped = _input_id_for_measurement(input_id)
        if mapped not in rows:
            return {"pass": None, "reason": f"no check row for {mapped}"}
        r = dict(rows[mapped])
        r.pop("case_id", None)
        r.pop("strategy", None)
        r.pop("input_id", None)
        return r

    def _replay_check(self, cid: str, strategy: str, rc, input_id: str) -> dict:
        rows = rc.verify()
        return dict(rows[0]) if rows else {"pass": None, "reason": "no replay check"}

    def release_case(self, case_id: str) -> None:
        for mapping in (self._prepared, self._checks):
            for key in [key for key in mapping if key[0] == case_id]:
                del mapping[key]
        self.store.release_case(case_id)

    def close(self) -> None:
        for case_id in {key[0] for key in self._prepared}:
            self.release_case(case_id)
