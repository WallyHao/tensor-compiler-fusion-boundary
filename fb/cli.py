"""Command line entry: fb env|corpus|calibrate|collect|pilot|verify|profile|seal|synth-export."""
from __future__ import annotations

import argparse
import json
import contextlib
import fcntl
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from .corpus import (
    PILOT_CASES,
    build_catalog,
    catalog_hash,
    template_counts,
    validate_catalog,
)
from .protocol import Protocol
from .rundir import RunContext
from .util import (
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    git_state,
    read_json,
    sha256_file,
    sha256_json,
    utcnow,
    write_jsonl_atomic,
)
from .verify import print_verification, verify_run


# ------------------------------------------------------------------ helpers
def build_protocol(args) -> Protocol:
    kw = {}
    if getattr(args, "blocks", None):
        b = int(args.blocks)
        kw["blocks"] = b
        kw["pattern_blocks"] = ((b + 1) // 2, b // 2)
    if getattr(args, "seed", None) is not None:
        kw["seed"] = int(args.seed)
    if getattr(args, "modes", None):
        kw["modes"] = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if getattr(args, "telemetry_interval", None):
        kw["telemetry_interval_s"] = float(args.telemetry_interval)
    p = replace(Protocol(), **kw)
    p.validate()
    return p


def _run_id(purpose: str) -> str:
    stamp = utcnow().replace(":", "").replace("-", "")[:15]
    return f"{purpose}-{stamp}Z"


@contextlib.contextmanager
def gpu_lock(path: Path, timeout_s: float, holder: str):
    ensure_dir(path.parent)
    fh = open(path, "a+")
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if timeout_s == 0 or time.time() - start > timeout_s:
                    raise RuntimeError(
                        f"GPU lock {path} is held by another process; "
                        f"waited {time.time() - start:.0f}s. Do not overlap compile/timing/profiling."
                    )
                time.sleep(2)
        fh.seek(0)
        fh.truncate()
        fh.write(f"holder={holder} pid={os.getpid()} since={utcnow()}\n")
        fh.flush()
        os.fsync(fh.fileno())
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def configure_torch(torch, device: str) -> dict:
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    torch.use_deterministic_algorithms(False)
    try:
        import torch._inductor.config as ind

        ind.triton.cudagraphs = False
        ind.max_autotune = False
        ind.coordinate_descent_tuning = False
    except Exception:
        pass
    return {
        "grad_enabled": torch.is_grad_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "inductor_cudagraphs": False,
        "device": str(torch.cuda.current_device()),
    }


def _persistent_check(run_root: Path) -> dict:
    resolved = run_root.resolve()
    out = {"run_root": str(resolved)}
    try:
        p = subprocess.run(
            ["findmnt", "-T", str(resolved), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS", "-n"],
            capture_output=True, text=True, timeout=10,
        )
        out["findmnt"] = p.stdout.strip()
        fields = p.stdout.strip().split()
        out["fstype"] = fields[2] if len(fields) > 2 else None
    except Exception as exc:
        out["error"] = str(exc)
        out["fstype"] = None
    out["ephemeral"] = bool(
        str(resolved).startswith(("/tmp/", "/run/"))
        or out.get("fstype") in ("tmpfs", "ramfs")
    )
    return out


def _env_or_capture(ctx: RunContext, device_index: int, recapture: bool,
                    run_root: Path) -> dict:
    from .env import collect_environment, save_environment, verify_device

    env_path = ctx.run_dir / "environment" / "environment.json"
    previous = None
    if env_path.exists():
        previous = read_json(env_path)
    env = collect_environment(device_index=device_index,
                              repo_root=str(Path(__file__).resolve().parent.parent))
    if previous is not None:
        original_uuid = (previous.get("gpu") or {}).get("uuid")
        current_uuid = (env.get("gpu") or {}).get("uuid")
        if original_uuid and original_uuid != current_uuid:
            raise ValueError("GPU UUID changed; create a new measurement run")
        initial = ctx.run_dir / "environment" / "environment_initial.json"
        if not initial.exists():
            atomic_write_json(initial, previous)
    save_environment(ctx, env)
    ctx.update(persistent_storage=_persistent_check(run_root))
    return env


def _check_device(env: dict, expect: str | None, expect_uuid: str | None,
                  allow_mismatch: bool, ctx: RunContext) -> tuple[dict, bool]:
    from .env import verify_device

    check = verify_device(env, expect, expect_uuid)
    ctx.update(device_check=check)
    ok = check["pass"] or allow_mismatch
    if not check["pass"]:
        print(
            "DEVICE CHECK FAILED:\n  " + "\n  ".join(check["problems"]),
            file=sys.stderr,
        )
        if allow_mismatch:
            print("  --allow-device-mismatch set: this run is for diagnostics only.",
                  file=sys.stderr)
    return check, ok


def _load_provider(spec: str, options: dict | None):
    from .provider import load_provider

    return load_provider(spec, options or {})


def _select_cases(spec: str | None, catalog: list[dict]) -> list[str]:
    from .runner import normalize_case_filter

    return normalize_case_filter(spec, catalog)


def _write_candidates(ctx: RunContext, provider_spec: str, formal: bool,
                      contract_ok: bool, note: str | None = None,
                      extra: dict | None = None) -> None:
    payload = {
        "provider": provider_spec,
        "formal": formal,
        "contract_ok": contract_ok,
        "note": note,
        "recorded_utc": utcnow(),
    }
    payload.update(extra or {})
    ctx.update(candidates=payload)


def _copy_candidate_provenance(ctx: RunContext, provider) -> dict:
    """Keep A's manifest and variant table with the measurement run."""
    import shutil

    run_path = getattr(provider, "run_path", None)
    if run_path is None:
        return {}
    src = Path(run_path)
    dst = ensure_dir(ctx.run_dir / "artifacts" / "candidates")
    copied = {}
    for name in ("run.json", "cases.jsonl", "variants.jsonl",
                 "environment.lock.json", "environment.json", "gpu.json"):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
            copied[name] = sha256_file(dst / name)
    return {"source_run": str(src), "source_files_sha256": copied}


# ------------------------------------------------------------------ commands
def cmd_env(args) -> int:
    proto = build_protocol(args)
    root = Path(args.run_root)
    rid = args.run_id or _run_id("environment")
    code = git_state(str(Path(__file__).resolve().parent.parent))
    ctx = RunContext.create(root, rid, proto, "environment", argv=sys.argv,
                            corpus_hash=None, cloud=_cloud(args),
                            owner=os.environ.get("FB_OWNER", "measurement-b"))
    ctx.update(code={k: code.get(k) for k in ("commit", "dirty", "diff_sha256")},
               code_hash=code.get("commit") or sha256_json(code.get("diff_sha256")))
    with gpu_lock(Path(args.gpu_lock or root / "gpu.lock"), args.lock_timeout,
                  holder=f"env {rid}"):
        try:
            import torch

            configure_torch(torch, args.device)
        except Exception as exc:
            ctx.log(f"torch config skipped: {exc}")
        env = _env_or_capture(ctx, args.device_index, True, root)
        check, ok = _check_device(env, args.expect_device, args.expect_uuid,
                                  args.allow_device_mismatch, ctx)
        ctx.finish("done" if ok else "aborted_device_mismatch")
        seal_run(ctx.run_dir, quiet=True)
    print(json.dumps({
        "run_dir": str(ctx.run_dir),
        "gpu": env.get("gpu", {}).get("name"),
        "uuid": env.get("gpu", {}).get("uuid"),
        "device_check": check,
        "environment_lock": str(ctx.run_dir / "environment" / "environment.lock.json"),
    }, indent=2, ensure_ascii=False))
    return 0 if ok else 2


def cmd_corpus(args) -> int:
    cases = build_catalog()
    problems = validate_catalog(cases)
    out = Path(args.out)
    ensure_dir(out)
    write_jsonl_atomic(out / "cases.jsonl", cases)
    atomic_write_json(out / "catalog.json", {
        "cases": cases, "catalog_hash": catalog_hash(cases),
        "template_counts": template_counts(cases),
        "generated_utc": utcnow(), "problems": problems,
        "pilot_cases": list(PILOT_CASES),
    })
    print(f"{len(cases)} cases -> {out}/catalog.json; problems={problems}")
    return 0 if not problems else 3


def cmd_calibrate(args) -> int:
    from .backend import CudaBackend
    from .calibration import run_calibration

    proto = build_protocol(args)
    root = Path(args.run_root)
    rid = args.run_id or _run_id("calibration")
    code = git_state(str(Path(__file__).resolve().parent.parent))
    ctx = RunContext.create(root, rid, proto, "calibration", argv=sys.argv,
                            cloud=_cloud(args),
                            owner=os.environ.get("FB_OWNER", "measurement-b"))
    ctx.update(code={k: code.get(k) for k in ("commit", "dirty", "diff_sha256")},
               code_hash=code.get("commit") or sha256_json(code.get("diff_sha256")))
    with gpu_lock(Path(args.gpu_lock or root / "gpu.lock"), args.lock_timeout,
                  holder=f"calibrate {rid}"):
        backend = CudaBackend(args.device)
        configure_torch(backend.torch, args.device)
        env = _env_or_capture(ctx, args.device_index, False, root)
        check, ok = _check_device(env, args.expect_device, args.expect_uuid,
                                  args.allow_device_mismatch, ctx)
        if not ok:
            ctx.finish("aborted_device_mismatch")
            return 2
        backend = CudaBackend(args.device)
        configure_torch(backend.torch, args.device)
        ctx.update(candidate_free=True)
        result = run_calibration(ctx, backend, proto)
        ctx.finish("done")
        seal_run(ctx.run_dir, quiet=True)
    print(json.dumps({
        "run_dir": str(ctx.run_dir),
        "launch_direct_us_per_kernel": result["launch_direct"].get("slope"),
        "launch_replay_us_per_kernel": result["launch_replay"].get("slope"),
        "bw_main_gbs": result["bandwidth"].get("bw_main_gbs"),
        "sizes_used": result["bandwidth"].get("sizes_used_bytes"),
        "warnings": [w for w in [
            None if result["launch_direct"].get("valid") else "launch_direct invalid",
            None if result["bandwidth"].get("valid") else "bandwidth invalid",
        ] if w],
    }, indent=2, ensure_ascii=False))
    return 0 if result["launch_direct"].get("valid") and result["bandwidth"].get("valid") else 5


def _collect(args, purpose: str, default_cases: str, default_blocks: int | None) -> int:
    from .backend import CudaBackend
    from .calibration import run_calibration
    from .runner import Runner
    from .telemetry import TelemetrySampler

    proto = build_protocol(args)
    root = Path(args.run_root)
    rid = args.run_id or _run_id(purpose)
    formal = purpose in ("full", "pilot")
    code = git_state(str(Path(__file__).resolve().parent.parent))
    ctx = RunContext.create(root, rid, proto, purpose, argv=sys.argv,
                            corpus_hash=None, cloud=_cloud(args),
                            owner=os.environ.get("FB_OWNER", "measurement-b"))
    ctx.update(code={k: code.get(k) for k in ("commit", "dirty", "diff_sha256")},
               code_hash=code.get("commit") or sha256_json(code.get("diff_sha256")))
    env_path = ctx.run_dir / "environment" / "environment.json"
    exit_code = 0
    try:
        with gpu_lock(Path(args.gpu_lock or root / "gpu.lock"), args.lock_timeout,
                      holder=f"{purpose} {rid}"):
            backend = CudaBackend(args.device)
            torch = backend.torch
            with torch.inference_mode():
                cfg = configure_torch(torch, args.device)
            ctx.update(torch_runtime_config=cfg)
            env = _env_or_capture(ctx, args.device_index, args.recapture_env, root)
            expected = args.expect_device
            if expected is None and formal:
                expected = os.environ.get("FB_EXPECT_DEVICE", "RTX 3090 Ti")
            check, ok = _check_device(env, expected, args.expect_uuid,
                                      args.allow_device_mismatch, ctx)
            if not ok:
                ctx.finish("aborted_device_mismatch")
                return 2
            from .telemetry import list_external_gpu_processes, our_process_ids

            if formal and (args.provider == "reference" or args.allow_unverified_candidates):
                ctx.log("formal runs require verified candidates")
                ctx.finish("aborted_invalid_formal_flags")
                return 9
            if formal and getattr(args, "allow_external_gpu", False):
                ctx.log("--allow-external-gpu is forbidden for formal runs")
                ctx.finish("aborted_invalid_formal_flags")
                return 9
            if formal and getattr(args, "no_telemetry", False):
                ctx.log("formal runs require telemetry; use --purpose dryrun for local tests")
                ctx.finish("aborted_invalid_formal_flags")
                return 9
            ctx.update(telemetry_enabled=not getattr(args, "no_telemetry", False))
            external = list_external_gpu_processes(our_process_ids())
            ctx.update(external_gpu_processes_at_start=external,
                       allow_external_gpu=bool(getattr(args, "allow_external_gpu", False)))
            if external and not getattr(args, "allow_external_gpu", False):
                ctx.log("external GPU process(es) present at start; refusing to overlap "
                        f"timing with them: {external}")
                ctx.finish("aborted_external_gpu")
                print("ERROR: stop the other CUDA job or pass --allow-external-gpu "
                      "for a non-formal dry run", file=sys.stderr)
                return 8
            persistent = ctx.meta.get("persistent_storage", {})
            if persistent.get("ephemeral") and not args.allow_ephemeral_root:
                ctx.log(
                    f"run root {root} looks ephemeral ({persistent}); "
                    "use a persistent path or pass --allow-ephemeral-root"
                )
                ctx.finish("aborted_ephemeral_root")
                return 6
            with torch.inference_mode():
                provider = _load_provider(args.provider, {"device": args.device})
                catalog = provider.catalog()
                problems = validate_catalog(catalog)
                if problems:
                    ctx.log(f"catalog problems: {problems[:10]} (n={len(problems)})")
                ctx.update(catalog_check={
                    "problems": problems,
                    "template_counts": template_counts(catalog),
                    "catalog_hash": catalog_hash(catalog),
                    "n_cases": len(catalog),
                }, corpus_hash=catalog_hash(catalog))
                specs = args.cases or default_cases
                ids = _select_cases(specs, catalog)
                source_info = _copy_candidate_provenance(ctx, provider)
                if hasattr(provider, "export_provenance"):
                    exported_rows = provider.export_provenance(ctx.run_dir, rid, ids)
                    prior = {(r["case_id"], r["strategy"]): r.get("artifact_hash")
                             for r in ctx.read_table("variants")}
                    for row in exported_rows:
                        key = (row["case_id"], row["strategy"])
                        if key in prior and prior[key] != row.get("artifact_hash"):
                            raise ValueError(f"candidate changed on resume: {key}")
                    new_rows = [r for r in exported_rows
                                if (r["case_id"], r["strategy"]) not in prior]
                    if new_rows:
                        ctx.append("variants", new_rows)
                selected_ids = set(ids)
                selected_catalog = [c for c in catalog if c["case_id"] in selected_ids]
                ctx.update(
                    expected_blocks=proto.blocks,
                    expected_case_count=len(selected_catalog),
                    planned_template_counts=template_counts(selected_catalog),
                    expected_modes=list(proto.modes),
                )
                coverage = {
                    "cases_in_catalog": len(catalog),
                    "cases_selected": len(ids),
                    "pairable_cases": sum(
                        1 for c in selected_catalog
                        if set(c.get("strategies") or ["split", "fused"]) >= {"split", "fused"}
                        and c.get("planned_status") == "planned"
                    ),
                    "excluded_cases": sum(
                        1 for c in selected_catalog
                        if c.get("planned_status") == "excluded"
                    ),
                }
                _write_candidates(
                    ctx, args.provider, formal,
                    contract_ok=(args.provider != "reference" and coverage["pairable_cases"] > 0),
                    note=None if args.provider != "reference" else
                    "reference provider is unverified development-only candidate code",
                    extra={"coverage": coverage, **source_info},
                )
                ctx.log(f"run {rid}: purpose={purpose} cases={len(ids)} blocks={proto.blocks} "
                        f"modes={proto.modes} provider={args.provider}")
                if args.calibrate:
                    fit_path = ctx.run_dir / "calibration_fit.json"
                    if fit_path.exists() and not args.recalibrate:
                        ctx.log("calibration already present; skipping (use --recalibrate)")
                    else:
                        run_calibration(ctx, backend, proto)
                sampler = None
                if not args.no_telemetry:
                    sampler = TelemetrySampler(ctx, proto.telemetry_interval_s,
                                               args.device_index).start()
                    if not sampler.available:
                        ctx.log(f"telemetry unavailable: {sampler.error}")
                try:
                    runner = Runner(
                        ctx, provider, backend, proto, telemetry=sampler,
                        formal=formal,
                        allow_unverified=args.allow_unverified_candidates,
                        stop_after_blocks=args.stop_after_blocks,
                    )
                    summary = runner.run(ids)
                finally:
                    if sampler is not None:
                        sampler.stop()
                status = "paused" if summary.get("paused") else "done"
                ctx.update(collection_summary={
                    "purpose": purpose,
                    "cases": summary["cases"],
                    "executed_blocks": summary["executed_blocks"],
                    "started_utc": summary["started_utc"],
                    "finished_utc": summary["finished_utc"],
                })
                ctx.finish(status)
    except KeyboardInterrupt:
        ctx.finish("interrupted")
        raise
    finally:
        if ctx.meta.get("status") not in ("paused", "done", "aborted_device_mismatch",
                                          "aborted_ephemeral_root", "aborted_external_gpu",
                                          "aborted_invalid_formal_flags", "interrupted"):
            ctx.finish("error")
    # Internal pass ignores checksums from a previous stage; they are about to
    # be regenerated for the files appended in this stage.
    result = verify_run(ctx.run_dir, write=True, check_sums=False)
    if ctx.meta.get("status") not in ("paused",) and not result["complete"]:
        ctx.update(status="done_incomplete")
    seal_run(ctx.run_dir)
    result = verify_run(ctx.run_dir, write=False)
    print_verification(result)
    if not result["integrity_pass"]:
        exit_code = 3
    elif purpose in ("full", "pilot") and result["totals"]["valid_pairs"] == 0:
        print("ERROR: no valid split/fused pair was collected; refusing to treat this as data.",
              file=sys.stderr)
        exit_code = 7
    if purpose == "pilot" and ctx.meta.get("status") != "paused" and not result["complete"]:
        print("ERROR: the pilot did not complete both cases and modes.", file=sys.stderr)
        exit_code = exit_code or 7
    if result["complete"] is False and exit_code == 0:
        print("NOTE: run is incomplete (see verification.md); raw failures were kept.",
              file=sys.stderr)
    print(f"run_dir={ctx.run_dir}")
    print(f"timings={ctx.run_dir / 'timings.jsonl'}")
    print(f"verification={ctx.run_dir / 'verification.md'}")
    return exit_code


def cmd_collect(args) -> int:
    return _collect(args, args.purpose, args.cases or "", args.blocks)


def cmd_pilot(args) -> int:
    if not args.cases:
        args.cases = ",".join(PILOT_CASES)
    if not args.blocks:
        args.blocks = 5
    return _collect(args, "pilot", args.cases, args.blocks)


def cmd_verify(args) -> int:
    r = verify_run(args.run_dir, write=args.write)
    print_verification(r)
    if args.write:
        print(f"wrote {Path(args.run_dir) / 'verification.json'} (re-seal afterwards)")
    return 0 if r["integrity_pass"] else 3


def cmd_profile(args) -> int:
    from .profile import run_profile

    root = Path(args.run_root) if args.run_root else Path(args.run_dir).parent
    with gpu_lock(Path(args.gpu_lock or root / "gpu.lock"), args.lock_timeout,
                  holder=f"profile {args.run_dir}"):
        out = run_profile(args.run_dir, args.cases, args.iters, args.device)
    print(json.dumps(out, indent=2))
    return 0


def cmd_seal(args) -> int:
    p = seal_run(Path(args.run_dir), quiet=False)
    print(f"sealed {p}")
    return 0


def cmd_synth(args) -> int:
    from .synth import export_synthetic

    out = export_synthetic(args.out, force=args.force)
    print(f"synthetic run: {out}")
    return 0


# ------------------------------------------------------------------ sealing
DERIVED_FILES = {
    "SHA256SUMS",
    "verification.json",
    "verification.md",
    "freeze/manifest.json",
}


def seal_run(run_dir: Path, quiet: bool = True) -> Path:
    run_dir = Path(run_dir)
    files: list[Path] = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir)
        if rel.as_posix() in DERIVED_FILES:
            continue
        files.append(p)
    entries = [f"{sha256_file(p)}  {p.relative_to(run_dir).as_posix()}" for p in files]
    atomic_write_text(run_dir / "SHA256SUMS", "\n".join(entries) + "\n")
    meta = read_json(run_dir / "run.json")
    calib = run_dir / "calibration_fit.json"
    manifest = {
        "run_id": meta.get("run_id"),
        "purpose": meta.get("purpose"),
        "synthetic": meta.get("synthetic"),
        "protocol_hash": meta.get("protocol_hash"),
        "corpus_hash": meta.get("corpus_hash"),
        "catalog_hash": (meta.get("catalog_check") or {}).get("catalog_hash"),
        "environment_lock_sha256": sha256_file(run_dir / "environment" / "environment.lock.json")
        if (run_dir / "environment" / "environment.lock.json").exists() else None,
        "calibration_fit_sha256": sha256_file(calib) if calib.exists() else None,
        "calibration_summary": meta.get("calibration"),
        "device_check": meta.get("device_check"),
        "gpu": (meta.get("environment") or {}).get("gpu"),
        "code": meta.get("code"),
        "sealed_utc": utcnow(),
        "file_count": len(files),
        "sha256sums_path": "SHA256SUMS",
        "seal_sha256": sha256_json(entries),
    }
    atomic_write_json(run_dir / "freeze" / "manifest.json", manifest)
    return run_dir


def _cloud(args) -> dict:
    return {
        "platform": args.cloud_platform,
        "instance_spec": args.instance_spec,
        "image_id": args.image_id,
        "persistent_root": getattr(args, "run_root", None),
        "note": "rental and release are handled by the owner; this tool only records metadata",
    }


# ------------------------------------------------------------------ parser
def _add_common(p, *, with_cases: bool = False, provider: bool = False):
    p.add_argument("--run-root", default=os.environ.get("FB_RUN_ROOT", "runs"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--modes", default=None)
    p.add_argument("--telemetry-interval", type=float, default=None)
    p.add_argument("--gpu-lock", default=os.environ.get("FB_GPU_LOCK"))
    p.add_argument("--lock-timeout", type=float, default=3600.0)
    p.add_argument("--expect-device", default=None)
    p.add_argument("--expect-uuid", default=None)
    p.add_argument("--allow-device-mismatch", action="store_true")
    p.add_argument("--allow-ephemeral-root", action="store_true")
    p.add_argument("--recapture-env", action="store_true")
    p.add_argument("--cloud-platform", default=os.environ.get("FB_CLOUD_PLATFORM"))
    p.add_argument("--instance-spec", default=os.environ.get("FB_INSTANCE_SPEC"))
    p.add_argument("--image-id", default=os.environ.get("FB_IMAGE_ID"))
    if with_cases:
        p.add_argument("--cases", default=None,
                       help="case ids or selectors: P1,P2,R1,R2,PILOT,P,R")
    if provider:
        p.add_argument("--provider", default=os.environ.get("FB_PROVIDER", "reference"),
                       help="'reference' or a directory containing provider.py")
        p.add_argument("--allow-unverified-candidates", action="store_true")
        p.add_argument("--calibrate", action="store_true")
        p.add_argument("--recalibrate", action="store_true")
        p.add_argument("--no-telemetry", action="store_true")
        p.add_argument("--stop-after-blocks", type=int, default=None)
        p.add_argument("--allow-external-gpu", action="store_true",
                       help="non-formal dry run only: permit another CUDA process")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fb", description="Fusion Boundaries measurement tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("env", help="capture environment and verify the device")
    _add_common(p)
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("corpus", help="emit the 72-case reference catalog")
    p.add_argument("--out", default="corpus")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("calibrate", help="launch-cost and bandwidth calibration")
    _add_common(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("collect", help="paired collection (full corpus or subset)")
    _add_common(p, with_cases=True, provider=True)
    p.add_argument("--purpose", default="full", choices=["full", "pilot", "dryrun", "synthetic"])
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("pilot", help="P1+R1 pilot on P1/N=65536/f32 and R1/M=256/K=512/f32")
    _add_common(p, with_cases=True, provider=True)
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("verify", help="re-check a run directory (read-only by default)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--write", action="store_true",
                   help="rewrite verification.json/md (invalidates SHA256SUMS until re-seal)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("profile", help="separate-process per-kernel profiling")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-root", default=None)
    p.add_argument("--gpu-lock", default=os.environ.get("FB_GPU_LOCK"))
    p.add_argument("--lock-timeout", type=float, default=3600.0)
    p.add_argument("--cases", required=True)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("seal", help="write SHA256SUMS and freeze manifest")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_seal)

    p = sub.add_parser("synth-export", help="synthetic paired sample for the analysis owner")
    p.add_argument("--out", default="runs/synthetic-c-sample")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_synth)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 4
