"""Command-line entrypoint: build, fixtures, selftest, trace, build-all."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .fixtures import RUN_ID, write_fixture_run
from .pipeline import build_all, build_run, load_and_trace
from .selftest import run_selftest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fusionaudit", description="Fusion-boundary data audit and analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="rebuild every table, figure and report from one raw run directory")
    build.add_argument("--run", required=True, type=Path, help="raw run directory containing run.json etc.")
    build.add_argument("--out", required=True, type=Path, help="output directory (separate from raw data)")
    build.add_argument("--expected-blocks", type=int, default=None, help="override run.expected_blocks")
    build.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_ITERATIONS)
    build.add_argument("--seed", type=int, default=config.BOOTSTRAP_SEED)
    build.add_argument("--allow-partial", action="store_true", help="include cases with fewer than expected blocks (not for main results)")
    build.add_argument("--strict", action="store_true", help="exit 2 if the audit reports errors")

    build_all_parser = sub.add_parser("build-all", help="build every run directory under a root")
    build_all_parser.add_argument("--runs", required=True, type=Path)
    build_all_parser.add_argument("--out", required=True, type=Path)
    build_all_parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_ITERATIONS)
    build_all_parser.add_argument("--seed", type=int, default=config.BOOTSTRAP_SEED)

    fixtures = sub.add_parser("fixtures", help="write the marked synthetic fixture run")
    fixtures.add_argument("--out", type=Path, default=Path("runs"))
    fixtures.add_argument("--force", action="store_true")

    selftest = sub.add_parser("selftest", help="run the four-scenario fixture self-test")
    selftest.add_argument("--workdir", type=Path, default=None)
    selftest.add_argument("--keep", action="store_true")

    trace = sub.add_parser("trace", help="print one case/mode derivation down to timing lines and artifacts")
    trace.add_argument("--run", required=True, type=Path)
    trace.add_argument("--case-id", required=True)
    trace.add_argument("--mode", default="direct", choices=["direct", "replay"])

    args = parser.parse_args(argv)
    if args.command == "build":
        summary = build_run(
            args.run,
            args.out,
            expected_blocks=args.expected_blocks,
            iterations=args.bootstrap,
            seed=args.seed,
            allow_partial=args.allow_partial,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.strict and summary["errors"]:
            return 2
        return 0
    if args.command == "build-all":
        summaries = build_all(args.runs, args.out, iterations=args.bootstrap, seed=args.seed)
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0
    if args.command == "fixtures":
        path = write_fixture_run(args.out, force=args.force)
        print(path)
        print(f"run_id={RUN_ID} synthetic=true")
        return 0
    if args.command == "selftest":
        return run_selftest(workdir=args.workdir, keep=args.keep)
    if args.command == "trace":
        payload = load_and_trace(args.run, args.case_id, args.mode)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if "error" not in payload else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
