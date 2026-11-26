"""End-to-end GPU test: compile small cases through the CLI, then verify.

Marked ``gpu``/``slow``: it spawns real Inductor compilations in subprocesses
(the only way to set TORCHINDUCTOR_CACHE_DIR before torch is imported) and
needs a CUDA device. Run with:

    .venv/bin/python -m pytest tests/test_gpu_pilot.py -m gpu -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = ["P1-n1024-f32-contig-t", "R1-m16-k128-f32-contig-t"]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((os.path.join(ROOT, "src"), ROOT))
    return subprocess.run([sys.executable, "-m", "fusionboundary", *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("gpu") / "runs" / "gpu-pilot")
    args = ["compile", "--run", path, "--strategy", "default,split,fused"]
    for case in CASES:
        args += ["--case", case]
    proc = _run(args)
    print(proc.stdout[-4000:])
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return path


def _rows(run_dir: str) -> dict[tuple[str, str], dict]:
    path = os.path.join(run_dir, "variants.jsonl")
    with open(path, encoding="utf-8") as fh:
        return {(r["case_id"], r["strategy"]): r for r in map(json.loads, fh)}


def test_all_variants_valid(run_dir):
    rows = _rows(run_dir)
    for case in CASES:
        for strategy in ("default", "split", "fused"):
            row = rows[(case, strategy)]
            assert row["status"] == "valid", row["errors"]
            assert row["checks_failed"] == 0
            assert row["graph_break_count"] == 0
            assert row["graph_count"] == 1


def test_split_materialises_t(run_dir):
    rows = _rows(run_dir)
    for case in CASES:
        split = rows[(case, "split")]
        assert split["t_materialized"] is True
        assert split["t_alloc_dtype"] == "float32"
        assert split["kernel_count"] >= 2
        assert split["guard_blocks"] >= 1


def test_fused_has_no_t_buffer(run_dir):
    rows = _rows(run_dir)
    for case in CASES:
        fused = rows[(case, "fused")]
        assert fused["t_materialized"] is False
        assert fused["edge_verdict"] == "fused"


def test_default_maps_to_fused(run_dir):
    rows = _rows(run_dir)
    for case in CASES:
        assert rows[(case, "default")]["default_maps_to"] == "fused"


def test_no_unrelated_partition_change(run_dir):
    rows = _rows(run_dir)
    for case in CASES:
        assert rows[(case, "split")]["unrelated_partition_change"] is False


def test_evidence_files_exist(run_dir):
    for case in CASES:
        d = os.path.join(run_dir, "artifacts", case, "split")
        for name in ("output_code.py", "edge_evidence.json", "evidence.md", "kernels.json"):
            assert os.path.getsize(os.path.join(d, name)) > 0, name
        with open(os.path.join(d, "edge_evidence.json"), encoding="utf-8") as fh:
            ev = json.load(fh)
        assert ev["matched_t_flows"], ev["verdict"]
        flow = ev["matched_t_flows"][0]
        assert flow["t_write_site"] and flow["t_read_site"]


def test_verify_reloads_through_interface(run_dir):
    proc = _run(["verify", "--run", run_dir, "--strategy", "split,fused",
                 "--case", CASES[0]])
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "kernels=2" in proc.stdout


def test_both_strategies_launch_archived_kernels(run_dir):
    from fb.fusionboundary_provider import FusionBoundaryProvider

    provider = FusionBoundaryProvider(run_dir)
    cases = {case["case_id"]: case for case in provider.catalog()}
    for case_id in CASES:
        case = cases[case_id]
        assert case["planned_status"] == "planned"
        for strategy, count in (("split", 2), ("fused", 1)):
            candidate = provider.prepare(case, strategy)
            assert candidate.meta["launch_verified"]
            assert len(candidate.meta["observed_kernel_names"]) == count
            replay, _ = candidate._replay()
            assert replay.meta["launch_verified"]
    provider.close()
