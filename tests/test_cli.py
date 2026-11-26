"""CLI wiring: manifest, case selection, status table (no GPU)."""

from __future__ import annotations

import argparse
import json
import os

import pytest

from fusionboundary import cli, corpus, records


def _args(**kw):
    base = dict(case=None, pilot=False, all=False, templates=None, dtypes=None,
                limit=0, strategy=None, device=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_select_cases_pilot():
    cases = cli._select_cases(_args(pilot=True))
    assert [c.case_id for c in cases] == ["P1-n65536-f32-contig-t", "R1-m256-k512-f32-contig-t"]


def test_select_cases_all_and_filters():
    assert len(cli._select_cases(_args(all=True))) == 72
    assert len(cli._select_cases(_args(all=True, templates="P1"))) == 12
    assert len(cli._select_cases(_args(all=True, dtypes="float16"))) == 36
    assert len(cli._select_cases(_args(all=True, limit=5))) == 5


def test_select_cases_by_id_and_unknown():
    cases = cli._select_cases(_args(case=["R2-m16-k128-f16-contig-t"]))
    assert cases[0].case_id == "R2-m16-k128-f16-contig-t"
    with pytest.raises(SystemExit):
        cli._select_cases(_args(case=["nope"]))


def test_strategy_parsing():
    assert cli._strategies(_args()) == ["default", "split", "fused"]
    assert cli._strategies(_args(strategy="split,fused")) == ["split", "fused"]
    with pytest.raises(SystemExit):
        cli._strategies(_args(strategy="bogus"))


def test_manifest_command(tmp_path, capsys):
    run = tmp_path / "runs" / "t1"
    rc = cli.main(["manifest", "--run", str(run)])
    assert rc == 0
    rows = [json.loads(line) for line in open(run / "cases.jsonl", encoding="utf-8")]
    assert len(rows) == 72
    assert os.path.exists(run / "cases.json")


def test_status_command_without_variants(tmp_path, capsys):
    run = tmp_path / "runs" / "t2"
    cli.main(["manifest", "--run", str(run)])
    rc = cli.main(["status", "--run", str(run)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "72 cases" in out
    assert "unstarted cases: 72" in out


def test_status_renders_variant_rows(tmp_path, capsys):
    run = tmp_path / "runs" / "t3"
    cli.main(["manifest", "--run", str(run)])
    rows = [
        {"case_id": "P1-n65536-f32-contig-t", "strategy": s, "status": st,
         "kernel_count": 1 if s != "split" else 2, "artifact_dir": f"artifacts/x/{s}"}
        for s, st in (("default", "valid"), ("fused", "valid"), ("split", "valid"))
    ]
    rows[0]["default_maps_to"] = "fused"
    records.write_jsonl(str(run / "variants.jsonl"), rows)
    records.write_jsonl(str(run / "checks.jsonl"), [
        {"case_id": "P1-n65536-f32-contig-t", "strategy": "split", "pass": True},
        {"case_id": "P1-n65536-f32-contig-t", "strategy": "fused", "pass": False},
    ])
    assert cli.main(["status", "--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "P1-n65536-f32-contig-t" in out
    assert "valid:2k" in out
    assert "1/2" in out          # one of two checks failed
    assert "fused" in out


def test_partition_map_written(tmp_path):
    run_dir = records.RunDir(str(tmp_path / "runs"), "t4")
    case = corpus.get_case("P1-n1024-f32-contig-t")
    # three identical partitions -> default maps to fused, no unrelated change
    for strategy in ("default", "split", "fused"):
        d = run_dir.artifact_dir(case.case_id, strategy)
        records.write_json(os.path.join(d, "kernels.json"), {"kernels": [
            {"name": "k0", "source_nodes": ["mul", "add", "relu"],
             "original_aten": ["aten.add", "aten.relu"], "reads": ["arg0_1"],
             "writes": ["buf0"], "grid": "grid(1024)", "source_sha256": "abc"},
        ]})
    rows = {s: {"case_id": case.case_id, "strategy": s, "status": "valid"} for s in
            ("default", "split", "fused")}
    records.write_jsonl(run_dir.variants_jsonl, rows.values())
    cli._annotate_mapping(run_dir, case, rows)
    mapping = json.load(open(run_dir.path + "/artifacts/" + case.case_id + "/partition_map.json"))
    assert mapping["partition_equal_default_fused"] is True
    assert mapping["default_maps_to"] == "fused"
    assert mapping["unrelated_partition_change"] is False
    table = {r["strategy"]: r for r in records.read_jsonl(run_dir.variants_jsonl)}
    assert table["default"]["default_maps_to"] == "fused"


def test_partition_map_flags_unrelated_rewrite(tmp_path):
    run_dir = records.RunDir(str(tmp_path / "runs"), "t5")
    case = corpus.get_case("P1-n1024-f32-contig-t")
    split_kernels = [
        {"name": "k0", "source_nodes": ["mul", "add"], "original_aten": ["aten.add"],
         "reads": ["arg0_1"], "writes": ["buf1"], "grid": "g", "source_sha256": "a"},
        {"name": "k1", "source_nodes": ["relu"], "original_aten": ["aten.relu"],
         "reads": ["buf1"], "writes": ["buf2"], "grid": "g", "source_sha256": "b"},
    ]
    fused_kernels = [
        {"name": "k0", "source_nodes": ["mul", "add", "relu"],
         "original_aten": ["aten.add", "aten.relu"], "reads": ["arg0_1"],
         "writes": ["buf0"], "grid": "g", "source_sha256": "c"},
        {"name": "kX", "source_nodes": ["clone"], "original_aten": ["aten.clone"],
         "reads": ["arg0_1"], "writes": ["buf7"], "grid": "g", "source_sha256": "d"},
    ]
    for strategy, ks in (("split", split_kernels), ("fused", fused_kernels)):
        d = run_dir.artifact_dir(case.case_id, strategy)
        records.write_json(os.path.join(d, "kernels.json"), {"kernels": ks})
    rows = {s: {"case_id": case.case_id, "strategy": s, "status": "valid"}
            for s in ("split", "fused")}
    records.write_jsonl(run_dir.variants_jsonl, rows.values())
    cli._annotate_mapping(run_dir, case, rows)
    mapping = json.load(open(run_dir.path + "/artifacts/" + case.case_id + "/partition_map.json"))
    assert mapping["same_op_set_split_fused"] is False
    assert mapping["unrelated_partition_change"] is True


def _fake_run(tmp_path, case_id="P1-n1024-f32-contig-t"):
    """A minimal run with three valid variants and their artifacts."""
    run = records.RunDir(str(tmp_path / "runs"), "t6")
    corpus.write_manifest(run.cases_jsonl)
    rows = []
    for strategy in ("default", "split", "fused"):
        art = run.artifact_dir(case_id, strategy)
        code = os.path.join(art, "output_code.py")
        with open(code, "w", encoding="utf-8") as fh:
            fh.write("# synthetic\n")
        for name in ("kernels.json", "edge_evidence.json", "compile_key.json", "evidence.md"):
            with open(os.path.join(art, name), "w", encoding="utf-8") as fh:
                fh.write("{}\n")
        rows.append({"case_id": case_id, "strategy": strategy, "status": "valid",
                     "artifact_dir": os.path.relpath(art, run.path), "kernel_count": 1,
                     "output_code_sha256": records.sha256_file(code)})
    records.write_jsonl(run.variants_jsonl, rows)
    records.write_jsonl(run.checks_jsonl, [
        {"case_id": case_id, "strategy": s, "input_id": f"in{i}", "mode": "direct", "pass": True}
        for s in ("default", "split", "fused") for i in range(4)
    ])
    return run


def test_check_run_clean(tmp_path, capsys):
    run = _fake_run(tmp_path)
    assert cli.main(["check-run", "--run", run.path]) == 0
    out = capsys.readouterr().out
    assert '"problems": []' in out
    assert '"valid": 3' in out


def test_check_run_detects_hash_drift(tmp_path, capsys):
    run = _fake_run(tmp_path)
    rows = records.read_jsonl(run.variants_jsonl)
    rows[-1]["output_code_sha256"] = "0" * 64
    records.write_jsonl(run.variants_jsonl, rows)
    assert cli.main(["check-run", "--run", run.path]) == 2
    assert "hash differs" in capsys.readouterr().out


def test_check_run_detects_missing_strategy_and_checks(tmp_path, capsys):
    run = _fake_run(tmp_path)
    rows = [r for r in records.read_jsonl(run.variants_jsonl) if r["strategy"] != "split"]
    records.write_jsonl(run.variants_jsonl, rows)
    assert cli.main(["check-run", "--run", run.path]) == 2
    assert "only" in capsys.readouterr().out
