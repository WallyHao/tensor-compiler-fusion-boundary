"""The committed 72-case manifest must equal a fresh generation (no drift)."""

from __future__ import annotations

import os

from fusionboundary import corpus

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED = os.path.join(REPO, "manifest", "phase1-72-cases.jsonl")


def test_committed_manifest_matches_generator(tmp_path):
    fresh = str(tmp_path / "cases.jsonl")
    corpus.write_manifest(fresh)
    with open(fresh, encoding="utf-8") as fh:
        generated = fh.read()
    with open(COMMITTED, encoding="utf-8") as fh:
        committed = fh.read()
    assert committed == generated, (
        "manifest/phase1-72-cases.jsonl is stale; regenerate it with "
        "`PYTHONPATH=src python -m fusionboundary manifest --run runs/manifest-<UTC>`")
