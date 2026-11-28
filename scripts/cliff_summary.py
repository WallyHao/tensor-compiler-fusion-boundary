#!/usr/bin/env python3
"""Reduce a cliff-probe run directory to one summary.json.

Reads the ``probe_cliff.py`` (per-role compile evidence) and
``probe_cliff_paired.py`` (paired A/B) outputs and writes a table with the
launch config of each side, the heuristic counters read from the generated
code, and the label implied by the paper's ±2% rule on the relative CI.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELDS = ("block", "num_warps", "num_stages", "n_regs", "n_spills",
          "occupancy", "shared_bytes")


def counters(code_path: Path) -> dict:
    if not code_path.exists():
        return {}
    text = code_path.read_text()
    out = {}
    for key in ("num_load", "num_reduction"):
        m = re.search(rf"{key}': (\d+)", text)
        out[key] = int(m.group(1)) if m else None
    return out


def kernels(doc: dict) -> list[dict]:
    rows = []
    for graph in doc.get("graphs", []):
        for k in graph["kernels"]:
            r = graph["resources"].get(k["name"], {})
            rows.append({"name": k["name"], "aten": k.get("original_aten"),
                         **{f: r.get(f) for f in FIELDS}})
    return rows


def label(ci_rel: list[float] | None) -> str:
    if not ci_rel:
        return "missing"
    lo, hi = ci_rel
    if lo > 0.02:
        return "fuse"
    if hi < -0.02:
        return "split"
    if -0.02 <= lo and hi <= 0.02:
        return "equivalent"
    return "uncertain"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = {"run": run.name, "series": [], "labels": {}}

    for paired_path in sorted((run / "json").glob("paired-*.json")):
        p = json.loads(paired_path.read_text())
        branch = int(re.search(r"-b(\d+)\.json$", paired_path.name).group(1))
        tag = f"wide-M{p['M']}-K{p['K']}-{p['dtype']}-b{branch}"
        fused_doc = json.loads((run / "json" / f"{tag}__fused.json").read_text())
        split_doc = json.loads((run / "json" / f"{tag}__split.json").read_text())
        fused_code = run / f"{tag}__fused_output_code.py"
        split_code = run / f"{tag}__split_output_code.py"
        rel_ci = p["B_ci95_relative"]
        out["series"].append({
            "M": p["M"], "K": p["K"], "dtype": p["dtype"], "B": branch,
            "t_fused_us": p["T_fused_us"], "t_split_us": p["T_split_us"],
            "B_us": p["B_us"], "B_ci95_us": p["B_ci95_us"],
            "relative": p["relative_benefit"], "relative_ci95": rel_ci,
            "label": label(rel_ci), "outputs_match": p["outputs_match"],
            "fused": {"kernels": kernels(fused_doc), "counters": counters(fused_code)},
            "split": {"kernels": kernels(split_doc),
                      "guard_calls": split_doc.get("guard_blocks"),
                      "realize": split_doc.get("realize_log"),
                      "counters": counters(split_code)},
        })

    counts = {}
    for row in out["series"]:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    out["labels"] = counts
    path = run / "summary.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"series": len(out["series"]), "labels": counts,
                      "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
