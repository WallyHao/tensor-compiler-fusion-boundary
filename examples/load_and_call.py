"""Load two candidates, call each once, verify. No timing protocol here.

    python examples/load_and_call.py runs/pilot-20261125 R1-m256-k512-f32-contig-t
"""

from __future__ import annotations

import json
import sys

from fusionboundary.runtime import CandidateStore


def main(argv: list[str]) -> int:
    run_path = argv[0] if argv else "runs/pilot"
    case_ids = argv[1:] or ["P1-n65536-f32-contig-t", "R1-m256-k512-f32-contig-t"]
    store = CandidateStore(run_path)
    for case_id in case_ids:
        pair = store.load_pair(case_id)
        if not pair:
            print(f"{case_id}: no loadable candidate")
            continue
        for strategy, cand in pair.items():
            x = cand.fixed_input()
            y = cand.call(x)
            rows = cand.verify()
            print(json.dumps({
                "case_id": case_id,
                "strategy": strategy,
                "load": cand.report.summary(),
                "one_call_output": {"shape": list(y.shape), "dtype": str(y.dtype)},
                "checks_passed": sum(1 for r in rows if r["pass"]),
                "checks_total": len(rows),
            }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
