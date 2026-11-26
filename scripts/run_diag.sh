#!/usr/bin/env bash
# Diagnostic controls for the two pilots (archived separately from the pair).
#   bash scripts/run_diag.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
PY=${PY:-$ROOT/.venv/bin/python}
RUN_ROOT=${FB_RUN_ROOT:-$ROOT/runs}
RUN_ID=${RUN_ID:-diag-$(date -u +%Y%m%dT%H%M%SZ)}
LOG="${RUN_ROOT}/${RUN_ID}.driver.log"
mkdir -p "$RUN_ROOT"

"$PY" -m fusionboundary compile --run "${RUN_ROOT}/${RUN_ID}" --pilot \
    --strategy diag_clone,diag_graph_break,diag_no_fusion 2>&1 | tee "$LOG"
"$PY" -m fusionboundary status --run "${RUN_ROOT}/${RUN_ID}" | tee -a "$LOG"
