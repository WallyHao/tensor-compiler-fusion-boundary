#!/usr/bin/env bash
# Audit every case's standalone producer config in its own process.
#
# The gate in README ``审阅发现`` needs one fact per case: does compiling the
# producer as its own graph pick the same tile as the producer kernel inside
# ``fused``?  Each case gets its own cache/trace directories (the probe does
# that), so runs are independent and can overlap; timing from this audit is not
# used for any claim.
#
# Usage:  PYTHON=/path/to/venv/bin/python JOBS=6 bash scripts/run_probe_all.sh [OUT_DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="${1:-$ROOT/runs/probe-standalone-producer-72-$(date -u +%Y%m%dT%H%M%SZ)}"
JOBS="${JOBS:-6}"

mkdir -p "$OUT"
PYTHONPATH="$ROOT/src" "$PY" - > "$OUT/cases.txt" <<'PY'
from fusionboundary import corpus

for case in corpus.build_corpus():
    print(case.case_id)
PY

export ROOT PY OUT
xargs -a "$OUT/cases.txt" -P "$JOBS" -I{} sh -c '
  PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/probe_independent.py" \
    --case "$1" --role producer --out "$OUT" >/dev/null 2>&1 || echo "FAIL $1"
' _ {}

echo "cases: $(wc -l < "$OUT/cases.txt") json: $(ls "$OUT"/json/*.json 2>/dev/null | wc -l)"
echo "$OUT"
