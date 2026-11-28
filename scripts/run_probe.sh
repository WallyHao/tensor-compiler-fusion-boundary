#!/usr/bin/env bash
# Driver for scripts/probe_independent.py: one fresh process per (case, role).
#
# Usage:  PYTHON=/path/to/venv/bin/python bash scripts/run_probe.sh [OUT_DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="${1:-$ROOT/runs/probe-independent-$(date -u +%Y%m%dT%H%M%SZ)}"

CASES=(
  "R1-m4096-k8192-f32-contig-t"
  "R1-m4096-k8192-f16-contig-t"
  "R1-m256-k8192-f32-contig-t"
)
ROLES=("producer" "consumer" "fused" "independent" "paired")

mkdir -p "$OUT/json"
{
  echo "run_id $(basename "$OUT")"
  echo "python $PY"
  echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/probe.log"

for case_id in "${CASES[@]}"; do
  for role in "${ROLES[@]}"; do
    echo "== $case_id $role $(date -u +%H:%M:%SZ)" | tee -a "$OUT/probe.log"
    PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/probe_independent.py" \
      --case "$case_id" --role "$role" --out "$OUT" >> "$OUT/probe.log" 2>&1 \
      || { echo "FAILED $case_id $role (see probe.log)" | tee -a "$OUT/probe.log"; }
  done
done

echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/probe.log"
echo "$OUT"
