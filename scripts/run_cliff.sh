#!/usr/bin/env bash
# Reproduce the 2026-11-28 cliff sweep (RTX 4090).
#
# For every (M, K, dtype, B) row: compile the fused and split variants in
# separate processes (fresh caches), then run the paired A/B harness on the
# archived wrappers. See docs/cliff-probe-4090.md for the result.
#
# Usage:  PYTHON=/path/to/venv/bin/python bash scripts/run_cliff.sh [OUT_DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="${1:-$ROOT/runs/cliff-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"

# M K dtype B blocks
SERIES="
256 8192 float32 5 30
512 8192 float32 5 30
512 8192 float32 8 30
1024 2048 float32 5 30
1024 4096 float32 5 30
1024 8192 float32 0 30
1024 8192 float32 1 30
1024 8192 float32 2 30
1024 8192 float32 3 60
1024 8192 float32 4 30
1024 8192 float32 5 30
1024 8192 float32 6 30
1024 8192 float32 7 30
1024 8192 float32 8 30
1024 8192 float32 10 30
1024 8192 float32 12 30
1024 8192 float16 5 30
1024 32768 float16 5 30
2048 8192 float32 5 30
2048 8192 float32 8 30
2048 8192 float32 12 30
4096 8192 float32 5 30
4096 8192 float32 8 30
4096 8192 float32 12 30
"

echo "$SERIES" | while read -r M K DT B BLOCKS; do
  [ -z "${M:-}" ] && continue
  tag="wide-M${M}-K${K}-${DT}-b${B}"
  echo "-- $tag"
  if [ ! -f "$OUT/${tag}__fused_output_code.py" ]; then
    rm -rf "$OUT/cache/${tag}" "$OUT/trace/${tag}"
    PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/probe_cliff.py" \
      --variant wide --M "$M" --K "$K" --dtype "$DT" --branches "$B" \
      --role fused --out "$OUT" --blocks 5 --per-block 5
    PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/probe_cliff.py" \
      --variant wide --M "$M" --K "$K" --dtype "$DT" --branches "$B" \
      --role split --out "$OUT" --blocks 5 --per-block 5
  fi
  PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/probe_cliff_paired.py" \
    --fused-code "$OUT/${tag}__fused_output_code.py" \
    --split-code "$OUT/${tag}__split_output_code.py" \
    --M "$M" --K "$K" --dtype "$DT" --branches "$B" \
    --out "$OUT" --blocks "$BLOCKS" --per-block 10
done

PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/cliff_summary.py" --run "$OUT"
echo "$OUT"
