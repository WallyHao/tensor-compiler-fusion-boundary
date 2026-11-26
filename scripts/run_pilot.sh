#!/usr/bin/env bash
# Pilot: P1/N=65536/fp32 and R1/M=256/K=512/fp32, three strategies, evidence, verify.
#
#   bash scripts/run_pilot.sh                 # run id from the current UTC minute
#   RUN_ID=pilot-mine bash scripts/run_pilot.sh
#
# Everything lands in runs/<RUN_ID>/. Safe to re-run: completed variants are
# skipped unless FORCE=1.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
PY=${PY:-$ROOT/.venv/bin/python}
RUN_ROOT=${FB_RUN_ROOT:-$ROOT/runs}
RUN_ID=${RUN_ID:-pilot-$(date -u +%Y%m%dT%H%M%SZ)}
ARGS=("$@")
[[ "${FORCE:-0}" == "1" ]] && ARGS+=(--force)
[[ -n "${DEVICE:-}" ]] && ARGS+=(--device "$DEVICE")

mkdir -p "$RUN_ROOT"
LOG="${RUN_ROOT}/${RUN_ID}.driver.log"
echo "run_id=${RUN_ID} log=${LOG}"
"$PY" -m fusionboundary pilot --run "${RUN_ROOT}/${RUN_ID}" "${ARGS[@]}" 2>&1 | tee "$LOG"
echo
echo "artifacts: ${RUN_ROOT}/${RUN_ID}/artifacts/"
echo "evidence:  ${RUN_ROOT}/${RUN_ID}/evidence/pilot-evidence.md"
echo "status:    PYTHONPATH=src $PY -m fusionboundary status --run ${RUN_ROOT}/${RUN_ID}"
