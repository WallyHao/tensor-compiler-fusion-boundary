#!/usr/bin/env bash
# RTX 3090 followup (plan/followup-3090): write the 20-case manifest and compile
# default/split/fused for each case in its own process with an isolated cache.
#
#   bash scripts/run_followup.sh              # all 20 configurations
#   PILOT=1 bash scripts/run_followup.sh      # the 3 pilot=yes configurations
#   RUN_ID=followup-cand-mine bash scripts/run_followup.sh
#
# The manifest is cases.csv-derived (src/fusionboundary/corpus.py); the phase-1
# 72-case corpus and `--all` are unchanged. Measurement commands are in
# docs/followup-3090-runbook.md.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$ROOT/.venv/bin/python}
RUN_ROOT=${FB_RUN_ROOT:-$ROOT/runs}
RUN_ID=${RUN_ID:-followup-cand-$(date -u +%Y%m%dT%H%M%SZ)}
LOG="${RUN_ROOT}/${RUN_ID}.driver.log"
mkdir -p "$RUN_ROOT"

sel=(--followup)
pilot=0
if [[ "${PILOT:-0}" == "1" ]]; then
  sel+=(--pilot)
  pilot=1
fi

echo "run_id=${RUN_ID} log=${LOG}"
"$PY" -m fusionboundary manifest --run "$RUN_ROOT/$RUN_ID" --followup
"$PY" -m fusionboundary compile --run "$RUN_ROOT/$RUN_ID" "${sel[@]}" \
    --strategy default,split,fused 2>&1 | tee "$LOG"
"$PY" -m fusionboundary status --run "$RUN_ROOT/$RUN_ID" | tee -a "$LOG"
if [[ "$pilot" == "0" ]]; then
  "$PY" -m fusionboundary check-run --run "$RUN_ROOT/$RUN_ID" | tee -a "$LOG"
fi
echo "artifacts: $RUN_ROOT/$RUN_ID/artifacts/"
