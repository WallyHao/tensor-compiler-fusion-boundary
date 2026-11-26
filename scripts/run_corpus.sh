#!/usr/bin/env bash
# Full 72-case corpus: manifest + default/split/fused per case, sequential on one GPU.
#
#   bash scripts/run_corpus.sh
#   TEMPLATES=P1,P2 bash scripts/run_corpus.sh                 # subset
#   STRATEGY=split,fused DTYPES=float32 bash scripts/run_corpus.sh
#
# Run it inside tmux/nohup on the 4090 instance; each case/strategy is its own
# process with an isolated Inductor cache, so a crash costs one variant.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
PY=${PY:-$ROOT/.venv/bin/python}
RUN_ROOT=${FB_RUN_ROOT:-$ROOT/runs}
RUN_ID=${RUN_ID:-corpus-$(date -u +%Y%m%dT%H%M%SZ)}
LOG="${RUN_ROOT}/${RUN_ID}.driver.log"
mkdir -p "$RUN_ROOT"

cmd=("$PY" -m fusionboundary compile --run "${RUN_ROOT}/${RUN_ID}" --all)
[[ -n "${TEMPLATES:-}" ]] && cmd+=(--templates "$TEMPLATES")
[[ -n "${DTYPES:-}" ]] && cmd+=(--dtypes "$DTYPES")
[[ -n "${STRATEGY:-}" ]] && cmd+=(--strategy "$STRATEGY")
[[ -n "${LIMIT:-}" ]] && cmd+=(--limit "$LIMIT")
[[ "${FORCE:-0}" == "1" ]] && cmd+=(--force)

echo "run_id=${RUN_ID} log=${LOG}"
if "${cmd[@]}" 2>&1 | tee "$LOG"; then
  :
else
  rc=${PIPESTATUS[0]}
  if [[ "$rc" -ne 2 ]]; then
    exit "$rc"
  fi
  echo "Non-valid variants recorded. Inspect exclusions in the status table."
fi
"$PY" -m fusionboundary status --run "$RUN_ROOT/$RUN_ID" | tee -a "$LOG"
if [[ -z "${TEMPLATES:-}${DTYPES:-}${STRATEGY:-}${LIMIT:-}" ]]; then
  "$PY" -m fusionboundary check-run --run "$RUN_ROOT/$RUN_ID" | tee -a "$LOG"
fi
