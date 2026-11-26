#!/usr/bin/env bash
# Development dry run: reference candidates, 3 blocks, both modes.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$ROOT/.venv/bin/python}
RUN_ROOT=${FB_RUN_ROOT:-$ROOT/runs}
RUN_ID=${RUN_ID:-dryrun-$(date -u +%Y%m%dT%H%M%SZ)}
EXPECT=${FB_EXPECT_DEVICE:-RTX 4060 Laptop GPU}
mkdir -p "$RUN_ROOT"
"$PY" -m fb collect \
  --purpose dryrun --provider reference \
  --allow-unverified-candidates \
  --run-root "$RUN_ROOT" --run-id "$RUN_ID" \
  --expect-device "$EXPECT" --cases PILOT --blocks 3 --calibrate
echo "dryrun: $RUN_ROOT/$RUN_ID"
