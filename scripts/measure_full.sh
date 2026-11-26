#!/usr/bin/env bash
# Formal measurement (B): all eligible cases, 30 paired blocks per case/mode,
# both submit modes, calibration, telemetry, separate-process profiling, seal.
#
# Detached launch (survives terminal disconnect):
#   cd ~/tensor-compiler-fusion-boundary && mkdir -p runs && \
#   FB_CANDIDATE_RUN=runs/cand-YYYYmmddTHHMMSSZ setsid nohup \
#     bash scripts/measure_full.sh > runs/full-launch.log 2>&1 < /dev/null &
#
# Restart after a crash with the SAME RUN_ID to resume (protocol hash checked;
# completed pairs are not re-timed).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$ROOT/.venv/bin/python}
RUN_ROOT=${FB_RUN_ROOT:-$ROOT/runs}
CAND=${FB_CANDIDATE_RUN:-}
if [[ -z "$CAND" || ! -f "$CAND/variants.jsonl" ]]; then
  echo "FB_CANDIDATE_RUN must point to Agent A's candidate run with variants.jsonl" >&2
  exit 2
fi
RUN_ID=${RUN_ID:-full-meas-$(date -u +%Y%m%dT%H%M%SZ)}
EXPECT=${FB_EXPECT_DEVICE:-RTX 3090 Ti}
mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/$RUN_ID.driver.log"
mkdir -p "$RUN_ROOT/$RUN_ID"
echo "run_id=$RUN_ID log=$LOG candidate_run=$CAND"

COMMON=(
  --run-root "$RUN_ROOT" --run-id "$RUN_ID"
  --provider "$CAND" --expect-device "$EXPECT" --calibrate
)
[[ -n "${FB_EXPECT_UUID:-}" ]] && COMMON+=(--expect-uuid "$FB_EXPECT_UUID")
[[ -n "${FB_CLOUD_PLATFORM:-}" ]] && COMMON+=(--cloud-platform "$FB_CLOUD_PLATFORM")
[[ -n "${FB_INSTANCE_SPEC:-}" ]] && COMMON+=(--instance-spec "$FB_INSTANCE_SPEC")
[[ -n "${FB_IMAGE_ID:-}" ]] && COMMON+=(--image-id "$FB_IMAGE_ID")
[[ -n "${FB_GPU_LOCK:-}" ]] && COMMON+=(--gpu-lock "$FB_GPU_LOCK")

# 1) paired timing (full catalog = every case for which A produced both candidates)
"$PY" -m fb collect --purpose full "${COMMON[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

if [[ "$RC" -ne 0 ]]; then
  echo "collection exited $RC; not profiling. Inspect $RUN_ROOT/$RUN_ID and rerun with RUN_ID=$RUN_ID." | tee -a "$LOG"
  exit "$RC"
fi

# 2) profiling in a separate process, only after all paired timing is done
if ! "$PY" -m fb profile --run-dir "$RUN_ROOT/$RUN_ID" --cases PROFILE --iters 10 2>&1 | tee -a "$LOG"; then
  echo "WARNING: profiling failed; paired timings are unaffected" | tee -a "$LOG"
fi

# 3) final checksums + manifest
"$PY" -m fb seal --run-dir "$RUN_ROOT/$RUN_ID" 2>&1 | tee -a "$LOG"
"$PY" -m fb verify --run-dir "$RUN_ROOT/$RUN_ID" 2>&1 | tee -a "$LOG"
echo "done: $RUN_ROOT/$RUN_ID (SHA256SUMS, verification.md, freeze/manifest.json)"
