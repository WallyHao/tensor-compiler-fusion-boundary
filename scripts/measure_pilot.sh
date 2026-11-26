#!/usr/bin/env bash
# Measurement pilot (B): P1/N=65536/fp32 and R1/M=256/K=512/fp32.
#
# One command, two stages: stage 1 commits two paired blocks and exits cleanly;
# stage 2 resumes the SAME run and must not re-time completed pairs.
#
#   FB_CANDIDATE_RUN=runs/pilot-20261125T120000Z bash scripts/measure_pilot.sh
#
# Optional: RUN_ID, FB_RUN_ROOT (persistent path on the cloud box),
# FB_EXPECT_DEVICE (default "RTX 4090"), FB_EXPECT_UUID,
# FB_CLOUD_PLATFORM/FB_INSTANCE_SPEC/FB_IMAGE_ID, BLOCKS (default 5).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$ROOT/.venv/bin/python}
RUN_ROOT=${FB_RUN_ROOT:-$ROOT/runs}
CAND=${FB_CANDIDATE_RUN:-}
if [[ -z "$CAND" || ! -f "$CAND/variants.jsonl" ]]; then
  echo "FB_CANDIDATE_RUN must point to Agent A's candidate run with variants.jsonl" >&2
  echo "  e.g. FB_CANDIDATE_RUN=runs/pilot-20261125T120000Z $0" >&2
  exit 2
fi
RUN_ID=${RUN_ID:-pilot-meas-$(date -u +%Y%m%dT%H%M%SZ)}
EXPECT=${FB_EXPECT_DEVICE:-RTX 4090}
BLOCKS=${BLOCKS:-5}
mkdir -p "$RUN_ROOT"

COMMON=(
  --run-root "$RUN_ROOT" --run-id "$RUN_ID"
  --provider "$CAND" --expect-device "$EXPECT"
  --blocks "$BLOCKS" --calibrate
)
[[ -n "${FB_EXPECT_UUID:-}" ]] && COMMON+=(--expect-uuid "$FB_EXPECT_UUID")
[[ -n "${FB_CLOUD_PLATFORM:-}" ]] && COMMON+=(--cloud-platform "$FB_CLOUD_PLATFORM")
[[ -n "${FB_INSTANCE_SPEC:-}" ]] && COMMON+=(--instance-spec "$FB_INSTANCE_SPEC")
[[ -n "${FB_IMAGE_ID:-}" ]] && COMMON+=(--image-id "$FB_IMAGE_ID")
[[ -n "${FB_GPU_LOCK:-}" ]] && COMMON+=(--gpu-lock "$FB_GPU_LOCK")

echo "run_id=$RUN_ID run_root=$RUN_ROOT candidate_run=$CAND"
echo "stage 1/2: two blocks, then clean exit (resume fixture)"
"$PY" -m fb pilot "${COMMON[@]}" --stop-after-blocks 2
echo
echo "stage 2/2: resume the same run; completed pairs are skipped"
"$PY" -m fb pilot "${COMMON[@]}"

echo
echo "timings:      $RUN_ROOT/$RUN_ID/timings.jsonl"
echo "verification: $RUN_ROOT/$RUN_ID/verification.md"
echo "environment:  $RUN_ROOT/$RUN_ID/environment/"
echo "follow-up:    PYTHONPATH=src $PY -m fb verify --run-dir $RUN_ROOT/$RUN_ID"
