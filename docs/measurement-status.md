# Measurement status (2026-11-25)

Scope: B (measurement owner). This file reports what actually ran. **No 4090 data
exists yet**; local RTX 4060 numbers are pipeline evidence only.

## Completed locally

| run | device | candidates | blocks | valid pairs | errors | contaminated rows | notes |
|---|---|---|---|---|---|---|---|
| `runs/env-local-3` | RTX 4060 Laptop | — | — | — | — | — | environment lock, UUID, fstype |
| `runs/dryrun-local-4` | RTX 4060 Laptop | reference (torch.compile) | 3 | 12/12 | 0 | 0 | both modes, calibration, telemetry |
| `runs/pilot-local-ref` | RTX 4060 Laptop | reference | 5 | 20/20 | 0 | 10 | stage-1 pause + resume; whole-pair retries |
| `runs/kernel-guard-test` | RTX 4060 Laptop | A (`cand-pilot-adapter`) | 1 | 0/2 | 0 | 0 | launch guard rejected wrong program |
| `runs/ext-gpu-test2` | RTX 4060 Laptop | — | 0 | — | — | — | preflight aborted: another CUDA process held the GPU (`exit 8`) |

Archived, checksummed, non-reportable:

- `fixtures/pilot-4060-local/` — reference-candidate pilot with telemetry on;
  20/20 pairs after resume, 10 contaminated attempt rows kept, 50 unique
  `(case, mode, block, attempt, strategy)` keys. This is the evidence that
  recovery does not double count.
- `fixtures/measurement-synthetic/` — 240 paired + 60 unpaired-default rows,
  every row `synthetic: true`; consumed by C's `fusionaudit build` with
  **0 errors** and 4 case/mode results.

## Calibration (local 4060, must be redone on the 4090)

From `runs/pilot-adapter-local2/calibration_fit.json` (L2 = 32 MiB):

| quantity | value |
|---|---|
| L direct | 9.60 µs/kernel, intercept 1.01 µs (R²=1.0) |
| L replay | 0.85 µs/kernel, intercept 5.92 µs (R²=0.998) |
| BW main | 225.6 GB/s, sizes 64/128/256 MiB |

Raw rows: `calibration.jsonl` (80 direct-launch, 80 replay, 420 bandwidth rows),
recomputable by `fb verify`. The 4090 has 72 MiB L2; the sweep extends to 256 MiB
so the three main sizes can exceed L2.

## Historical blockers

1. **No 4090 instance.** No SSH endpoint, instance UUID, image id or persistent
   volume was provided. Calibration/pilot/full cannot be executed on the target.
2. **A's `CandidateStore.load()` mis-executes split.** Repro and acceptance test
   are in [`measurement-blockers.md`](measurement-blockers.md). B's adapter
   profiles the loaded program and refuses to time it (`prepare_error`), so no
   wrong split row exists.
3. **Full corpus not compiled yet.** A's run only has the two pilots; 70/72
   cases have no candidate rows. On the 4090, A must compile all 72 first.

## Expected full-run accounting

72 cases × 2 modes × 30 blocks = 4,320 paired blocks; ×2 strategies = **8,640
timing rows** if every case is pairable. Cases with a missing/invalid variant are
recorded as `planned_status=excluded` with `exclusion_reason`; failed attempts are
kept and the first complete attempt is selected. Independent `default` timing
adds up to 72 × 2 × 30 = 4,320 unpaired rows only where default is a third
partition.

## Commands

Pilot (two stages, proves resume):

```bash
FB_CANDIDATE_RUN=$FB_RUN_ROOT/pilot-YYYYmmddTHHMMSSZ bash scripts/measure_pilot.sh
```

Full, detached:

```bash
cd ~/tensor-compiler-fusion-boundary && mkdir -p "$FB_RUN_ROOT" && \
FB_CANDIDATE_RUN=$FB_RUN_ROOT/full-YYYYmmddTHHMMSSZ setsid nohup \
  bash scripts/measure_full.sh > "$FB_RUN_ROOT/full-launch.log" 2>&1 < /dev/null &
```

Restart/resume: same `RUN_ID` and `FB_CANDIDATE_RUN` (protocol hash enforced).

## Integration rerun (2026-11-26)

The RTX 4060 integration rerun covered P1 and R1 with the archived TorchInductor
wrappers, actual kernel-launch checks, both submission modes, and numerical
validation. The collector paused after 2 paired blocks and resumed to 16/16
valid pairs. Strict analysis produced four case/mode results with no audit
errors or warnings. This was a development smoke run with external GPU activity
permitted and telemetry disabled. The 4090 remains the target for calibrated,
telemetry-enabled pilot and formal measurements.
