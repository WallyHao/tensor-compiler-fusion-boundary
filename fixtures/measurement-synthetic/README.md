# SYNTHETIC paired sample (NOT MEASUREMENTS)

Every row in this run is fabricated by `fb.backend.FakeBackend`, and every row
carries `"synthetic": true`. run.json has `"purpose": "synthetic"`.

Purpose: give the analysis owner (C) a minimal, schema-complete paired sample
to build table readers and verification queries before real 4090 data exists.

Contents:
- `cases.jsonl` - 2 cases (P1/N=65536/f32, R1/M=256/K=512/f32)
- `timings.jsonl` - 240 paired rows (2 cases x 2 modes x 30 blocks x 2 strategies)
  plus 60 unpaired `default` rows for the one case whose default is a distinct
  partition (P1), marked `paired: false`
- `variants.jsonl`, `checks.jsonl` - one row per case/strategy and per input seed
- `calibration.jsonl`, `calibration_fit.json` - fabricated launch/bandwidth rows
- `telemetry.jsonl` - fabricated 1 Hz samples
- `case_order.jsonl`, `state/`, `blocks/` - ordering and atomic block commits
- `verification.json` / `verification.md` - independent checks

Do not use any number here in the report. Real data uses a separate run_id and
`synthetic: false`.

Minimal pairing query (first complete attempt per block, both strategies, same n):

```sql
-- conceptual, one row per block attempt
SELECT case_id, mode, block, attempt, n,
       max(CASE WHEN strategy='split' THEN us_per_call END) AS split_us,
       max(CASE WHEN strategy='fused' THEN us_per_call END) AS fused_us
FROM timings
GROUP BY case_id, mode, block, attempt, n
HAVING count(DISTINCT strategy) = 2
   AND min(quality) = 'ok'
   AND count(*) = 2;
```
