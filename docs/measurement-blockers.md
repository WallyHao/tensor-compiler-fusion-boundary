# Measurement blockers and handoff to A

The entries below record the original failure. The archived-wrapper loader now
uses the verified code bytes instead of resolving the FX graph through Dynamo.
The local two-case integration run validated the split and fused launches in both
submission modes. Acceptance on the RTX 3090 host remains pending.

Date: 2026-11-25 (local RTX 4060 Laptop, torch 2.6.0+cu124, triton 3.2.0)

## B1 — `CandidateStore.load()` does not reproduce the split artifact

`split_guard`/`realize_edge_at_lowering` are applied by `strategies.compile_variant`
but **not** by `runtime.CandidateStore.load`. In a fresh process, loading the split
candidate and observing actual launches gives the fused kernel:

```
split artifact ['triton_poi_fused_add_mul_0', 'triton_poi_fused_relu_1']
observed       ['triton_poi_fused_add_mul_relu_0']
```

Repro (from the repo root, with A's candidate run):

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import torch
from fusionboundary.runtime import CandidateStore
c = CandidateStore("runs/cand-pilot-adapter").load("P1-n65536-f32-contig-t", "split")
x = c.fixed_input()
for _ in range(2): c.call(x)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    for _ in range(3): c.call(x)
    torch.cuda.synchronize()
print(sorted({e.name for e in p.events()
              if "CUDA" in str(e.device_type) and e.name
              and "memcpy" not in e.name.lower()}))
PY
```

Expected: the two kernels listed in
`artifacts/P1-n65536-f32-contig-t/split/kernels.json`. Observed: the fused kernel.
So B cannot time a split/fused pair: the "split" side is actually the fused program.

B's adapter now fails closed: it profiles each loaded variant once (outside timed
regions) and compares the observed kernel names with `kernel_names`; a mismatch
raises `prepare_error` and no timing row is written for that case.

Hypothesis for the fix: `load()` must recreate the same compilation conditions as
`compile_variant` (edge realization + split guard) and must not resolve to another
variant's compiled code. A minimal acceptance test: in one process, load split and
fused back-to-back, then profile each; observed kernel name sets must equal
`kernels.json` for both. The same test should run for the R1 pilot.

After the fix, rerun B's pilot (same commands as the runbook §4) and compare the
`checks.jsonl` and `variants.jsonl` kernel names; only then start the full run.

## B2 — no cloud 4090 instance is reachable from this environment

No SSH target, instance UUID, image id or persistent-volume mount point was
provided. B cannot log in, so calibration/pilot/full on the 4090 are **not done**;
the commands in `docs/measurement-runbook.md` are ready to run once the instance
is supplied. Do not reuse the local 4060 numbers.

## B1 update (2026-11-26): fresh candidate run, same failure

After A added `strategies.split_context` (runtime.py 23:56, strategies.py 23:52),
the two pilots were recompiled into `runs/cand-pilot-adapter2` (6/6 variants valid,
split `kernels.json` lists 2 kernels). Reloading in a fresh process still executes
the fused kernel:

```
split expected ['triton_poi_fused_add_mul_0', 'triton_poi_fused_relu_1']
observed       ['triton_poi_fused_add_mul_relu_0']
report.warnings = ['split guard did not fire during reload']
report.cache_activation.patched = []
```

`activate_cache_dir(path)` returns `patched: []` on torch 2.6.0 because
`torch._inductor.codecache.cache_dir` / `runtime_utils.cache_dir` are functions,
not patched string attributes. The split compilation therefore likely resolves
its cache before/independently of the switch, and reload lands on a fused entry.
Suggested acceptance test (must pass before B can pair):

```python
store = CandidateStore(run)
a = store.load(case, "split"); b = store.load(case, "fused")
# profile 3 calls of each; observed name sets must equal kernels.json for each,
# and the split set must contain 2 kernels for the pilots.
```
