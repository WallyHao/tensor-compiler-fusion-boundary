= Tracing a result to the executable

== A paired observation in the archive

The first direct-submission block of the 256-row inverse-RMS case is in run `followup-full-meas-3090-20261126T082429Z`, lines 661–662 of `timings.jsonl`: same FP32 shape, block zero and attempt. Fused ran first at 11.4767 µs per call (5.416992 ms / 472); split took 13.0646 µs (6.166496 ms / 472), saving 1.5879 µs. Both records passed the quality check, and the 30-block mean is 1.588 µs. The split wrapper writes the row mean with one kernel and reads it in a pointwise kernel that adds the offset and evaluates the reciprocal square root; the fused wrapper emits one reduction kernel whose output already includes both. These kernel names connect the archive to the loaded program and establish materialization; the cache level serving the read remains unknown.

== A loader failure found during development

The need for this check emerged in the pointwise pilot: a split artifact for 65,536 FP32 elements loaded in a fresh process produced one kernel combining affine and ReLU. Numerical checks could not reveal it because both partitions compute the same function; timing would have compared fused with itself. The loader now verifies the wrapper's SHA-256 digest, loads it directly, and after warm-up profiles three invocations, requiring each expected kernel name and multiplicity. Replay also checks for a prologue kernel that fills the graph's RNG offset; event timings include that prologue, which matters when comparing one computational launch.

#include "appendix-tables.typ"

== Reconstructing the reported summaries

The repository distributes four archives (base RTX 3090, RTX 4090, H800, and follow-up/repeat), each with a digest; a checksum manifest covers each formal run. After verification, the paired analysis recovers blocks and per-configuration estimates on a CPU using `paper/evidence.md`. The five runs yield 144, 144, 24, 40 and 8 configuration–mode results; each strict audit recorded zero errors and warnings. `paper/build_results.py` reconstructs estimates, threshold counts, layout contrasts and decision metrics and checks the source-model and split-only digests. Keeping the prediction file with its freeze time shows that target decisions preceded paired acquisition.

The repository accompanying this manuscript contains the four archives covering five runs, their SHA-256 manifests, the analysis and plotting code, and `paper/evidence.md` with CPU-only reconstruction commands. The bootstrap uses seed 2026 and 10,000 block resamples. The archives and analysis code are available from the authors.
