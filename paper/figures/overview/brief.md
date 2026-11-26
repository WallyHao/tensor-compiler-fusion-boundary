# Project overview

Show how the study turns one TorchInductor edge into an audited pair of executable alternatives, then evaluates a decision on the target device. The diagram is an experimental workflow for a systems paper.

- **Candidate construction.** A fixed fragment and edge enter two independent compiler runs. The split run realizes the intermediate and prevents the merge. Ordinary compilation supplies the fused candidate when the generated code spans the edge. Inspect both wrappers and verify the launched kernels and outputs.
- **Target-device evaluation.** Use a source-device fit, target calibration and a separate timing of the split path to freeze a prediction. Collect paired split and fused measurements under direct CUDA submission and graph replay. Use the paired estimates to assign labels and score the frozen decisions.
- **Scope.** Boxes depict decisions and measurements at the selected edge. Other stages of a multi-kernel reduction can remain in both candidates. The validation lane treats the source fit as an input. The compiler default is scored from its independently verified partition in the paper.

## Drawing plan

Use two reading regions across a 444 × 173 pt canvas. Stack a split/fused branch above the shared checks on the left. On the right, keep independent prediction and paired timing boxes on separate rows. Route both outputs to the same scoring box. Borrow the grouped alternatives and orthogonal arrows from FigGenie's KPerfIR and MonoNN exemplars (`osdi25-016_fig1`, `osdi24-052_fig6`), with blue reserved for the fused candidate and near-neutral fills elsewhere. The diagram renders at 394 pt in the manuscript.

Check the split-only timing is upstream of the frozen prediction, paired timings only enter scoring, branches remain legible, and target boxes fit their labels at publication size. The final caption is in the Typst source.

## Editing and export

Edit the labeled node and edge groups in `overview.svg` and update the same IDs in `spec.json` when the content changes. Use FigGenie's `scripts/spec_validate.py` to check the specification and `scripts/check.py` with `--spec spec.json` to validate the drawing. Run `scripts/embed_fonts.py overview.svg -o overview.final.svg` to regenerate the standalone SVG with TeX Gyre Heros embedded. The manuscript reads the editable SVG.
