# Research Plan (draft v0.1)

**Title**: Fusion Boundaries: Cross-Hardware Modeling and Prediction of Kernel Fusion Profitability in Tensor Compilers

**Status**: first draft, written after topic selection. Scope is technical merit only; no schedule is fixed.

---

## 1. Summary

Kernel fusion is the highest-leverage graph-level optimization in tensor compilers, and megakernel systems push it to its limit by fusing an entire model into one launch. Yet every production compiler decides *where to stop fusing* with a rule that is fixed at build time and tuned for one hardware class. This project measures the true profitability boundary of fusion on several GPU generations, decomposes the benefit into launch-overhead removal, memory-traffic elimination and occupancy loss, and tests whether a small, physically interpretable cost model predicts that boundary on hardware it was never fitted on. The deliverable is a measured phase boundary, a three-term model, an argument that existing rules fail by *form* rather than by constant, and an out-of-sample validation against three rule families (TVM, TorchInductor, XLA).

## 2. Problem and motivation

- Fusion removes intermediate tensors from HBM and removes kernel launches. Both effects are real, but they scale differently: launch cost is roughly constant per kernel, while memory savings scale with tensor size and inverse bandwidth.
- Current systems push fusion in opposite directions. MPK (OSDI 2026) fuses multi-GPU LLM inference into a single persistent megakernel; MegaBake and megakernel-zoo compile one engine per model and GPU; "One Megakernel is Not Enough" (arXiv 2610.35336, 2026-10-26) reports that in 13 of 13 model/batch combinations the best design is *not* a single megakernel, but 2-4 megakernel regions or a conventional graph.
- The rules that decide this are ad hoc and hardware-blind:
  - TVM Relax `FuseOps` groups nodes by an operator pattern label (`kElemWise`, `kInjective`, `kCommReduce`, `kOpaque`); shape and hardware do not enter the decision.
  - TorchInductor gates fusion with a scalar `score_fusion_memory_threshold` (default 10) and a hard `max_fusion_size` (default 64), both global constants.
  - XLA decides through `InstructionFusion::ShouldFuse`, returning a `FusionDecision` with a reason; heuristics are tuned per backend by hand.
- Portability is an open problem. arXiv 2611.22844 (2026-11-17) adapts Mirage to an Ascend NPU through Triton and reports gains up to 87.0% on fused workloads together with substantial regressions on other shapes, concluding that a portable IR does not imply a portable optimization space.
- Gap: no published criterion predicts the fusion boundary from measurable hardware constants, and no study separates *which* term of the benefit dominates at which scale.

## 3. Research questions

- **RQ1 (decomposition).** For a given subgraph, how much of the fusion benefit comes from launch-overhead removal, how much from memory-traffic elimination, and how much is lost to occupancy and wave quantization?
- **RQ2 (boundary).** Where does the sign of the benefit flip in (subgraph size / arithmetic intensity, SM count) space, and does the boundary move across GPU generations in the direction the decomposition predicts?
- **RQ3 (prediction).** Can a three-parameter model, calibrated with two hardware constants, predict the sign of the benefit on unseen subgraphs and unseen hardware?
- **RQ4 (expressiveness).** Do the three existing rule families misjudge because their constants are wrong, or because a scalar threshold / discrete label cannot express a function of both shape and hardware?

## 4. Hypotheses

- **H1.** The benefit boundary moves with SM count: at fixed subgraph size, fusion becomes relatively more attractive as SM count grows, because per-operator kernels finish sooner while the launch gap stays constant. Falsified if the boundary is flat across 84 and 128 SMs.
- **H2.** A model with three interpretable terms predicts the sign of the benefit with higher accuracy than each framework's own decision rule on held-out subgraphs.
- **H3.** No value of Inductor's scalar threshold reaches the model's accuracy when swept over its full range; the best achievable accuracy is bounded well below the model. This is evidence of a form limitation, not a tuning problem.
- **H4.** Refitting only the hardware constants (launch latency and effective bandwidth) on a new GPU preserves accuracy; refitting the three coefficients is unnecessary.

## 5. Background: what the three rule families encode

| System | Decision site | Inputs to the decision | Hardware awareness |
|---|---|---|---|
| TVM Relax | `src/relax/transform/fuse_ops.cc`, `FuseOps(fuse_opt_level)` | operator pattern label, dataflow legality | none |
| TorchInductor | `torch/_inductor/scheduler.py` (`can_fuse`, `can_fuse_vertical`, `WhyNoFuse`) | memory-score of the fusion, node counts, layout | only through global constants |
| XLA | `xla/hlo/transforms/instruction_fusion.h` (`ShouldFuse`, `ChooseKind`) | profitability heuristic per backend, legality | per-backend, hand-tuned |
| Megakernel systems | MPK, MegaBake, megakernel-zoo | whole-program fusion, per-GPU worker counts | per-GPU configuration, not a rule |

The three rules occupy different points on the legality/cost/hardware axis; none is a function of all three. That is the structural observation the project tests.

## 6. Method

### 6.1 Counterfactual design

The central methodological requirement: **hold the kernels fixed and vary only the partition**. Comparing different frameworks end to end measures code-generation quality, not the fusion rule. For each subgraph the harness therefore produces two measurements on the *same* backend:

- `T_split`: fusion disabled or restricted to single nodes.
- `T_fused`: fusion enabled with the framework's own rule, and separately with maximal fusion.

Benefit: `B = T_split - T_fused`; relative benefit: `Delta = B / T_split`.

Knobs (verified against current sources):

- TorchInductor: `torch._inductor.config.max_fusion_size` (64 -> 1 disables grouping), `torch._inductor.config.score_fusion_memory_threshold` (10 -> large value suppresses marginal fusions); decision reasons via `WhyNoFuse`.
- TVM: `relax.transform.FuseOps(fuse_opt_level)` at levels 0/1/2, and `FuseOpsByPattern` for rule-driven grouping; node patterns observable from the annotated IR.
- XLA: rule extraction only. Parse `FusionDecision` reasons from HLO dumps; do not run the full sweep (import path cost is not justified for a rule that is already documented as per-backend).

### 6.2 Subgraph corpus

Five categories, chosen to separate the terms of the model:

| Category | Examples | Isolates |
|---|---|---|
| A. Elementwise chains | 1-6 chained add/mul/gelu | launch cost vs memory traffic |
| B. Reduction + epilogue | layernorm, softmax, reduce + scale | legal-fusion boundaries, sync |
| C. GEMM epilogue | matmul + bias + activation | compute-bound fusion |
| D. Attention | QK^T, softmax, PV | hard fusion, layout transforms |
| E. Model segments | Llama/Qwen MLP block, ResNet bottleneck | realism check |

Sizes: hidden dimension 512-4096, sequence 128-2048, batch 1-32. Target 30-50 distinct subgraphs, each expressible in at least Inductor and TVM (PyTorch export -> Relax for TVM).

### 6.3 Hardware constants

Measured once per GPU, not fitted:

- `L`: empty-kernel launch latency, median and distribution, with and without CUDA Graph replay.
- `BW_eff`: STREAM-like copy/triad kernel, achieved bandwidth at the sizes in the corpus.
- Occupancy: achieved occupancy and registers per thread from Nsight Compute where permissions allow; otherwise from `ptxas -v` register counts and block sizes as a proxy.

### 6.4 Model

All terms are times; the model predicts the relative benefit:

```
Delta_hat = alpha * (bytes_removed / BW_eff) / T_split
          + beta  * (kernels_removed * L) / T_split
          - gamma * occupancy_loss_ratio
```

`bytes_removed` and `kernels_removed` are computed from the graph, exactly. `BW_eff` and `L` are measured. Only `alpha`, `beta`, `gamma` are estimated, by least squares on training points; sign prediction uses `Delta_hat > 0`. A logistic variant is reported as a robustness check.

Interpretation is the point: `alpha` near 1 means the benefit is memory traffic; `beta` near 1 means it is launch overhead; a large `gamma` marks occupancy-sensitive hardware.

### 6.5 Validation protocol

- **V1, held-out subgraphs.** Fit on 70%, test sign accuracy on 30%.
- **V2, held-out hardware.** Fit coefficients on 3090 Ti (84 SMs), test on 4090 (128 SMs) refitting only `L` and `BW_eff`; then test once on 5090 (170 SMs) as the never-seen machine.
- **V3, baselines.** Each framework's own decision on the same subgraphs; a roofline-only model (bytes term only); always-fuse and never-fuse.
- **V4, end-to-end.** One small LLM under Inductor default versus a partition chosen by the model, where per-region intervention is possible; otherwise report the predicted-versus-actual boundary on the model's subgraphs.

### 6.6 Expressiveness experiment

Sweep `score_fusion_memory_threshold` over a wide range and `max_fusion_size` over 1..64; for each setting compute the accuracy of the resulting decisions against the counterfactual truth. Plot best-achievable accuracy against the threshold. If the curve plateaus below the model's V1 accuracy, H3 is supported.

## 7. Experimental matrix

| Hardware | SMs / BW | Role |
|---|---|---|
| RTX 3090 Ti | 84 / 936 GB/s | development, coefficient fitting |
| RTX 4090 | 128 / 1008 GB/s | first holdout, SM-count contrast |
| RTX 5090 | 170 / 1792 GB/s | final holdout, never used for fitting |
| RTX 4060 Laptop | 24 / 256 GB/s | optional sanity check, extreme small-SM point |

Per hardware: Inductor full sweep over the corpus; TVM sweep over the subset with a working import path; XLA rule extraction on category A and D only.

## 8. Deliverables

1. A measured phase boundary in (size, SM count) space with the three rules' implicit boundaries overlaid.
2. A threshold-sweep curve showing the best achievable accuracy of a scalar rule.
3. A three-coefficient model with out-of-sample accuracy on subgraphs and on one unseen GPU.
4. Failure-case analysis: each misjudgment attributed to a term of the model and to a line of framework code.
5. Optional practice artifact: a partitioner or a patch that applies the model's decision at the region level.

## 9. Threats to validity

- **Codegen confound.** Removing fusion may change which kernels are generated at all. Mitigation: same backend, same kernel family, report generated-kernel counts alongside timing.
- **Knob side effects.** `max_fusion_size` and the memory threshold interact with scheduling and layout. Mitigation: sweep both, report sensitivity.
- **Profiler availability.** Nsight Compute often needs elevated permissions on rented instances. Mitigation: register-count proxy, cross-checked on one machine where ncu works.
- **Backend support on new hardware.** sm_120 support in TVM and Triton is not guaranteed. Mitigation: 5090 holdout runs Inductor only; scope the claim accordingly.
- **Timing noise.** Laptop and consumer cards throttle. Mitigation: lock clocks where possible, report medians and interquartile ranges, run each configuration in a randomized order.
- **Small-sample overfitting.** Three coefficients fitted on tens of points. Mitigation: leave-one-hardware-out, report coefficient confidence intervals.
- **Subgraph extraction bias.** Mitigation: include both synthetic chains and real model segments; state that synthetic categories are controls, not evidence about end-to-end speedups.

## 10. Failure branches

- Boundary does not move across SM counts: report the negative result, re-fit without the launch term, and pivot the claim to occupancy as the driver.
- Model does not beat the rules on held-out subgraphs: report that the rules are adequate within measurement noise, with the threshold-sweep evidence showing the ceiling.
- Per-region intervention is infeasible in Inductor: keep V4 descriptive (predicted vs actual on model subgraphs) and drop the end-to-end speedup claim.

## 11. Work breakdown (no dates)

- **P0 Infrastructure.** Environment on 3090 Ti and 4090; framework builds; reproducible benchmark harness; clock locking.
- **P1 Corpus and counterfactual harness.** Subgraph definitions; export paths; fuse/split toggles; decision-reason logging.
- **P2 Measurement sweep.** Hardware constants; latency and occupancy across the matrix; raw artifact storage.
- **P3 Model.** Fit, interpret coefficients, validate V1-V3.
- **P4 Expressiveness and holdout.** Threshold sweep; 5090 run; failure-case analysis.
- **P5 Artifact and write-up.** Figures, reproduction scripts, optional partitioner, report.

## 12. Team roles (3 people)

- **Harness and framework integration**: export paths, toggles, decision logging, reproducibility.
- **Measurement and profiling**: hardware constants, timing methodology, occupancy, artifact hygiene.
- **Modeling and evaluation**: fitting, validation, figures, failure attribution, writing.

Roles overlap at P3 and P5; every number in the report must be reproducible by a second member.

## 13. Success criteria

The project succeeds if, on at least one unseen GPU, the model predicts the sign of the fusion benefit better than all three framework rules on held-out subgraphs, and the threshold sweep shows that no scalar setting of the existing rule closes the gap. If either fails, the measured boundary, the decomposition and the documented failure modes remain the contribution.
