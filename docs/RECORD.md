# Discussion record: graph-vs-operator optimization

## Background

**User's words**:

> I need to do a course project and want it to reach an excellent level. The project is research plus practice, teams of 2-3 (no bonus for smaller teams). Task 1 is research on language implementation and compilation techniques: a report in week 12, graded excellent (90-100), good (80-89), fair (70-79), poor (<70), plus invited presentations in week 15/16 for the excellent ones. Suggested categories include lexers/parsers, IR and optimization, runtime and memory management, type systems, tools and ecosystem, and AI compilers (TVM, MLIR, XLA; graph-level versus operator-level optimizations in deep learning compilers). I have not chosen a topic yet, but I am roughly aiming at graph optimization. What concrete topic suggestions do you have?

**My reading**:

A research report (plus a possible invited talk) for a compilers course, team of 2-3, graded on a four-band rubric. The likely scoring topic is Category 6 (AI compilers), specifically graph-level versus operator-level optimizations. The goal is the excellent band, not merely passing. No topic is fixed, so the shape of the contribution is still open.

**My guesses (unconfirmed)**:

- The user wants a topic that is defensible under a 90-100 rubric, i.e. produces its own evidence rather than surveying known systems.
- The user has roughly one GPU-class laptop and no prebuilt ML/compiler stack.
- The report deadline may be close, because the current date falls late in a typical autumn semester.

**Constraint found**:

- The machine is a laptop with an RTX 4060 (8 GB VRAM), 20 CPU cores and 23 GB RAM. A CUDA driver is present.
- `$HOME` is mounted on a 2 GB tmpfs, while `~/Workspace` is a separate 475 GB partition. Any environment with PyTorch or TVM must live under `~/Workspace`, not under `$HOME`.
- No PyTorch, TVM or ONNX Runtime is installed. The system Python is 3.14.9, which has no current PyTorch wheel; `uv` is available and can provision Python 3.12/3.13.
- Consequence: experiments are feasible on one 8 GB GPU but must be small-model and bounded in autotuning time, and the toolchain has to be built from scratch before any measurement.

**Recon (2026-11-23)**:

- Academic: the live questions in tensor-graph compilers are (a) where to cut fusion and megakernel boundaries (MPK, OSDI 2026; "One Megakernel is Not Enough", arXiv 2610.35336, 2026-10-26, finds no single megakernel is optimal in any of 13 model-batch combinations); (b) integrating graph-level rewrites with operator-level scheduling (Trinity, ASPLOS 2026, tile-level equality saturation; EqiForge, arXiv 2609.12330, 2026-09-11; Cleave, arXiv 2610.07742, 2026-10-06); (c) portability of graph superoptimization across backends (arXiv 2611.22844, 2026-11-17, Mirage adapted to Ascend 910B3 through Triton: up to 87.0% gain on fused workloads, regressions on other shapes); (d) default-compilation regressions (PassNet, arXiv 2605.29357, 2026-05-28: 43% of real-world subgraphs slow down under TorchInductor defaults, and LLMs reach up to 3x on individual subgraphs); (e) correctness of graph rewrites (DiTOX on the ONNX optimizer, 2025; AlignGuard on torch.compile, 2026).
- GitHub activity on 2026-11-23: active: pytorch/pytorch (Inductor commit the same day), apache/tvm (11-22), openxla/xla, triton-lang/triton, tile-ai/tilelang (9k stars), iree-org/iree, flagos-ai/FlagTree, mirage-project/mirage (2.6k stars), megabake/megabake (39 stars, 11-21). Dead: facebookresearch/TASO (repository 404), microsoft/nnfusion (last push 2024-09).
- Consequence: only the maintained projects above can carry an experiment; most tensor-superoptimizer papers have no runnable artifact.
- Hardware reach of the reference systems: MegaBake is SM90-and-bf16-only; megakernel-zoo refuses any GPU other than sm_90a or sm_100a by design; MPK exposes 96 workers and an 8B-parameter demo and requires worker counts to match physical SMs. The available card is an RTX 4060 Laptop (AD107, 24 SMs, 8 GB, 256 GB/s, sm_89), so every published megakernel baseline has to be rebuilt by hand for it. A community single-author engine for Qwen3-0.6B on an RTX 3090 exists (Pogud/MegaQwen) but carries no verification weight.
- MPK readiness on consumer hardware: maintainers state the demo is tuned for A100-40GB; a user running it on one RTX 4090 reported 25 ms/token under PyTorch against 15 ms/token under MPK, a 40% gain, so sm_89 executes but is untuned. Repository configurations are per-GPU: A100 96 workers over 108 SMs, H100 128 over 132, B200 144 over 160, H20 64 or 72 over 78. Tasks are hand-written kernels and the superoptimizer is not yet integrated with the megakernel.

## Settled

- Project duration is not a constraint: the user explicitly instructed that topic choice ignore time cost; it constrains scoping to technical merit only.
- Proposals must be grounded in dated literature and live code rather than recalled knowledge: the user rejected un-researched advice; it constrains every later suggestion to carry evidence.
- The team writes CUDA kernels and has an HPC background: stated directly by the user; it removes the kernel-engineering risk from the megakernel topic.
- Hardware: the team owns an RTX 3090 Ti and can rent RTX 4090 or 5090 instances; no sm_90a/sm_100a machine is planned. The 4090 (128 SMs, 24 GB) is the primary target because it matches the H100's SM count; the 3090 Ti (84 SMs, sm_86) is the development and cross-check target; the 5090 is optional because neither Mirage nor the megakernel zoos claim support for sm_120.
- Project folder: `tensor-compiler-fusion-boundary` — chosen by the user; it fixes the artifact path and the project's public identity.
- Project title: "Fusion Boundaries: Cross-Hardware Modeling and Prediction of Kernel Fusion Profitability in Tensor Compilers" — proposed by the assistant and adopted when the user selected a folder name derived from it.

## My calls (to review)

- Recommended an empirical attribution study ("how much of the end-to-end gain comes from graph-level passes versus operator-level tuning, and where is the boundary") instead of a survey of XLA/TVM/MLIR; confirmation pending.
- Ranked three candidates after recon: (1) fusion-boundary experiment, (2) gain attribution with a failure atlas, (3) differential testing of graph rewrites; (1) preferred because the 2026 literature asks the same boundary question and it can be measured on the available 8 GB GPU; confirmation pending.
- Reframed candidate (1) as "find the cut criterion", not "reproduce MPK": with hand-built baselines on a 24-SM card, the contribution is a transferable decision rule, since every published megakernel system targets sm_90a/sm_100a and none runs on sm_89; confirmation pending.
- Proposed a three-way decomposition as the measurement spine: per-operator eager, per-operator under CUDA Graph (launch overhead removed, boundaries kept), and megakernel (boundaries removed), so the difference between the last two isolates structural fusion from launch-overhead removal; target 4090, develop on the 3090 Ti.

## Rejected options

## Log

### 2026-11-23 14:17
**Q**: What is the exact deadline of the week-12 report? Recommend: if four weeks or more remain, run the full ablation across two backends and three to four models; if one to two weeks remain, cut to one backend and two small models, or fall back to an analytical report; ask because the deadline decides the experiment's size, which decides whether the topic is feasible at all.
**A**: unanswered
**Decision**: undecided
**New open questions**: deadline date; whether the team aims at a publication-style contribution or only at the top grade band; available GPU time per week.

### 2026-11-23 14:25
**Q**: (instruction rather than question) Run two research passes — academic trends, and GitHub activity with commit dates — then propose topics, ignoring project duration. Recommend: a fusion-boundary experiment, because the 2026 literature asks the same question and it produces comparable measurements on the available GPU.
**A**: instruction accepted; the user requires dated evidence before advice.
**Decision**: literature and repository state recorded under Recon; three candidates ranked under "My calls"; final choice undecided pending the team's kernel skills.
**New open questions**: whether the team can write CUDA/Triton kernels and debug GPU synchronization; that answer decides between the megakernel-boundary topic and the measurement topic.

### 2026-11-23 14:32
**Q**: Can the team write CUDA/Triton kernels and debug GPU synchronization; recommend the megakernel-boundary topic if yes, the attribution topic if no, because the first one depends on kernel engineering while the second does not.
**A**: The team writes CUDA kernels and has some supercomputing background.
**Decision**: candidate (1) adopted, reframed as finding the cut criterion rather than reproducing MPK; every published megakernel engine targets sm_90a or sm_100a, so baselines on the 24-SM card must be built by hand and the result becomes a transfer claim. Constrains the experiment design to baselines the team owns.
**New open questions**: what hardware beyond the 4060 is reachable (cluster, model, count); that decides whether the study runs at datacenter scale or becomes a small-SM transfer study.

### 2026-11-23 14:35
**Q**: What hardware is reachable beyond the laptop GPU; recommend 4090 as the primary target if rentable, because 128 SMs match the H100's 132 and make the SM-count confound small.
**A**: The team can rent 4090 or 5090 instances and owns a 3090 Ti.
**Decision**: primary measurements on a 4090, development and cross-machine checks on the 3090 Ti, 5090 optional; the claim is a decomposition of the megakernel gain into launch-overhead removal and cross-operator structure, with the cut point measured as a function of SM count, bandwidth and shape. Constrains the experiment to three comparable execution modes on the same kernels.
**New open questions**: which model family and size range anchors the workload; whether the deliverable includes an executable cut planner or stops at a measured rule.
