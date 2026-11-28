// Build from the repository root: typst compile --font-path paper/template/fonts paper/report.typ paper/report.pdf
#import "template/approximate-acmsmall.typ": acmart

#let article = acmart(
  format: "acmsmall", nonacm: true, publication: (:),
  title: "When Does Local Kernel Fusion Save Time? Paired Measurements of TorchInductor on Three GPUs",
  shorttitle: "Local Kernel Fusion in TorchInductor",
  authors: ((name: "Junhao Liu", affiliation: none, email: none, orcid: none),),
  abstract: [
    Fusing a producer with its consumer can eliminate a kernel launch and the write and read of an intermediate tensor. We measure the combined saving in TorchInductor by compiling both partitions of a single edge, verifying their generated kernels, and timing the two executions in paired blocks. Under direct CUDA submission, the lower bound of the 95% interval for relative savings exceeded 2% in the same 68 of 72 configurations on both the RTX 3090 and RTX 4090. Under CUDA Graph replay, 25 cases favored fusion on the RTX 3090 and 27 on the RTX 4090, with 13 in common. In a 20-case RTX 3090 reduction follow-up, the median saving fell from 2.36 µs per call under direct submission to 0.68 µs under replay. At fixed shape, transposing the input reduced the percentage saved in all four direct-submission contrasts even as the point estimates of the time saved rose. A model of launch and intermediate-transfer costs matched the compiler default on clear-case accuracy and mean regret in validation, including a 12-case H800 PCIe subset. The default fused every measured edge, and no case in the grid established a practical advantage for splitting. A reduction stress family whose epilogue performs $B$ shifted reads produced one: splitting was faster by up to 28.0 µs (72.9%) for 512 and 1024 rows at $K = 8192$, while fusion still won outside that narrow region.
  ],
  keywords: ("kernel fusion", "TorchInductor", "GPU performance", "CUDA Graphs"),
)
#show: article.show_
#(article.make-title)()

= Introduction

Consider the last two operations in a row reduction. One kernel computes a sum for each row, and another replaces negative sums with zero. The compiler could instead have the threads that finish each sum apply the rectifier before writing the output. Both implementations read the same input and produce the same output. One launches an additional kernel and passes an array of sums between the kernels. How much time should a caller expect to recover by eliminating that boundary? Counting the operations alone gives little guidance. Thousands of input values may contribute to one sum, yet the handoff still moves just one value per row. The cost of issuing the second kernel also depends on whether the caller submits each invocation separately or replays a captured graph.

@fig-overview traces the study from candidate construction to paired measurement and prediction.

#figure(
  image("figures/overview/overview.svg", width: 100%, alt: "A fixed fragment branches to split and fused compiler runs. After validation, independent split timing and target calibration feed frozen predictions, while direct and replay split–fused pairs supply the outcomes used to score the decisions."),
  placement: none,
  caption: [Workflow for evaluating one TorchInductor fusion edge. Candidate validation checks numerical outputs, the generated implementation of the edge and the kernels launched on the device. Target predictions are frozen before paired timing.],
) <fig-overview>

TorchInductor provides a setting in which this question can be asked about executable programs. Its lowering and scheduling decisions determine which operations appear together in generated kernels @inductor @scheduler. A source expression can contain a named intermediate even when the compiled program never stores it. Conversely, a compiler may reuse one allocation for two logical tensors while still writing and reading the intermediate. A comparison between two settings of a compiler flag thus needs an independent check that the setting actually changed the edge. We follow each candidate from the selected expression through its generated wrapper to the kernels observed on the device.


We examine two decisions. The first concerns the edge itself: how does the time saved by fusion change with submission mode, shape and layout? We report both the time saved per call and its fraction of split latency. The second concerns the compiler policy. Given both compiled candidates, a device calibration and an independent timing of the split path, can a model trained on the RTX 3090 select a faster partition than TorchInductor selects by default?

Two RTX devices execute the same grid of pointwise chains and reduction epilogues. The H800 PCIe executes a subset selected before collection on that device. A separate RTX 3090 experiment extends the reduction length, varies input layout and tests the inverse root-mean-square factor used in normalization. Four repeated configurations provide contemporaneous controls for the transposed inputs. Thus the cross-device comparison holds the expression fixed across hosts, whereas the layout comparison changes the stride at a fixed shape on one host.

The distinction matters for interpreting the results. TorchInductor’s default fused every measured edge, so a clear win measures the cost of imposing a split and gives the default and an always-fuse policy the same decision. The grid contains no split-favored example on which a more selective policy could improve beyond the practical margin. A stress family in which the consumer's work grows with the number of shifted reads does contain such cases (@sec-cliff); it changes what the boundary is, not the grid's labels.

= Where the fusion decision occurs

Fusion is a scheduling choice: a compiler must establish legality, construct a combined implementation and decide whether to use it. TVM Relax groups operators for later compilation @relaxfuse; OpenXLA's priority fusion ranks candidates using device-specific cost estimates @priorityfusion @xlacost; FusionStitching generates fused kernels for memory-intensive computations with different parallel structures @fusionstitching; and streaming-inference work asks where to cut an inference region @megakernel. We narrow this to two executable partitions of one edge, so that each timed program can be verified before a latency difference is attributed to the cut.

In the PyTorch compilation path, a caller supplies a Python function to the compiler, while the GPU backend can express tiled operations as Triton kernels @compile @inductor @triton. On the original RTX 3090 grid the pointwise and reduction templates reached the same final partition by different routes. In all 24 pointwise cases the producer was incorporated into its consumer during lowering. By the time the scheduler ran, a veto could no longer act on a separate producer node. In the 48 reduction cases, the scheduler combined the reduction with its epilogue. A one-kernel wrapper therefore identifies the final partition but does not identify the pass at which a split must be imposed.

The physical size of that intermediate suggests a first distinction among cases. A pointwise chain of length $n$ that materializes one FP32 intermediate writes $4n$ bytes of values and reads them back. In an FP32 row reduction, the analogous handoff involves $4M$ bytes each way for $M$ rows, regardless of the row length. In both cases the logical traffic follows from the program. Cache hits, memory transactions, overlap and the cost of the instructions that access those values determine what the handoff costs on a particular device @cudaguide. Treating logical bytes as a traffic counter would attach a hardware interpretation to a quantity obtainable before executing either candidate.

The submission path provides a second distinction. CUDA launches are asynchronous with respect to the host. CUDA events delimit an interval on a stream, which may include gaps before subsequent work reaches the device @cuda @events. Capturing and replaying a sequence changes how that work is issued @graphs. The split candidate still contains its producer and consumer under replay. Their dependence and their intermediate survive capture. Comparing modes thus changes the conditions under which a launch is paid for while preserving an edge whose materialization we can inspect. The difference between the two measured savings incorporates all changes in those conditions. It does not independently measure a host API call.

#include "methods.typ"
#include "prediction.typ"
#include "findings.typ"
#include "cliff.typ"
#include "replication.typ"

= Interpreting the measured boundary

== A pointwise counterpoint to the reductions


#figure(
  image("figures/pointwise-scaling.svg", width: 100%, alt: "Two panels show the six FP32 vector lengths for the affine–ReLU and tanh–affine chains. At small lengths the replay markers are near zero. Both modes approach a 50% saving at the largest length for both chains."),
  placement: none,
  caption: [Relative fusion saving for contiguous FP32 pointwise chains on the RTX 3090. Markers show estimates and whiskers show per-configuration paired 95% intervals; solid and dashed segments follow direct submission and CUDA Graph replay. The shaded band marks the ±2% practical margin. The length axis is logarithmic (base 2).],
) <fig-pointwise>

The reduction follow-up eliminates an intermediate with one scalar per row. In the pointwise grid the intermediate grows with the full input. @fig-pointwise shows how the benefit varies across the sampled lengths. On the RTX 3090, consider P1 with a contiguous FP32 vector of length 1,024. The split and fused paths took 8.146 and 5.096 µs per call under direct submission. Under replay they took 7.958 and 7.880 µs. The replay estimate saved 0.98%, but its interval extended slightly beyond 2%. A decision rule based on the point estimate alone would call the variants equivalent. The interval rule leaves that decision open.

Now keep the expression, type, layout and device fixed, and increase the vector length to 16,777,216. Under direct submission the split time was 321.424 µs and the fused time was 160.886 µs. Under replay the times were 324.411 and 164.274 µs. In absolute terms the saving was about 160 µs in either mode. The logical intermediate at this size occupies 64 MiB. Writing and reading it accounts for 128 MiB of logical accesses across the cut. Eliminating a kernel that moves an entire vector can therefore retain a substantial advantage even when replay makes submission cheaper. The timing and the generated access pattern support this explanation. Establishing how many bytes reached DRAM would require additional evidence.

The intermediate lengths reveal a distinction the two endpoints alone obscure. At 524,288 FP32 elements, P1 under replay saved an estimated 1.40% (95% interval −0.88% to 3.23%). At the same length and precision, P2, whose producer computes a hyperbolic tangent, saved 10.09% (9.92% to 10.25%). Both splits pass a 2 MiB vector to the consumer. A rule that sees only the logical intermediate size must assign both cases the same decision, although the measured evidence for fusion differs. By 4,194,304 elements the two savings were close, 47.51% and 47.02%. These observations bound the comparison to the sampled lengths. They do not locate a length at which splitting wins.

== Limits of the comparison

The practical margin and the timing interval answer different questions. Moving the margin changes the minimum gain worth acting on; narrowing an interval changes what the experiment can resolve. A tight interval inside the equivalence band supports interchangeability at that tolerance, whereas a wide interval spanning the band and a substantial gain does not. This distinction matters most under replay, where the two RTX devices have similar fusion-win counts but very different unresolved counts.

Six replay point estimates on the original grids favored splitting (two on the RTX 3090, four on the RTX 4090), but every interval included zero; for example, the RTX 3090 FP32 P2 chain at length 8,192 saved −0.73% (95% interval −2.25% to 0.42%). These estimates do not establish a practically significant loss from fusion.

The intervals describe variation among paired blocks within one acquisition and cannot estimate variability across days or hosts; the four-case repeat only tests stability near the observed margin. Registers and shared memory describe assigned resources, and management telemetry places a run in device context @smi, but neither shows active warps or cache-level traffic. Those claims need performance counters @nsight. A wrong prediction locates a failed decision; it does not show that occupancy caused the error.

== What the split-favored region changes

The stress family changes what the grid's negative result means. Within the 72 original
configurations the default's choice cannot be improved because the producer kernel is the
same in both partitions; the split candidate differs only by a launch and a handoff. The
family shows that this invariance belongs to those templates, not to fusion: when the
consumer adds enough loads to the same kernel, the producer's launch configuration
changes, and the split candidate can be faster by a practical margin (@sec-cliff).

The region is also a caution for the cost model. The static occupancy difference is zero
for eight of its ten split-favored configurations and positive for one fusion-favored
configuration; the sign follows the epilogue's coupling to the reduction tile. A predictor
that uses only launch count, logical bytes and occupancy therefore cannot locate the
region. Its success on the original grid reflects the absence of such cases rather than
the adequacy of the features.

Finally, the region is narrow. The same family favors fusion at 256 rows, at 2048 rows
and above, at the shorter and longer reduction lengths, and in FP16. Reporting the
boundary requires the four axes, not a claim that fusion is generally harmful.

== How much does a wrong decision cost?

For a repeatedly called fragment, a wrong choice is paid on every invocation; counting errors treats a fraction of a microsecond and a hundred microseconds alike. On the RTX 4090 replay grid, always selecting split incurred mean regret 4.570 µs and maximum 127.732 µs, while always selecting fused incurred 0.00583 µs and 0.221 µs. The latter maximum is computed from measured mean latencies before imposing a margin, so it identifies a case to examine rather than a significant loss.

Feature availability changes the question. In the RTX 3090 follow-up under replay, the two-term rule had zero observed regret on the 12 cases with complete resource metadata and on all 20 cases; the three-term rule had zero on the 12 it could score. A policy deployed for every call must still specify a fallback for the other eight, and crediting a successful fallback would evaluate a different policy.

Finally, each configuration has weight one regardless of invocation frequency. Turning per-case regret into end-to-end saving requires call frequencies, the surrounding schedule and prediction overhead; a predictor that ties the default in steady-state latency has not yet recovered the cost of compiling alternatives and timing the split path.

= Conclusion

For these single-consumer TorchInductor fragments, eliminating a verified intermediate generally helped under direct submission, and replay reduced which advantages were large enough to resolve. The decrease in absolute benefit occurred in all 20 RTX 3090 follow-up configurations. For a large FP32 pointwise vector on the same device, fusion retained about 160 µs of benefit in either mode, so per-call microseconds and relative savings must be reported together.

The two-term predictor matched the default on the measured targets, and the resource term added two clear errors on RTX 4090 replay while reducing coverage in the follow-up. On the original grid there is no measured decision benefit to replacing the default with either predictor. The stress family supplies the missing case: the resource term is zero for eight of its ten split-favored configurations and positive for one fusion-favored one, so neither the default nor the three-term model localizes the region. A legal local fusion can therefore lose by a practical margin, and the measured region bounds where a local decision can differ from the default. How often such epilogue coupling occurs in model graphs remains open.

#pagebreak()
#bibliography("references.bib", title: [References], style: "association-for-computing-machinery")

#pagebreak()
#heading(numbering: none)[Disclosure of AI Assistance]

OpenAI Codex drafted and revised the English text throughout this report from the project's existing research plan, implementation, experimental records and results. AI agents checked numerical statements against the archived summaries, prepared bibliographic entries and reviewed the methods and findings. Codex also assembled the Typst source, drew the workflow diagram with FigGenie, scripted the data plots from the audited experiment tables, and checked the compiled document. The measurements reported here come from the project archives and predate preparation of this manuscript. The stress-family section and the revised boundary discussion were prepared in a later AI-assisted revision from the archived run and its audited table.
