= Experimental design

== Workloads and selected edges

The unit of comparison is a single intermediate $t$ with one producer and one consumer; each configuration fixes the expression, shape, element type, layout and edge. A split candidate materializes $t$; a fused candidate computes across it. Both return the same logical result through one compiled entry point, but the compiler chooses how each kernel performs its work, so elapsed-time differences include generated-code changes on both sides of the edge.

Two pointwise templates vary the intermediate size directly. P1 applies a rectified linear unit to $t = 1.25x + 0.25$; P2 computes $t = tanh(x)$ and returns $0.75t + 0.25$. Writing and reading the full intermediate moves a vector the size of the input twice. The grid uses 1,024, 8,192, 65,536, 524,288, 4,194,304 and 16,777,216 elements in FP16 and FP32, giving 24 pointwise configurations.

The other two original templates reduce an $M times K$ array along its second dimension. R1 sums each row in FP32 and applies a rectified linear unit. R2 takes the row maximum and applies the affine function used in P2. The maximum retains the input type. For these reductions, increasing $K$ increases the number of inputs contributing to each output while leaving the intermediate length at $M$. Changing $M$ alters both the number of independent rows and the intermediate length. The original grid crosses three row counts (16, 256, and 4,096), four reduction lengths (128, 512, 2,048, and 8,192), and the same two input types, giving 48 contiguous reduction configurations.

RTX 3090 and 4090 each cover all 72 original configurations; H800 PCIe covers 12 preselected before collection. A second RTX 3090 experiment adds eight long reductions, four transposed inputs and four inverse-root-mean-square fragments (N1). N1 computes each row's mean squared entry and returns the reciprocal square root after adding $10^(-5)$, an intermediate in RMS normalization @rmsnorm; timing ends before multiplying the input by that factor. Four original configurations were re-measured as contiguous controls. In each layout comparison the transposed view has shape $(M,K)$ and stride $(1,M)$, its control stride $(K,1)$, with identical logical values at different physical addresses.

@fig-workload-coverage summarizes which comparisons each device supports.

#figure(
  image("figures/workload-coverage.svg", width: 100%, alt: "The original RTX grids each contain 12 cases of each pointwise family and 24 of each reduction family. The H800 row covers a selected subset. Only the follow-up includes inverse RMS."),
  caption: [Measured configurations by workload family and cohort. Each configuration contributes one count to its row across both submission modes. Four follow-up controls repeat original RTX 3090 configurations.],
) <fig-workload-coverage>

== Constructing and checking the candidates

Using PyTorch 2.6.0+cu124 and Triton 3.2.0, we compile the same function in three runs: default, fused and split. Default and fused use ordinary Inductor compilation; the default run records Inductor's partition. A fused candidate qualifies only if one kernel spans the edge. The split run materializes $t$: because Inductor can inline the affine producer of P1 before scheduling, a scheduler veto alone would leave both expressions in one kernel; the split implementation realizes $t$ during lowering and then vetoes the merge. This keeps one compiled graph and no Python call between producer and consumer.

Each candidate carries evidence of how the compiler implemented the edge. In split, the producer must write $t$ with the expected shape and type and the consumer must read it; an allocation count alone would miss a consumer overwriting the storage it just read, so we inspect kernel bodies and buffer arguments. In fused, one kernel must span both computations without storing and reloading $t$. All measured configurations mapped to fused; an edge that ordinary compilation leaves split would need a separate fused candidate.

Validation generates three pseudorandom CPU inputs (seeds 17, 29, 43) and an all-zero input, casts them to the specified type and verifies shape, type and stride; eager expressions provide the reference. Absolute and relative tolerances are $(2 times 10^(-4), 10^(-4))$ for R1 and N1, $(10^(-6), 10^(-5))$ for remaining FP32 cases, and $(0.002, 0.002)$ for remaining FP16 cases; reductions also record FP64 errors. After capture we replay the first seeded input, and the zero input exercises N1 near its largest output factor.

== Measuring a pair <sec-methods-pair>

Each configuration is timed under direct CUDA submission and CUDA Graph replay. Direct submits the wrapper on every iteration; replay captures one invocation and repeatedly replays the graph. Compilation and capture precede timing. Both modes use CUDA events over a batch on the device stream; elapsed time divided by invocations gives microseconds per call, including stream gaps. This measures steady repeated execution on prepared inputs and excludes input preparation or first-use capture.

A block contains two batches, one per candidate. A trial sets a common repetition count targeting a five-millisecond batch (ten percent margin, capped), keeping calls identical across halves. Each formal configuration and mode has 30 blocks: 15 split-first and 15 fused-first, with configuration order shuffled. A pair shares a local window, and first-position bias is balanced.

The harness records configuration, mode, block, strategy and attempt; the audit matches both halves, checks repetition counts and validity, and selects the first qualified attempt. An external CUDA process or a non-idle throttle change invalidates the pair (up to two retries); a telemetry gap alone does not. The follow-up contributes 1,200 pairs across both modes.

== Estimating the benefit

Let $s_i$ and $f_i$ be per-call times in block $i$. Fusion's absolute benefit is the mean $s_i - f_i$; the relative reduction divides it by the mean $s_i$. The same microsecond saving can therefore give different relative reductions when layout changes split latency, so we report both.

Confidence intervals resample complete blocks with replacement 10,000 times @bootstrap, retaining both candidate times and recomputing the mean difference and split-time denominator; the 2.5th and 97.5th percentiles give 95% intervals. A relative interval above 2% favors fusion, below −2% favors splitting, and within $[-2%, 2%]$ supports practical equivalence; all others are indeterminate. We also evaluate labels at 1% and 5%.

For layout contrasts, blocks are paired within each layout; the analysis resamples transposed and contiguous sets independently, estimates the benefit within each, and subtracts contiguous from transposed. Differences in relative reduction are in percentage points. These intervals assume independent blocks and carry no simultaneous-coverage adjustment; counts summarize measured cases and imply no sampling distribution over applications.

== A stress family for the epilogue <sec-methods-stress>

The third family targets the epilogue rather than the intermediate. For an $M times K$
FP32 input the entry computes

$ t_m = sum_k x_(m,k), quad y_(m,k) = t_m ( x_(m,k) + sum_(s=1)^B x_(m,k+s) ) $

with row indices taken modulo $K$. Each of the $B$ shifted reads is a distinct load in
the generated kernel, and $B = 0$ reduces to multiplying every column of a row by that row's sum. The split candidate
is the same entry with $t$ materialized during lowering and the merge vetoed, as in the
main grid. We vary $M in {256, 512, 1024, 2048, 4096}$, $K in {2048, 4096, 8192,
32768}$, FP16 and FP32, and $B$ from 0 to 12, and time 30 paired blocks per
configuration (60 at $B = 3$) with the protocol of @sec-methods-pair; the label rule is
unchanged. The family was compiled and timed on one RTX 4090. Compilation evidence
records each kernel's block shape, warps, registers, spills, occupancy and the
`num_load`/`num_reduction` counters embedded in the generated code, and per-kernel self
times come from the CUDA activity profiler.
