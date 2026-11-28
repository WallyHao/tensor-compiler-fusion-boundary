= Experimental results

This section answers four questions. (1) How much of the direct-submission saving survives graph replay? (2) Does transposing the reduction input change the sign or only the magnitude? (3) Do labels transfer across devices, and can a calibrated model improve on the default? (4) How sensitive are the labels to the practical margin?

#figure(
  placement: none,
  table(
    columns: 4,
    align: (left, center, center, center),
    table.header([Cohort], [Configurations], [Direct (F/E/I/S)], [Replay (F/E/I/S)]),
    [RTX 3090 base], [72], [68 / 3 / 1 / 0], [25 / 33 / 14 / 0],
    [RTX 4090 base], [72], [68 / 4 / 0 / 0], [27 / 7 / 38 / 0],
    [RTX 3090 subset], [12], [10 / 2 / 0 / 0], [2 / 8 / 2 / 0],
    [RTX 4090 subset], [12], [10 / 2 / 0 / 0], [6 / 2 / 4 / 0],
    [H800 subset], [12], [12 / 0 / 0 / 0], [7 / 0 / 5 / 0],
    [RTX 3090 follow-up], [20], [20 / 0 / 0 / 0], [13 / 3 / 4 / 0],
  ),
  caption: [Fusion labels at the 2% margin. Each mode cell counts fusion-favored / equivalent / inconclusive / split-favored configurations. The three subset rows use the 12 cases measured on all three devices.],
) <tab-labels>

== How much of the fusion benefit survives graph replay?

#figure(
  placement: none,
  image("figures/followup-saving.svg", width: 100%, alt: "One row per RTX 3090 reduction configuration. Direct-submission circles lie to the right of graph-replay diamonds in every row. The controls, long reductions, transposed inputs and inverse-RMS fragments are grouped separately."),
  caption: [Absolute fusion saving across the 20 RTX 3090 follow-up configurations. Markers show the split-minus-fused time and whiskers show per-configuration paired 95% intervals. Gray connectors join the two estimates for the same configuration. Each row gives the template and the shape $M times K$.],
) <fig-followup>


All 20 reduction configurations in the RTX 3090 follow-up favored fusion under direct submission. Fusing the selected edge saved between 1.52 and 3.18 µs per invocation, or 3.7% to 37.6% of split latency. Under CUDA Graph replay, 13 configurations still favored fusion, three met the equivalence criterion, and four had inconclusive intervals. The estimated absolute saving was smaller under replay in every configuration (@fig-followup). Its median across configurations fell from 2.36 µs under direct submission to 0.68 µs under replay. Each configuration contributes one estimate to these medians, including each of the four contiguous controls measured alongside the new cases.

Fusion removed one computational kernel launch in each follow-up configuration. Twelve configurations went from two launches to one. The other eight went from three to two, retaining a separate stage of the reduction. A calibration with minimal kernels in the formal run measured 6.54 µs per additional kernel under direct submission and 0.91 µs under replay. The smaller slope under replay is consistent with the smaller savings measured for the reduction fragments. Its magnitude cannot be carried over directly to these fragments. Even the largest direct saving was below 6.54 µs, and moving the epilogue into the reduction kernel changed both its instructions and its accesses to intermediate data.

Increasing the reduction length illustrates why a fixed launch count can accompany widely different relative savings. Consider the contiguous FP32 cases with 256 rows. At a reduction length of 8,192, fusion saved 11.93% for the row sum followed by ReLU (R1) and 11.52% for the row maximum followed by an affine transformation (R2) under direct submission. At length 32,768 the reductions were 3.86% and 3.66%, respectively. The absolute saving for R1 changed little, from 1.55 to 1.62 µs, while its split latency rose from 13.02 to 41.95 µs. Both templates still favored fusion at the longer length. Under replay the estimates also declined between the endpoints, from 7.72% to 3.05% for R1 and from 7.28% to 2.74% for R2. Increasing the number of inputs per row changes how much of the split execution time a saving at the row output can recover.

The inverse-RMS fragment (N1) extends the comparison to a reduction with an input transformation as well as an output transformation. For 256 rows of length 8,192, the split code writes the row mean of the squared inputs to an intermediate array. A second kernel reads each mean, adds the numerical offset, and evaluates its reciprocal square root. The fused code performs these operations in one kernel. Direct submission took 13.067 µs for split and 11.479 µs for fused, a saving of 12.15% with a paired 95% interval of 12.13%–12.17%. The saving under replay for this configuration was 7.73%. Across the four inverse-RMS configurations, direct savings ranged from 12.2% to 37.6%. Under replay, both 256-row cases favored fusion, while the estimates at 16 rows were 0.22% and 1.25%. These timings end when the reciprocal square roots become available. A complete normalization layer would also use those factors to scale the original inputs.

== What changes when the reduction input is transposed?

#figure(
  placement: none,
  image("figures/layout-effects.svg", width: 100%, alt: "Under direct submission, all four estimates lie left of zero in the relative panel and near or right of zero in the absolute panel. Under replay, the absolute change is positive for both 16-row reductions and negative for both 256-row reductions."),
  caption: [Change in fusion saving when switching from contiguous to transposed FP32 input on the RTX 3090, with $K = 8192$. Each marker is the transposed estimate minus its matched contiguous estimate. Whiskers are 95% bootstrap intervals for the difference, resampling complete split–fused blocks separately within each layout.],
) <fig-layout>


Four matched comparisons isolate the change from contiguous to transposed input at a fixed shape, data type, and reduction operator. Under direct submission, transposition reduced the relative saving in all four comparisons. The decrease ranged from 0.48 to 12.60 percentage points. In absolute time, the point estimate of the saving increased in each comparison. The two panels of @fig-layout show the divergence, which follows from the increase in split latency after transposition.

For R1 with 16 rows of length 8,192, the contiguous split path took 8.281 µs and fusion saved 3.092 µs. With the transposed input, the split path took 12.514 µs and fusion saved 3.096 µs. The nearly identical absolute savings represented 37.34% and 24.74% of their respective split times. The estimated change in the absolute saving was only 0.004 µs, with a 95% interval from −0.143 to 0.133 µs. R2 at the same shape showed a similar pattern. Its absolute saving rose by an estimated 0.064 µs, with an interval that also included zero. The two reductions thus supply examples in which a change of more than 12 percentage points accompanies little evidence for a change in microseconds saved.

At 256 rows, the direct-submission differences were smaller on the percentage scale and better resolved on the time scale. Transposition increased the absolute saving by 0.098 µs for R1 and 0.185 µs for R2. The corresponding intervals were 0.070–0.114 and 0.166–0.198 µs. Yet the relative saving declined by 1.04 and 0.48 percentage points, respectively. In both cases the split path became slower with transposed input, as it had at 16 rows.

Replay changed the relationship with row count. At 16 rows, transposition increased the absolute saving by 0.176 µs for R1 and 0.182 µs for R2. At 256 rows, it decreased the saving by 0.220 µs for R1 and 0.135 µs for R2. All four individual intervals for these replay contrasts excluded zero. For example, contiguous R1 at 16 rows saved just 0.047 µs under replay, compared with 0.223 µs for the transposed input. The respective classifications were equivalent and fusion-favored. All four transposed candidates retained two computational kernels after fusion, whereas their contiguous controls used one. The layout contrast therefore includes any scheduling changes induced by the new stride. Attributing the latency change specifically to memory transactions would require hardware counters.

== Which decisions transfer across devices?

#figure(
  placement: none,
  image("figures/device-labels.svg", width: 100%, alt: "Two three-by-three matrices count paired RTX 3090 and RTX 4090 labels. Direct submission places 68 cases in the shared fusion cell and three in the shared equivalence cell. Replay places 13 in the shared fusion cell and distributes cases across all nine cells."),
  caption: [Joint fusion labels for the same 72 configurations on the two RTX devices. Each matrix cell counts configurations at the intersection of the RTX 3090 row and RTX 4090 column. Neither device had a split-favored interval at the 2% practical margin.],
) <fig-devices>


@fig-devices cross-tabulates the two device labels for every configuration in the original 72-case grid. Under direct submission, 68 configurations favored fusion on both devices. Three further cases met the equivalence criterion on both. The remaining configuration was inconclusive on the RTX 3090 and equivalent on the RTX 4090. Under replay, 25 configurations favored fusion on the RTX 3090 and 27 favored it on the RTX 4090, with only 13 configurations in both sets. Of the 12 configurations that favored fusion only on the RTX 3090, ten were inconclusive and two were equivalent on the RTX 4090. Of the 14 that favored fusion only on the RTX 4090, three were inconclusive and 11 were equivalent on the RTX 3090. A change in classification across devices thus need not imply a change in the sign of the saving. On the RTX 3090, the other 47 replay results comprised 33 equivalent and 14 inconclusive cases. On the RTX 4090, seven were equivalent and 38 were inconclusive. An inconclusive interval leaves the practical decision unresolved even when the estimated saving is positive.

We compare the H800 PCIe on a preselected subset of 12 cases. Restricting all three devices to exactly those cases gives 10 clear direct-submission wins for fusion on each RTX device and 12 on the H800. Under replay the counts were two, six, and seven for the RTX 3090, RTX 4090, and H800, respectively. The selection rule precludes estimating the frequency of replay wins across the full H800 grid from the observed seven of 12. The comparisons also span different hosts, drivers, and software patch environments, so a change in label cannot be attributed to GPU architecture alone.

== Do the predictive models improve on the default?

On target devices and in the RTX 3090 follow-up, the two-term model matched Inductor's default on clear-case accuracy and mean regret. On the RTX 4090 replay grid, the two-term rule, the default and always-fuse each selected all 27 clear winners, yet each incurred mean regret 0.00583 µs over all 72 configurations. Accuracy counts resolved labels; regret compares the selected time with the faster measured time, so unresolved cases can add regret even when every clear case is correct.

Adding the resource term reduced RTX 4090 replay accuracy to 25 of 27: both errors were FP16 contiguous R2 with 16 rows, where selecting split cost 0.622 µs at length 512 and 0.324 µs at 2,048; mean regret rose to 0.01861 µs. In the RTX 3090 follow-up, the two-term rule predicted all 20 cases per mode, while the three-term rule could be scored on only 12; all eight omissions were cases where fusion reduced kernel count from three to two (the four transposed inputs and four 16-row long reductions). Its eight correct replay decisions therefore say nothing about those inputs.

Across all cohorts, every clear label favored fusion. The tests can identify a policy that sometimes sacrifices time by splitting a fusion-favored edge; they cannot estimate its ability to recognize a genuinely split-favored edge. Matching the default establishes a tie on the reported measures and leaves that capability untested.

== Sensitivity near the decision threshold

The practical margin affects the reported frequency of clear wins. In the RTX 3090 follow-up, reducing the margin from 2% to 1% yielded 20 direct and 15 replay wins for fusion. Increasing it to 5% yielded 18 direct and seven replay wins. Neither alternative produced a split-favored case. The direction of the clear decisions thus survived both changes, while the replay count varied by more than a factor of two between the narrowest and widest margins.

A second acquisition revisited four cases selected for replay estimates close to zero, with a different seed for collection order. Across the eight case–mode combinations, all point estimates retained their sign, and the largest absolute change in relative saving was 0.80 percentage points. Seven classifications remained the same. Contiguous FP32 R2 with 16 rows of length 8,192 moved from equivalent to inconclusive under replay as its estimated saving rose from 0.56% to 1.05%. The 95% interval widened from 0.37%–0.74% to 0.14%–2.14%. The repeat therefore crossed the 2% upper equivalence limit even though its point estimate remained inside the band. Because these four cases were selected after the main acquisition, the repeat describes stability near the observed decision margin rather than an overall reproducibility rate.

== A pointwise counterpoint to the reductions

#figure(
  image("figures/pointwise-scaling.svg", width: 100%, alt: "Two panels show the six FP32 vector lengths for the affine–ReLU and tanh–affine chains. At small lengths the replay markers are near zero. Both modes approach a 50% saving at the largest length for both chains."),
  placement: none,
  caption: [Relative fusion saving for contiguous FP32 pointwise chains on the RTX 3090. Markers show estimates and whiskers show per-configuration paired 95% intervals; solid and dashed segments follow direct submission and CUDA Graph replay. The shaded band marks the ±2% practical margin. The length axis is logarithmic (base 2).],
) <fig-pointwise>

The reduction follow-up eliminates an intermediate with one scalar per row. In the pointwise grid the intermediate grows with the full input. @fig-pointwise shows how the benefit varies across the sampled lengths. On the RTX 3090, consider P1 with a contiguous FP32 vector of length 1,024. The split and fused paths took 8.146 and 5.096 µs per call under direct submission. Under replay they took 7.958 and 7.880 µs. The replay estimate saved 0.98%, but its interval extended slightly beyond 2%. A decision rule based on the point estimate alone would call the variants equivalent. The interval rule leaves that decision open.

Now keep the expression, type, layout and device fixed, and increase the vector length to 16,777,216. Under direct submission the split time was 321.424 µs and the fused time was 160.886 µs. Under replay the times were 324.411 and 164.274 µs. In absolute terms the saving was about 160 µs in either mode. The logical intermediate at this size occupies 64 MiB. Writing and reading it accounts for 128 MiB of logical accesses across the cut. Eliminating a kernel that moves an entire vector can therefore retain a substantial advantage even when replay makes submission cheaper. The timing and the generated access pattern support this explanation. Establishing how many bytes reached DRAM would require additional evidence.

The intermediate lengths reveal a distinction the two endpoints alone obscure. At 524,288 FP32 elements, P1 under replay saved an estimated 1.40% (95% interval −0.88% to 3.23%). At the same length and precision, P2, whose producer computes a hyperbolic tangent, saved 10.09% (9.92% to 10.25%). Both splits pass a 2 MiB vector to the consumer. A rule that sees only the logical intermediate size must assign both cases the same decision, although the measured evidence for fusion differs. By 4,194,304 elements the two savings were close, 47.51% and 47.02%. These observations bound the comparison to the sampled lengths. They do not locate a length at which splitting wins.

== How much does a wrong decision cost?

For a repeatedly called fragment, a wrong choice is paid on every invocation; counting errors treats a fraction of a microsecond and a hundred microseconds alike. On the RTX 4090 replay grid, always selecting split incurred mean regret 4.570 µs and maximum 127.732 µs, while always selecting fused incurred 0.00583 µs and 0.221 µs. The latter maximum is computed from measured mean latencies before imposing a margin, so it identifies a case to examine rather than a significant loss.

#figure(
  placement: none,
  table(
    columns: 7,
    align: (left, center, left, center, center, center, center),
    table.header([Cohort], [Mode], [Policy], [Coverage], [Correct], [Mean regret (µs)], [Max regret (µs)]),
    [RTX 4090], [replay], [Default], [1.00], [27/27], [0.00583], [0.221],
    [RTX 4090], [replay], [Two-term], [1.00], [27/27], [0.00583], [0.221],
    [RTX 4090], [replay], [Three-term], [1.00], [25/27], [0.01861], [0.622],
    [RTX 4090], [replay], [Always split], [1.00], [0/27], [4.570], [127.732],
    [3090 follow-up], [direct], [Two-term], [1.00], [20/20], [0], [0],
    [3090 follow-up], [direct], [Three-term], [0.60], [12/12], [0], [0],
    [3090 follow-up], [replay], [Two-term], [1.00], [13/13], [0], [0],
    [3090 follow-up], [replay], [Three-term], [0.60], [8/8], [0], [0],
  ),
  caption: [Decision metrics at the 2% margin. Coverage counts predictions before label selection; correct counts clear labels; regret compares the selected time with the faster measured time over every prediction.],
) <tab-decisions>

Feature availability changes the question. In the RTX 3090 follow-up under replay, the two-term rule had zero observed regret on the 12 cases with complete resource metadata and on all 20 cases; the three-term rule had zero on the 12 it could score. A policy deployed for every call must still specify a fallback for the other eight, and crediting a successful fallback would evaluate a different policy.

Finally, each configuration has weight one regardless of invocation frequency. Turning per-case regret into end-to-end saving requires call frequencies, the surrounding schedule and prediction overhead; a predictor that ties the default in steady-state latency has not yet recovered the cost of compiling alternatives and timing the split path.
