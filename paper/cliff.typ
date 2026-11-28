= A split-favored region in a reduction epilogue <sec-cliff>

This section measures the family, explains why the fused kernel loses and bounds the claim.

The 72-case grid was not expected to produce a split-favored interval. In the archived
configurations of both RTX devices, the split and fused candidates share the producer
kernel's block shape, warp count and stage count; they differ by at most four registers.
Splitting therefore removes a launch, adds the write and read of $t$, and leaves the
producer's schedule unchanged, so the fused side keeps the better schedule and pays for
it only in the handoff. A split-favored case requires the opposite: moving the consumer
into the producer must change the producer's schedule enough to outweigh that handoff.
We built a family in which the consumer's work grows with one parameter while the edge
remains a single intermediate.

== The measured region

The family and its protocol are described in @sec-methods-stress. Of 24 configurations,
10 favored splitting, 13 favored fusion and one was inconclusive (@fig-cliff). In the 24
measured points, every sampled configuration with $M in {512, 1024}$, $K = 8192$, FP32
and $B >= 3$ favored splitting (10 of 24); the design does not cover $B = 9$ and $B = 11$
at $M = 1024$, nor $M = 512$ at $B in {3,4,6,7}$, so the region boundary is unmeasured.
The largest relative difference was at $M = 1024, B = 8$: 38.42 µs split against 66.42 µs
fused, a split-minus-fused difference of −28.00 µs (95% interval −28.14 to −27.86 µs), or
72.9% of split latency. The largest absolute difference in the split-favored direction
was at $B = 12$: −28.32 µs (95% interval −28.45 to −28.17 µs), or 56.4%. At $B = 3$ and
$B = 5$ the differences were −8.70 µs (−8.78 to −8.62) and −20.42 µs (−20.52 to −20.31).
The crossover lies between $B = 2$ (+1.89 µs, +6.4%) and $B = 3$.

#figure(
  image("figures/cliff-saving.svg", width: 100%, alt: "Two panels plot the split-minus-fused time against the number of shifted reads B. For M=1024 the relative saving falls from +25% to -73% between B=2 and B=8; the absolute panel runs from +7 to -28.3 microseconds. M=512 stays near or below zero, while M=2048 and M=4096 remain between +18% and +29%."),
  caption: [In the measured points, splitting wins for $M in {512,1024}$, FP32 and $B >= 3$; the largest relative difference is 72.9% at $M=1024, B=8$ (−28.00 µs), while the largest absolute difference in the split-favored direction is −28.32 µs at $B=12$. Markers show the split-minus-fused time and whiskers the paired 95% intervals; filled markers are split-favored labels. Color identifies $M$; all series use $K=8192$.],
) <fig-cliff>

We find the region narrow and not monotone. At $M = 256, B = 5$ fusion won by 11.38 µs
(40.2%). At $M = 2048$ it won by 48.70, 37.16 and 36.65 µs at $B = 5, 8, 12$, and at
$M = 4096, B = 5$ by 119.62 µs (28.7%); the launch-configuration cap that appears at
$B >= 8$ did not reverse the sign at either size. Changing the reduction length or the
type also removes the region: $M = 1024, K = 2048, B = 5$ favored fusion by 12.21 µs,
$K = 4096$ was inconclusive (+1.11 µs, interval −0.98 to 3.52), and FP16 at the same
$M, K, B$ favored fusion by 6.77 µs. The region is an intersection of conditions rather
than a threshold in one variable.

== Why the fused kernel loses

The fused kernel runs the epilogue in the reduction's launch configuration: one program
per row, `XBLOCK = 1`, 16 warps. The split consumer is a separate pointwise kernel with
`XBLOCK = 1024`, four warps, and a grid that grows with $M K$. Self times exclude launch
and stream gaps; summed self times are up to 2.6 µs below the paired-block wall times, so
the comparison below is a decomposition, not a time budget. At $M = 1024, K = 8192$
the profiled self times per call were 20.96 µs for the fused kernel at $B = 0$, against
8.77 µs for the split producer and 18.37 µs for its consumer. At $B = 3$ the three values
were 37.15, 9.02 and 19.09 µs; at $B = 5$, 50.26, 8.76 and 20.55 µs; and at $B = 8$,
65.18, 8.88 and 26.97 µs. The producer time is flat at about 9 µs. The split consumer
grows by roughly 1 µs per shifted read on average; the fused kernel grows by about 5 µs. Moving the
epilogue into the reduction therefore multiplies the cost of each additional read by
roughly five, and the crossover follows when the fusion saving on the handoff no longer
covers that difference (@fig-mechanism).

Two mechanisms produce the difference. First, the epilogue's parallelism is tied to the
number of rows rather than to the number of elements: the fused grid has $M$ programs,
whereas the split consumer's grid has $M K / 1024$. The same epilogue that runs as a
full-width pointwise kernel in the split candidate is serialized after the reduction in
the fused one. Second, the reduction heuristic reacts to the counters of the fused
kernel: when $M >= 1024$ and `num_load + num_reduction` reaches 10, `_reduction_configs`
marks the kernel register-intensive, lowers `MAX_RBLOCK` from 2048 to 1024 and caps the
warp count at 8 instead of 16, and a post-compile repair can halve `RBLOCK` again. The
fused kernel crossed that threshold at $B = 7$ (nine loads plus the reduction) and
selected `RBLOCK = 512` at $B = 10$, while the split producer keeps one load and one
reduction and remains at `RBLOCK = 2048`, 16 warps. A separate register path appears at
$B = 6$: the fused kernel needs 55 registers and falls to two-thirds occupancy, whereas
the split producer needs 26 and stays at full occupancy.

Neither path alone explains the sign. At $M = 4096, B = 8$ the fused kernel is capped to
`RBLOCK = 1024` and 8 warps with occupancy 0.83, yet it is 97.0 µs faster than the split.
With 4096 programs the fused epilogue's reuse of each row stays inside one kernel, and
the split consumer re-reads the input after the producer has streamed it.

#figure(
  image("figures/cliff-mechanism.svg", width: 100%, alt: "Left: profiled self time against shifted reads for the fused kernel, the split producer and the split consumer at M=1024. The fused kernel grows about five microseconds per read, the consumer about one; the producer is flat. Right: the fused RBLOCK steps from 2048 to 1024 and 512 and its warps from 16 to 8; the split producer stays at 2048 and 16 warps."),
  caption: [The fused epilogue inherits the reduction tile: its self time grows about 5 µs per shifted read, against about 1 µs for the split consumer; at $B=7$ the load counter reaches 10 and `RBLOCK`/warps are capped.],
) <fig-mechanism>

== Scope

The region was measured on one device and one family; Section 7.1 states what the
comparison does not establish. The main grid's templates never reach the load threshold,
with at most three loads per fused kernel, so its 68 fusion-favored labels per device are consistent
with the mechanism rather than contradicted by it. We do not claim that fusion generally
loses, nor that the four axes we varied are exhaustive. The claim is narrower and
measured: a legal local edge can lose by a practical margin when the consumer's load
count ties the epilogue to the reduction's tile.
