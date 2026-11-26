= Predicting the faster partition

== Information available at decision time

We evaluate a choice between two compiled candidates. At decision time on a target device, the predictor has the candidate code, device calibration and a separate timing of the split candidate. Evaluation then compares the prediction with paired split and fused measurements. The per-call scores exclude the time needed to compile the candidates, calibrate the device and collect the split timings.

Let $k_s$ and $k_f$ be split/fused launch counts and $L$ the calibrated time per additional minimal kernel under the submission mode. The launch feature is $x_1 = (k_s - k_f)L$. Calibration varies sequence length over 1, 8, 32 and 128 kernels, takes the median over 20 blocks, and fits a line with an intercept; direct and replay give separate $L$ because submission cost differs even for identical kernels.

For an intermediate of $n$ elements and $w$ bytes, split writes $n w$ bytes and the consumer reads $n w$. A device copy of $s$ bytes adds $2s$ logical bytes; we measure throughput under direct submission and take the median of per-size medians at the three largest buffers above L2. The bandwidth $W$ gives $x_2=(2 n w)/W$ for both modes. A reduction may read millions of inputs but only row results cross the edge, so $x_2$ can be small. The estimate assumes transfer cost proportional to logical volume; cache residency and rereads violate this.

The optional third feature estimates static occupancy. Compiler artifacts give the resident warp fraction for kernels on each side; the regional summary takes the minimum per candidate. With summaries $o_s$ and $o_f$, $x_3=T_s(o_s-o_f)$, where $T_s$ is measured split latency, so $x_3$ has time units. Lower occupancy after fusion gives positive $x_3$. Occupancy here is launch-configuration capacity, not observed active warps; a small grid can leave it unused. The feature is unavailable when either candidate lacks it.

== Fitting and holding out templates

Both models fit the observed saving in microseconds by ordinary least squares without an intercept: the two-term prediction is $a x_1+b x_2$, and the extension adds $c x_3$. Each selects fusion when the prediction is positive. We also compare Inductor's verified default, both constant choices, and a byte threshold that selects fusion above a cutoff. The threshold search uses adjacent byte-count midpoints and constant thresholds, minimizes training regret, and breaks ties toward the larger threshold.

Within the original RTX 3090 grid, four folds hold out P1, P2, R1 and R2 in turn; all shapes and both dtypes of the held-out expression enter the same test fold, and coefficients and threshold are re-estimated. Here $T_s$ uses the mean split time from the paired blocks that also score the case, so the resource feature shares observations with evaluation. Across folds, the two-term model, default and constant fusion each chose the winner in all 68 clear direct cases (mean regret 0); the byte threshold chose the winner in 64 of 68. Under replay the two-term model chose the winner in all 25 clear cases.

== Freezing and scoring target predictions

After template-level evaluation, source models were refitted on the full RTX 3090 grid by mode. The two-term launch/transfer coefficients were 0.386/0.983 under direct and 0.194/0.998 under replay; all 72 source configurations had complete resource features. The third coefficient was −0.150 direct and +0.0254 replay; the sign change cautions against a device-independent occupancy penalty. Coefficients stayed fixed during target validation. Both two-term coefficients are positive in each mode, so any candidate that removes a launch and moves fewer logical bytes is predicted to favor fusion; this says little about the magnitude.

On each target the harness recomputed features from target compilation and calibration, measured split execution separately, and wrote timestamped predictions before formal paired timing, with digests for the source model, candidates, calibration and split-only records. The same procedure covers the new RTX 3090 workload; four of its cases are repeat controls.

Accuracy uses only clear labels with a prediction; regret measures time lost against the faster measured candidate over every prediction, so inconclusive/equivalent cases can contribute. Coverage counts predictions before label selection. The resource model's two-term comparator is restricted to complete-resource cases, giving both models the same error opportunities.
