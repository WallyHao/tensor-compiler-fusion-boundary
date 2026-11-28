# 行文风格与结构对照：手稿 vs. DNNFusion / AStitch / Korch

对照对象：`paper/report.typ` 及其 include 的五个章节文件；三篇 ACM 论文的全文抽取（pypdf）：

- **DNNFusion**，PLDI 2021，DOI 10.1145/3453483.3454083；
- **AStitch**，ASPLOS 2022，DOI 10.1145/3503222.3507723；
- **Korch**，ASPLOS 2024，DOI 10.1145/3620666.3651383。

选择理由：三篇都处理“融合/编排决策”，但体裁分别是融合方案搜索（DNNFusion）、融合空间与代码生成（AStitch）、把编排形式化为优化问题（Korch）。手稿同时做测量、预测和边界，正好落在三者之间。

## 0. 结论摘要

1. **结构上，手稿是“测量论文”，三篇是“技术论文”。** 三篇的骨架是：背景/挑战 → 提出系统 → 设计/实现 → 评估 → 相关工作 → 结论；手稿是：问题 → 实验设计 → 预测 → 结果 → 新区域 → 追溯 → 解释 → 结论。这不是排版差异，而是贡献类型差异。
2. **最大的四个结构风险**：引言没有显式的 gap → approach → contributions 段；相关工作被压进 §2，且没有与 DNNFusion/AStitch/Korch 这类“融合决策”工作划界；§7 追溯节夹在结果与解释之间，像工程日志；§8 解释占 14.2%，比引言还长，且与 §6.3 有重复。
3. **风格上，手稿明显更克制**：`we/our` 密度 2.6/k（三篇 7.9–12.1/k），hedge 3.6/k（三篇 0.95–1.31/k），平均句长 18.3 词（三篇 20.4–25.9），没有 `novel`、没有 contributions、没有路标短语，正文 0 张表。克制与测量体裁一致，但“断言后置、主语物化、缺少路标”降低了可扫读性。
4. **最该保留的**：置信区间、负结果、不写“首次/最优”、把编译器默认行为当作基线。这些恰恰是三篇技术论文的弱项。

## 1. 结构对照

### 1.1 章节顺序

| 手稿 | DNNFusion | AStitch | Korch |
|---|---|---|---|
| 1 Introduction | 1 Introduction | 1 Introduction | 1 Introduction |
| 2 Where the fusion decision occurs（背景+相关工作） | 2 Blessing and Curse of Deep Layers（动机研究） | 2 Background and Current Challenges | 2 Overview |
| 3 Experimental design | 3 Classification of Operators and Fusion Opportunity | 3 Key Design Methodology | 3 Operator Fission |
| 4 Predicting the faster partition | 4 DNNFusion's Design | 4 Compiler Design and Optimizations | 4 Kernel Orchestration |
| 5 Experimental results | 5 Evaluation | 5 Implementation | 5 Implementation |
| 6 A split-favored region | 6 Related Work | 6 Evaluation | 6 Evaluation |
| 7 Tracing a result to the executable | 7 Conclusions and Future Work | 7 Related Work | 7 Related Work |
| 8 Interpreting the measured boundary | | 8 Conclusion | 8 Future Work |
| 9 Conclusion | | A Artifact Appendix | 9 Conclusion |
| | | | A Artifact Appendix |

三篇的共同点：**相关工作在评估之后、结论之前**；评估是单独一节；设计/实现占正文最大份额；结论很短（1.5–2.9%），未来工作要么并入结论（DNNFusion），要么单列（Korch），要么省略（AStitch）。

手稿的不同点：相关工作前移并压缩；没有“设计/实现”节（§3 是实验设计，§4 是预测模型）；结果拆成 §5、§6、§7 三段，随后又有 §8 解释；结论 4.3%，没有未来工作段。

### 1.2 篇幅分配（正文词数占比，近似）

| 部分 | 手稿 | DNNFusion | AStitch | Korch |
|---|---:|---:|---:|---:|
| 引言 | 7.7% | 12.3% | 11.4% | 12.8% |
| 背景 + 相关工作 | 6.0%（§2） | 2.6% + 5.5% | 20.7% + 5.7% | 10.4% |
| 方法/设计/实现 | 15.8%（§3）+ 11.2%（§4） | 14.0% + 35.0% | 15.9% + 21.0% + 2.2% | 4.3% + 15.5% + 23.1% + 7.7% |
| 结果/评估 | 24.5%（§5）+ 11.7%（§6）+ 4.6%（§7） | 27.2% | 21.1% | 21.3% |
| 解释/讨论 | 14.2%（§8） | — | — | — |
| 结论（+未来工作） | 4.3% | 2.9% | 1.5% | 2.4% + 2.2% |

两个直接读数：

- **结果+解释合计 55.0%**，三篇的评估是 21.1–27.2%。手稿用接近一半正文讲结果与含义，这与“测量论文”的定位一致，但也意味着读者在读到 §5 之前已经花了 27% 的篇幅在设计与预测上，真正的结果出现得晚。
- **引言只有 7.7%**，三篇是 11.4–12.8%。引言短不是问题，问题是它承担了“动机 + 两个问题 + 设备网格 + 默认全融合的 caveat”，却没有承担“与已有工作划界”和“贡献清单”。

### 1.3 差异逐条

**（1）引言的功能不同。** 三篇在引言里完成四件事：现状 → 缺口 → 提出命名系统 → 贡献 bullets。DNNFusion 用 “In this paper, we observe that the fusion patterns … are too restricted”（l. 119–121）直接立 gap；AStitch 用 “we propose AStitch … opens a new multi-dimensional optimization space”（l. 38–41）；Korch 用 “This task has been mostly ignored by prior approaches”（l. 156–157）。三篇都在引言末尾给出 4–5 条 bullet（DNNFusion l. 197–222；AStitch l. 221–247；Korch l. 220–229）。

手稿的引言以一个归约例子开场，随后提出两个问题、设备网格和“默认全融合”的 caveat，没有针对已有工作的 gap 句、没有贡献段、没有 roadmap，也没有引用任何融合决策系统。它把“与已有工作的关系”留给了 §2 的一段。对 ACM 审稿人，这会在前两页留下“这篇论文要主张什么”的空缺。

**（2）相关工作的位置与密度。** 三篇都把相关工作放在评估之后，作为“划界”而不是“背景”：DNNFusion §6 用 4 段、AStitch §7 用 5 段、Korch §7 用 5 段，按主题分组，每段以 “Comparing with …” 或 “A key difference …” 收尾。手稿的 §2 只有一段真正讲系统（TVM Relax、XLA priority fusion、FusionStitching、megakernel），其余三段讲 pass 位置、逻辑字节和提交路径。它更像“背景”，不像“相关工作”。

这里不建议照搬“相关工作后置”：测量论文的边界主张必须相对已有阈值来读，相关工作前移是合理的。问题是**密度**：手稿没有引用 DNNFusion、AStitch、Korch、Neptune、Hidet、HFuse、RedFuser 这些直接相邻的工作，也没有一句“我们与它们的差别是……”。

**（3）没有“设计/实现”节。** 三篇的核心都是提出一个系统：DNNFusion 的设计占 35%，AStitch 的设计+编译器占 36.9%，Korch 的 fission+orchestration+实现占 46.3%。手稿没有系统可提；它的“设计”是协议（配对 block、CI、等价边界）和预测模型。这本身没问题，但需要把“协议也是贡献”说清楚，否则审稿人会按技术论文的模板找“系统贡献”而找不到。

**（4）评估的组织不同。** 三篇的评估都是“一节、多小节”：setup → 端到端 → 分解/案例 → 开销。DNNFusion 在评估开头列 4 个 numbered objectives（l. 1135–1147）；AStitch 用 “Overall Results / Performance Breakdown / Ablation / Counter Analysis” 命名小节；Korch 用 “end-to-end → adaptation → case study → tuning time”。手稿的结果跨 §5（replay、transpose、跨设备、模型、阈值）、§6（新区域）、§7（追溯）、§8（解释），没有一句“本节回答哪几个问题”。

**（5）结论与未来工作。** 三篇结论 1.5–2.9%，只做两件事：重述主张（部分给未来工作）。手稿结论 4.3%，重述了三组结果，最后一句 “How often such epilogue coupling occurs in model graphs remains open.” 是未来工作的引子，但没有展开。

**（6）§7 是手稿独有的结构。** “Tracing a result to the executable” 用 304 词（4.6%）讲归档记录、loader 失败和重建。三篇没有对应的正文节；AStitch 和 Korch 把这类内容放进 Artifact Appendix。它在正文中的位置（结果之后、解释之前）会打断“结果 → 含义”的阅读路径。

## 2. 行文风格对照

### 2.1 量化指标（近似）

| 指标 | 手稿 | DNNFusion | AStitch | Korch |
|---|---:|---:|---:|---:|
| 正文词数 | 6,638 | 10,165 | 8,883 | 7,601 |
| 摘要词数 | 210 | ~300 | ~250 | ~182 |
| 平均句长（词） | 18.3 | 25.9 | 20.4 | 22.7 |
| 句长中位数 | 17 | 21 | 18 | 19 |
| `we/our/us` 每千词 | 2.6 | 8.4 | 12.1 | 7.9 |
| hedge 每千词 | 3.6 | 1.3 | 1.1 | 0.95 |
| 数字每千词 | 59 | 84 | 73 | 56 |
| `novel` 次数 | 0 | 4 | 2 | 0 |
| `contribution(s)` 次数 | 0 | 2 | 1 | 2 |
| 显式路标（in this section / we propose 等） | 0 | 4 | 16 | 6 |
| 图 / 表 | 7 / 0 | 10 / 6 | 16 / 6 | 13 / 2 |

说明：手稿在 Typst 源码上统计，三篇在 pypdf 抽取文本上统计；句子切分与词数均为近似值，适合看相对差异，不宜当作精确测量。hedge 词表包括 may/might/could/would/suggest/appear/likely/possibly/perhaps/indicate/consistent。

### 2.2 人称与断言强度

三篇用 “we” 做事、用系统名做主语做断言：AStitch 的 “AStitch outperforms all other techniques we compare”；Korch 的 “Korch outperforms existing frameworks by up to 1.7×”；DNNFusion 的 “DNNFusion achieves up to 8.8× more loop fusions”。手稿很少用“系统名 + 强断言”的句型；“we”更多用于“we measure / we report”，较少用于主张：

- “The region is narrow and not monotone.”（§6.1）
- “We do not claim that fusion generally loses.”（§6.3）
- “A comparison between two settings of a compiler flag thus needs an independent check.”（§1）

前两句是好的：第一句是 verdict，第二句是范围限定。但整篇的默认主语是 “the measurement / the region / the interval / the comparison”，而不是 “we”。这使文章显得客观，也显得被动；读者很难感到“作者发现了什么”。

hedge 密度是 3.6/k，三篇约 1/k。手稿的 hedge 多数是认识论上的必要限定（区间、样本、设备），不是空泛的 “may”。问题在于**双重限定**：一个句子里同时出现 “suggests / appears / consistent with / cannot”，读者会失去锚点。建议保留区间和 “we do not claim”，删掉没有信息量的 “appears consistent with”。

### 2.3 句子与段落

手稿平均句长 18.3 词，是三篇里最短的；这本身是优点。问题在段落开头：三篇的结果段通常第一句就是断言（“Figure 6 shows …”, “AStitch outperforms …”, “On A100, Korch …”），手稿常用“对象 + 状态”开头（“The region is narrow …”, “The replay estimate saved 0.98% …”, “The comparison with the H800 …”）。这不算错，但连续多段如此，会让读者每一段都要自己找重点。

三篇的段落多为 5–12 行、一段一个 claim，段内先结论后证据；手稿的段落更长，常在一段里完成“观察 → 数字 → 解释 → 限定”。在 §5、§6 这种数字密集的地方，建议把“限定”移到段末或 §8.2，把第一句留给 verdict。

### 2.4 证据的呈现单位

三篇的评估以“相对加速比 + 基线 + 设备”为单位，几乎不给方差：DNNFusion 明确写 “All experiments were run 100 times but as the variance was very small, we only report averages.”（l. 1200–1201）；AStitch 10 次取平均；Korch 单次测量报比值。手稿以“配对差值 + 95% 区间 + 等价边界”为单位，这是方法上的优势，不应为了“像 ACM”而退回去。

但呈现方式可以借三篇：它们把大量数字放进表和图，正文只留“范围 + 机制”。手稿正文数字密度 59/k，却没有一张表；读者要在段落里追 µs 和百分比。建议至少加两张表：

1. 各 cohort × mode 的标签计数（fusion / split / equivalent / inconclusive）；
2. 决策指标（accuracy、mean/ max regret、coverage）按设备与模型。

### 2.5 路标与可扫读性

三篇的路标密度是 4–16 次，手稿是 0。三篇的常见手法：

- 评估开头列 objectives / “We first evaluate … Then …”；
- 小节内用 run-in bold label（“Workloads.”、“Baselines.”、“Redundant Computing.”）；
- 结果段用 “Figure X shows …” 开头；
- 相关工作每段以 “A key difference …” 收尾。

手稿的章节标题本身是清楚的，但正文没有 “In this section …”“We first …”“Figure X shows …” 这类引导。对习惯跳读的审稿人，这会增加定位成本。

## 3. 哪些差异是体裁决定的

以下差异**不应**为了“像三篇”而改：

| 差异 | 原因 |
|---|---|
| 没有命名系统、没有 “we propose” | 手稿的贡献是测量协议、边界和预测评估，不是新系统；硬造一个系统名会过度承诺 |
| 没有 “novel / first / optimal” | 测量论文的主张受采样与设备限制，不能写成优先权或最优性 |
| 保留置信区间与负结果 | 三篇的“平均值 + 最好倍数”正是手稿要替代的做法 |
| 相关工作前移 | 测量论文的边界主张要相对已有阈值来读；后置反而难读 |
| 句子短、限定多 | 与配对测量、区间解释的体裁一致 |

以下差异**是风险**，与体裁无关：

| 风险 | 证据 |
|---|---|
| 引言没有贡献段/roadmap | 三篇都有 4–5 条 bullet；手稿 0 条，审稿人前两页找不到“本文主张什么” |
| 相关工作没有与最近系统划界 | §2 只引 TVM/XLA/FusionStitching/megakernel，未提 DNNFusion/AStitch/Korch/Neptune 等 |
| 结果节没有评估问题 | 三篇都有 objectives 或按 claim 命名的小节；手稿 §5 直接进入 “All 20 …” |
| §7 像工程日志 | 304 词讲归档行号与 loader 失败，位置在结果与解释之间 |
| §8 过长且有重复 | 14.2% 正文；§8.1/§8.4 是结果，§8.3 与 §6.3 重复 |
| 0 张表 | 三篇 2–6 张表；手稿的数字全在段落里 |
| “we” 密度过低 | 2.6/k vs 7.9–12.1/k；断言的主语被物化 |

## 4. 修改建议（按优先级）

### P0：不改结构也能做，直接影响审稿人判断

1. **引言末尾加贡献段（3 条，不必用 bullet 以外的格式）。** 位置：设备段之后、§2 之前。措辞用 “we measure / we test / we provide”，不要用 “we propose”。示例（未写入正文）：

   > This paper makes three contributions. First, it defines a paired-block protocol that verifies the edge at the kernel level before timing. Second, it measures where the fusion saving survives replay and layout changes. Third, it evaluates whether a device-calibrated predictor can beat the compiler default, and reports where it cannot.

2. **引言最后一段加一句 roadmap。** 例：

   > Section 2 locates the decision, Section 3 defines the protocol, Sections 5 and 6 report the grid and the stress family, and Section 8 discusses the limits.

3. **摘要补一句 gap 和一句贡献。** 现在摘要 210 词、10 句，第一句是通用陈述，随后直接 “We measure”。建议在第二句前加：“Prior fusion heuristics assume that moving the consumer into the producer leaves the producer's schedule intact; this paper tests when that assumption fails.” 并把贡献句放在结果句之前。把 H800 子集的细节移到正文，摘要控制在 190 词左右。

4. **§5 开头列 3–4 个评估问题/目标**（DNNFusion 的 numbered objectives 是最接近的模板），每个对应一个现有小节。例：“(1) Does the saving survive replay? (2) Does layout change its sign? (3) Do labels transfer across devices? (4) Does a calibrated model beat the default?” 不需要改小节标题。

5. **加两张表**：cohort × mode 的标签计数；决策指标（accuracy/regret/coverage）。模板支持表格样式（`figure.where(kind: table)` 已配置），成本低。

### P1：结构性调整，收益大但改动中等

6. **扩写 §2 的相关工作**，加 1–2 段，至少覆盖 DNNFusion、AStitch、Korch、Neptune、Hidet、HFuse、RedFuser，每段以 delta 句收尾。例：

   > DNNFusion and Korch choose a fusion plan from operator classes or an ILP; we hold the plan fixed and measure the sign of one edge. AStitch expands the fusion space for memory-intensive kernels; our stress family shows that the expanded space contains a region where the fused kernel loses.

   这同时补上“最近工作”的空白，不需要把相关工作移到文末。

7. **把 §7 移入附录**（模板 `appendix` 已支持），正文只保留 §3.2 的一段 loader 校验，或 §5 的一句“归档行号与重建脚本见附录 A”。这样结果与解释之间的阅读路径不再被工程细节打断，审计链仍然保留。

8. **重组 §8**：把 §8.1（pointwise 反例）与 §8.4（regret）移入 §5 对应小节，它们本质是结果；把 §8.3 合并进 §6.3；保留 §8.2 并改名为 “Threats to validity”，按 construct / statistical / external 三类写。这样 §8 从 14.2% 降到约 5–6%，结论更短，结构更接近 “results → discussion → conclusion”。

9. **结论加一句未来工作**：第二个压力家族、硬件计数器、更多设备。现在最后一句 “remains open” 已经点到，但可以明确写成 future work。

### P2：风格润色，不改论证

10. **把 “we” 密度提到 5–6/k**：把 “The region is narrow …” 改成 “We find the region narrow …”；把 “The comparison with the H800 …” 改成 “We compare the H800 …”。不要全篇改；每节开头和每个结果段的第一句改即可。

11. **删双重 hedge**：保留 “95% interval”“cannot”“do not claim”，删掉没有信息量的 “appears consistent with”“may suggest”。把限定集中到 §8.2。

12. **结果段 verdict-first**：每段第一句写结论，第二句给数字，第三句给限定。例（§5.1 开头）：

    > Replay preserves the sign of the saving but not its magnitude: 13 of 20 cases remain fusion-favored, and the median falls from 2.36 to 0.68 µs.

13. **加路标**：每节第一段用 “In this section …” 或 “We first …”；图表首次出现时用 “Figure X shows …”。

## 5. 三篇各自的可借鉴与不可照搬

| 论文 | 可借鉴 | 不可照搬 |
|---|---|---|
| DNNFusion | 评估开头的 numbered objectives；范围+机制的句子；run-in bold 标签 | “for the first time” 的优先权框架；用一句 “variance was very small” 打发方差 |
| AStitch | 引言用一张证据图立前提；“Key Inefficiency:” 式 run-in claim；把无法运行的基线降级为案例研究并说明原因；把 overhead 当作一等结果 | 以 “novel / new space / long-overlooked gap” 为主的 novelty 框架；部署轶事当作验证；只有平均值没有离散度 |
| Korch | 评估小节内的 run-in bold 标题；把组件移植到外部系统做隔离实验；把调优/编译开销单列成表；在结果内部写一句 self-limitation | 把 “optimal” 写进标题与主张；贡献 bullets 完全不提测量设计；相关工作完全后置且按竞争系统组织 |

## 6. 结论

手稿的结构对“测量 + 预测 + 边界”这个题目是自洽的：协议、配对估计、等价边界、负结果、压力家族，构成一条完整的证据链。它不需要变成 DNNFusion/AStitch/Korch 那样的系统论文。

需要补的是**读者预期管理**：贡献段、roadmap、评估问题、两张表、与最近融合决策工作的划界。需要收缩的是 §7 与 §8：前者进附录，后者拆回结果与 threats。风格上，克制可以保留，但应把“我们发现了什么”放到每段第一句，把 “we” 从 2.6/k 提到 5–6/k，并加上三篇都在用的路标。这样既保留测量论文的可信度，也符合 ACM 读者的跳读习惯。
