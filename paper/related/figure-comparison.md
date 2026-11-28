# 数据绘图对照：我们的论文 vs DNNFusion / AStitch / Korch

对照对象：`paper/report.pdf` 的 7 张图、2 张表；三篇 ACM 论文的 PDF 页面（逐页渲染后阅读其图）。选这三篇的理由：它们都以“融合/编排”为主线，且图的类型覆盖了我们论文需要的三种证据——基线比较（DNNFusion）、融合空间与时间分解（AStitch）、决策与 kernel 映射（Korch）。

> 2026-11-28 更新：P1 已实现——新增 `cliff-mechanism`、`decision-quality`、`split-fused-latency`，重绘 `cliff-saving`、`pointwise-scaling`、`device-labels`，图题改为结论先行；正文现为 10 图、2 表。下文保留为修改前对照。

## 0. 结论

1. **我们的图在统计表达上强于三篇**：每张估计图都有配对 95% 区间、零线、±2% 等价带，颜色与标记同时编码模式；三篇基本没有误差条（AStitch 10 次取平均，DNNFusion 明确只报平均，Korch 单次）。
2. **我们的图在“用图讲机制、讲决策”上明显弱于三篇**。三篇的主图是**比较型**：1× 基线分组柱、时间分解堆叠柱、kernel 映射图、带数值标签的加速比；我们的主图是**差异型**：saving 点须图、矩阵、覆盖率堆叠条。
3. **两个硬缺陷已修**：device 图脚注原写 “Diagonal 72/72 (direct)”，实际对角线是 **71/72**（68 个 fuse–fuse + 3 个 equivalent–equivalent）；图用 “Inconclusive”、表用 “uncertain”，已统一为 “inconclusive”（表头 F/E/I/S）。
4. **三个最大缺口**：§6.2 的机制只有文字，没有图；§5.4/表 2 的决策评估没有图；全文没有一张 split vs fused 的绝对时延或时间分解图。
5. 正文当前 16 页，上限 18 页，**有 2 页空间**；加两张关键图（机制、决策）约 1 页，仍在限制内。

## 1. 三篇的图法

| 论文 | 图/表 | 代表图 | 视觉语法 |
|---|---|---|---|
| DNNFusion (PLDI'21) | 10 图 / 6 表 | Fig. 6 各模型相对 TASO 的加速比；Fig. 7 CPU/GPU 双面板分组柱 | 1× 基线、分组柱、模型作 x 轴、CPU/GPU 双面板；无误差条 |
| AStitch (ASPLOS'22) | 16 图 / 6 表 | Fig. 1 两个占比（执行时间、kernel 数）；Fig. 13 执行时间分解；Fig. 14–16 occupancy/SM efficiency | 100% 堆叠/分组柱、归一化分解、按 op 排序的折线；无误差条 |
| Korch (ASPLOS'24) | 13 图 / 2 表 / 1 算法 | Fig. 1 系统流水线；Fig. 6 2×5 分组柱；Fig. 12 kernel 映射 + 每 kernel 延迟 | 架构图带章节标注、柱顶数值标签、kernel 映射示意、逐 kernel 时间分解 |
| 本论文 | 7 图 / 2 表 | Fig. 3 20 行 saving 点须；Fig. 5 设备标签矩阵；Fig. 7 压力家族双面板 | 点估计+95% 区间、零线/±2% 带、矩阵计数、颜色+标记双编码 |

共同点：三篇的图都服务于“比较”——相对基线、相对组件、相对策略；图题短，面板有 (a)/(b) 标注，关键值常直接写在图上。我们的图服务于“差异是否显著”——这在测量论文里是对的，但只覆盖了问题的一半。

## 2. 具体缺陷

### D1. 机制没有图（最严重）

§6.2 的核心解释是：融合后 epilogue 继承归约的 launch 配置（`XBLOCK=1`、16 warps、grid=$M$），每增加一个 shifted read，融合 kernel 约增加 5 µs，而 split consumer 只增加约 1 µs；当 `num_load+num_reduction ≥ 10` 时 `RBLOCK` 从 2048 降到 1024/512、warps 从 16 降到 8；$B=6$ 还有寄存器路径（55 regs、occupancy 0.67）。这些全部是文字。

对照：Korch Fig. 12 把两个策略的 kernel 映射和每 kernel 毫秒数画在一起；AStitch Fig. 13 用堆叠柱分解执行时间，Fig. 15/16 用折线展示 occupancy/SM efficiency 随 op 的变化。我们的核心机制却没有对应图，读者只能相信文字。

**建议图**：`fig-mechanism`，2 面板：
- (a) per-kernel self time vs $B$：producer、split consumer、fused 三条线；标注斜率约 1 vs 5 µs/read。
- (b) `RBLOCK`（阶梯线）、warps（阶梯线）、occupancy（折线）vs $B$；在 $B=6,7,10$ 标出寄存器/计数器阈值。
数据：`paper/tables/cliff-results.csv` 已有 config/warps/regs/occupancy/num_load；per-kernel self time 在 run summary（与 `paper/evidence.md` 的 profiler 记录）。

### D2. 决策评估没有图

§5.4 与表 2 是全文第二个研究问题的全部证据；论文没有一张图展示预测与实测的关系。现有的 `policy-regret.svg` 已生成但未使用，且它是 3 根 ns 级柱，和表 2 的 µs 单位不一致。

对照：Korch Fig. 13 用两根水平柱直接标出 “1.25× slowdown / 2.88× speed-up”，把决策后果放在图上；Fig. 6 的柱顶有 2.7× 等数值。我们的表 2 信息完整，但不可扫读。

**建议图**：`fig-decision`，二选一：
- 预测 vs 实测 saving 散点：x 为实测 $B$，y 为模型预测 $B$，画 $y=x$、±2% 带、标出两个 FP16 错误点；或
- 按 cohort 的 regret CDF/柱：default、two-term、three-term、always-split；单位统一为 µs，标出 coverage。
数据：`case-results.csv` + prediction freeze 文件。

### D3. 没有绝对时延/加速比图

我们的 7 张图全部是 saving（差值）或其占比；没有一张画 split 与 fused 的绝对时间，也没有 1× 基线。读者无法从图上看出 “66.42 vs 38.42 µs” 这类实际量级。

对照：DNNFusion Fig. 6/7 以 1× 为基线画加速比；AStitch Fig. 11/12 是端到端加速比柱；Korch Fig. 7/12 直接画延迟。测量论文可以保留差值图，但应有一张“split vs fused 绝对时延”的图。

**建议图**：`fig-latency`，对 20 个 follow-up 配置或 4 个模板 × 2 模式画 dumbbell（split–fused 两点一线），或在同一图上画 speedup = split/fused 的柱，1× 线，关键点标数值。

### D4. 没有时间分解

AStitch Fig. 13 把执行时间拆成组件；Korch Fig. 12 把每个 kernel 的毫秒数写在映射图上。我们的图只给总 saving，无法看出“省下的是 launch、handoff 还是 compute”。

**建议**：在 `fig-mechanism` 或单独一张 `fig-breakdown` 中，对 $B=0,3,5,8$ 画堆叠柱：producer、split consumer、fused kernel、handoff；这比再画一条总 saving 线更能解释符号反转。

### D5. Figure 1 是流程图，不是编译器结构图

Korch Fig. 1 是带章节标注的流水线（fission → orchestration → executable），读者一眼看到“方法在哪一段”。我们的 Fig. 1 是 candidate → timing → prediction 的工作流，没有画出 lowering/scheduler/codegen，也没有标出 split/fused 在哪个 pass 分叉；而 §2 的核心论点正是“veto 加在哪一个 pass”。图的抽象层级与论文论点不匹配。

**建议**：把 Fig. 1 改成两层：上层是 Inductor pass 流水线（lowering → scheduler → codegen），标出 split 的 realize/veto 位置与 fused 的合并位置；下层保留现有测量流程。

### D6. 图题是描述，不是结论

我们的图题多为 3–5 行描述（“Markers show … whiskers show …”），把方法细节放在图题里；三篇的图题更短，常先给结论/比较对象，再给编码。例如 Korch Fig. 6 的图题是端到端性能比较，AStitch Fig. 1 的图题直接给出两个占比。

**建议**：图题第一句写 finding（如 “Replay halves the median saving, from 2.36 to 0.68 µs.”），第二句写编码与区间；把协议细节（30 blocks、10,000 bootstrap）压缩为一句或移到正文。

### D7. 压力家族图的图例过密、双编码不直观

`cliff-saving` 的图例有 7 项，字号 6.6 pt：5 个 $M$ 颜色 + filled/open 两种标记；数据同时用颜色编码 $M$、用填充编码 split/fuse。读者需要把两个图例叠加才能解码。图上也没有标出 crossover（$B$ 在 2 与 3 之间）或 split 区域。

对照：AStitch/Korch 的图例通常 2–5 项，关键值直接标在柱上；Korch 用 (a)/(b) 两个策略图，不需要双重图例。

**建议**：在 $M=512/1024$、$B\ge3$ 的区域加浅色底纹或边框，直接标注 “split-favored”；图例拆成两行或移入图题；字号提到 7.2 pt；关键点标 28.0 µs/72.9%。

### D8. Pointwise 图把绝对量注在相对轴上

`pointwise-scaling` 的 y 轴是 “Fusion saving (% of split latency)”，但图内标注 “≈160 µs/call” 是绝对时间；两个面板都标了同一句，容易让读者把 160 µs 当成 160%。三篇的图不会在同一轴上混用两种单位；它们要么分面板，要么把绝对时间放到第二 y 轴。

**建议**：加一个绝对 saving 的右轴或第三面板；或把 “≈160 µs” 移到图题，并在图上只保留百分比。

### D9. 设备标签图的两个硬问题（已修）

- 脚注原写 “Diagonal 72/72 (direct)”，实际是 **71/72**：68 个 fuse–fuse + 3 个 equivalent–equivalent，另有 1 个 3090 uncertain / 4090 equivalent 不在对角线上。replay 的对角线是 27/72（13 fuse–fuse + 4 equivalent–equivalent + 10 uncertain–uncertain），原值正确。
- 术语不统一：图用 “Inconclusive”，表 1 用 “uncertain”，正文两者混用。已把表 1 表头改为 F/E/I/S、图题改为 “inconclusive”，与图一致。

仍可改进：矩阵没有 colorbar，0–72 的线性色标被 68 的格子主导；计数已写在格子里，但读者仍需猜色阶。可加一条 0/72 的色标，或用分段色阶。

### D10. 已生成但未使用的图

`paper/figures/` 里有 `launch-calibration.svg`、`policy-regret.svg`、`threshold-sensitivity.svg`、`paired-trace.svg`，以及更早的 `layout-contrasts.svg`、`3090-followup.svg`；`report.typ` 一个都没有引用。它们不是“多余的好图”，而是需要重做才能用：

- `policy-regret` 用 ns，正文/表 2 用 µs；3 根柱没有区间，信息量低于表 2。
- `threshold-sensitivity` 与表 1 的 ±1/2/5% 计数重复；如果加图，应删表或改成图。
- `paired-trace` 是 30 个点的原始序列，没有聚合；更适合附录，不适合正文。
- `launch-calibration` 有信息（direct/replay 的每 kernel 斜率），但目前是 1×2 的小图，可并入 §4.1 或 §5.1。

**建议**：要么把 `policy-regret`/`launch-calibration` 重做成正文图，要么从构建脚本中移除，避免“生成了但没人用”的死资源。

### D11. 图的数量与信息密度低于三篇

7 图/16 页 vs 10/17、16/15、13/15。三篇每张图常含 2–5 个面板、5–15 个模型/策略；我们的图多为 1–2 个面板、4 个模板或 20 个配置。这不是说要多画图，而是说**该有图的地方没有图**：机制、决策、分解、基线比较。

### D12. 字号与数值标签

三篇的图例/刻度多在 7–8 pt，关键柱顶有数值标签；我们的 cliff 图例 6.6 pt、device 脚注 7.0 pt、pointwise 注释 6.9 pt，已接近印刷下限。至少把 cliff 图例和 device 脚注提到 7.2 pt；在关键点加数值（28.0 µs、72.9%、97.0 µs）。

### D13. 引言没有数据图

AStitch 用 Fig. 1 的两个占比（执行时间 63.2%、kernel 数 89.6%）在引言建立前提；我们的第一张图是工作流，第一张数据图要到 §5。测量论文不一定要照搬，但如果要在引言就让读者看到“边界存在”，一张小的 saving-vs-size 或标签分布图会比纯文字例子更有效。

## 3. 不应照搬的

| 三篇的做法 | 为什么不能照搬 |
|---|---|
| 只报平均值、无误差条 | 我们的配对区间是方法贡献；不要为了“像 ACM”退回去 |
| 1× 基线的柱状图作为唯一证据 | 柱状图掩盖配对结构与区间；可以作为补充，不应替代点须图 |
| 把“最优/首次”写进图题或标注 | 测量论文的主张受采样限制 |
| 用部署轶事/单一案例当验证 | 我们的证据链是配对测量与审计表 |

## 4. 建议的优先级

| 优先级 | 动作 | 预计页数 | 数据来源 |
|---|---|---|---|
| P0（已做） | 修正 device 对角线 71/72；统一 inconclusive；表头 F/E/I/S | 0 | `case-results.csv` |
| P1 | 新增 `fig-mechanism`（self time + RBLOCK/warps/occupancy vs B） | ≈0.5 页 | `cliff-results.csv` + profiler summary |
| P1 | 新增 `fig-decision`（预测 vs 实测，或 regret CDF） | ≈0.5 页 | `case-results.csv` + prediction freeze |
| P1 | 新增 `fig-latency`（split–fused dumbbell 或 speedup 柱，1× 线） | ≈0.4 页 | `case-results.csv` |
| P2 | 重画 Fig. 1 为 pass 流水线 + 测量流程 | ≈0.2 页 | 现有 `overview.svg` |
| P2 | 图题改为结论先行；cliff 加区域底纹/标注；pointwise 分单位 | 0（替换） | 现有图 |
| P2 | 处理未使用图：重做 `policy-regret`（µs）或删除；`threshold-sensitivity` 与表 1 二选一 | 0 | `build_figures.py` |

正文 16 页，加 2 张 P1 图约 17 页，仍在 18 页内；如果三张都加，需要压缩 §5.6/§5.7 或把 `fig-latency` 做成小面板。

## 5. 结论

我们的绘图系统在“统计诚实”上已经优于三篇：配对区间、等价带、色盲友好、标记/颜色双编码、精确表格。缺的不是绘图技巧，而是**用图承担论证**：三篇把“为什么快/为什么选这个策略”画出来，我们把机制和决策留在了文字和表格里。优先补 `fig-mechanism` 与 `fig-decision`，再补一张绝对时延/分解图；这三张图会把论文从“有数据的报告”变成“用图讲清楚边界的论文”。
