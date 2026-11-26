# 设计决策与理由（编译/取证侧）

本文件记录实现选择、被否决的替代方案，以及每条选择对应的可复核证据。版本：
`torch==2.6.0`（CUDA 12.4 构建）、`triton 3.2.0`，见 `requirements.lock`。

## 1. split 的真实机制：lowering 请求物化 + 调度器否决该边再融合

实测（torch 2.6.0，P1/N=65536）：

| 手段 | 结果 |
|---|---|
| 仅调度器否决 | 无效。pointwise 链在 lowering 阶段就被内联成**一个** IR 节点，调度器只看到 `op0`，`Scheduler.can_fuse_vertical` 调用次数 = 0 |
| 仅 `.clone()`（`diag_clone`） | 无效。克隆本身是 pointwise，仍被内联进同一 kernel，分区不变 |
| `torch._dynamo.graph_break()`（`diag_graph_break`） | 能切 kernel，但产生两个编译图、两次 Python 入口调用，违反"同一单图入口" |
| **lowering `realize()` + 调度器否决** | 有效：2 个 kernel，t 物化为具名 buffer |

因此主 split 分两步，两步都在同一个 `torch.compile` 调用内完成：

1. **lowering 阶段请求物化**：hook `torch._inductor.graph.GraphLowering.run_node`。先从
   post-grad FX 图解析出边的两端（消费者 = 第一个命中模板消费者算子集合的 `call_function`
   节点，生产者 = 它的第一个 `call_function` 输入），当该生产者被 lower 时对其返回的
   `TensorBox` 调用 `realize()`，得到 t 的具名 buffer（如 `buf0`）。日志
   `realize_log: [{fx_node: "add", buffer: "buf0", status: "realized"}]`。
2. **调度器只阻止跨该边合并**：`Scheduler.can_fuse_vertical` 对 (producer, consumer) 返回
   `False`，且**以第 1 步得到的 buffer 名作为匹配锚**（`match="materialized_buffer"`），
   不用算子名（2.6 的调度节点叫 `op0`/`op1`，按算子名匹配会失效）。其余融合判断原样转发。

结果：同一张图、同一个 wrapper、同一次 Python 调用，wrapper 内部两次 kernel 发射，t 被写出
再读回。注意 Inductor 的 allocator 会把 t 的存储直接交给消费者（wrapper 里出现
`buf1 = buf0; del buf0  # reuse`，消费者 kernel 用 `in_out_ptr0` 读改写同一块），所以逻辑上
存在"写一次 t + 读一次 t"的流量，但不额外占显存——这一点对 x₂（名义传输时间）的解释很关键，
已记录为 `matched_t_flows[*].in_place_read`。

版本脆弱性由两道保险处理：

1. 门函数按候选表解析（`Scheduler.can_fuse_vertical` → `SchedulerNode.can_fuse_vertical`），
   实际符号写入 `compile_key.json.guard.symbol`；
2. 若既没有 realize 命中、也没有否决命中，或命中了却仍只有 1 个 kernel，driver 换下一个
   候选门、换独立缓存目录重编，并把每次尝试写进日志与 `variants.jsonl`。

## 2. 为什么三个策略各自独立编译、独立缓存

Inductor 的 FX-graph 缓存键不包含"运行时的 Python monkeypatch"。若 default/fused/split
共用缓存目录，第二次编译会直接命中第一次的产物，split 的否决根本不生效（等于静默失败）。
所以每个 (case, strategy) 使用独立 `TORCHINDUCTOR_CACHE_DIR`，并在 `compile_key.json`
里记录该目录与产物哈希。测量进程加载并执行校验过的归档 wrapper，避免 FX 缓存重新选取分区。
非计时 profiler 核对实际发射，文件快照用于诊断缓存变化。

default 与 fused 走同一条编译路径，但仍分别编译：分区是否等价由生成代码签名比较得出
（`partition_equal_default_fused`），而不是假设。默认若形成第三种分区，
`default_maps_to="distinct"`，该行保留但移出"默认二选一"评价。

## 3. 切边判定只看生成代码

判据来自 `output_code.py`（AST 解析）与 `ir_post_fusion`：

* split：存在 buffer `B`，shape/dtype 等于 t，由 kernel *i* 的 `tl.store(out_ptr…)` 写出、
  由 kernel *j>i* 的 `tl.load(in_ptr…)` 读回，且 *i*/*j* 的 `# Source Nodes`/`Original ATen`
  分别命中生产者/消费者算子集合；
* fused：某个 kernel 的算子集合同时命中两侧，且不存在 t 形状的写→读链。

不依赖"策略名"或配置位，因此控制手段若引入了无关改写（多/少算子），
`same_op_set_split_fused=false`、`unrelated_partition_change=true` 会被直接标出。

## 4. 容差与 fp64

容差严格按工作单表格：fp32 逐元素与 amax `(1e-6, 1e-5)`；fp32 累加求和 `(2e-4, 1e-4)`；
其余 fp16 `(2e-3, 2e-3)`。R1 的 t 恒为 fp32（`sum(..., dtype=float32)`），所以 fp16 输入也
走求和容差。

fp64 参考只作诊断：eager fp32 求和与 Inductor 的树形归约顺序不同，误差量级不同，用它做
门限会把"实现差异"误判成"错误"。`checks.jsonl` 里同时记 `max_abs_err`（对 eager）与
`fp64.max_abs_err`，并给出顺序累加的解析界 `8·eps·K` 以便解释量级。

## 5. 诊断控制（单独归档）

| 策略 | 期望现象 | 用途 |
|---|---|---|
| `diag_graph_break` | `graph_count>1`、`graph_break_count>0`，两个编译图 | 量化"用图断裂换切边"的代价：额外 Python 调用 |
| `diag_clone` | 分区通常不变（仍融合） | 证明显式拷贝不是可控切边手段 |
| `diag_no_fusion` | `max_fusion_size=0`：所有算子各自成 kernel，`same_op_set` 一般不变但分区全变 | 对照"全局禁融合"这种粗粒度控制 |

它们不进主配对表；`variants.jsonl` 里以 `diag_*` 列出，状态与原因照记。

## 6. 计时边界

主对照是稳态调用：`Candidate.call()` 只做 wrapper 自身的分配与 kernel 发射；编译、缓存
解析、Triton 预编译、数值校验全部在 `load()` 内完成并记录。`capture_replay()` 提供
CUDA Graph 机制对照（同一固定输入、同一输出 buffer），单独成表。
