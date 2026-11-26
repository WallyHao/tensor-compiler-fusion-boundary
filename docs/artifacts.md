# 产物与取证口径

`runs/<run_id>/` 遵循 `plan/phase1-4090/04-data.md`。本文件只说明"哪一份产物支撑哪一条
结论"，以及取证代码在 `src/fusionboundary/forensics.py` 中依赖的格式。

## 目录

```
runs/<run_id>/
  run.json                 purpose/命令/环境快照/GPU 快照/git 状态/时间戳
  cases.jsonl              72 行清单（corpus 生成，含 tolerance、logical_save_bytes）
  variants.jsonl           (case_id, strategy) 每次编译一行，含状态、kernel 列表、边判定
  checks.jsonl             (case_id, strategy, input_id, mode) 数值校验行
  cache/<case_id>/<strategy>/       每个策略独立 TORCHINDUCTOR_CACHE_DIR
  logs/<case_id>-<strategy>.log     该 worker 的完整 stdout/stderr
  artifacts/<case_id>/
      partition_map.json   三策略分区等价性与 default 映射
      <strategy>/
          trace/           Inductor 自己写的 trace 目录（原样保留）
              output_code.py      生成的 wrapper + Triton kernel 源码（主证据）
              ir_pre_fusion.*     调度前 IR dump
              ir_post_fusion.*    调度后 IR dump
              fx_graph*.*         FX 图（若该版本写出）
          kernels.json     逐 kernel 关联表（见下）
          edge_evidence.json  边判定的机器可读记录
          evidence.md      人读的取证摘要（可直接进报告）
          compile_key.json 编译键：shape/dtype/配置实际取值/版本/缓存目录/产物哈希
          compile_result.json 状态、首次调用耗时、notes、errors
          fx_graph.json    dynamo 级证据：图数量、graph break 数、FX 代码
          checks.jsonl     本变体的数值校验行
  evidence/pilot-evidence.md     两个试点的汇总证据（报告素材）
```

## 逐 kernel 关联表（`kernels.json`）

每项来自**生成代码本身**，不是配置推断：

| 字段 | 来源 |
|---|---|
| `launch_index` | `wrapper.call` 中 `.run(...)` 的出现顺序 |
| `name` | `async_compile.triton('name', ...)` |
| `source_nodes` / `original_aten` | kernel 前的 `# Source Nodes:` / `# Original Aten:` 注释 |
| `params` / `reads` / `writes` | kernel 形参 + wrapper 调用实参的按位对应；`out_ptr*` 为写，`in_ptr*`/`in_out_ptr*` 为读 |
| `grid` | wrapper 中 `grid=...` 表达式 |
| `grid_dims` | kernel 内 `tl.program_id(i)` 的最大维度 + 1 |
| `constexprs` | kernel 形参中被注解为 `tl.constexpr` 的名字 |
| `source_sha256` | kernel 源码哈希，用于跨策略比较分区是否被无关改动 |

两个容易混淆的字段：`t_allocated` 只表示"存在与 t 同 shape/dtype 的 buffer"（fused 时
输出 buffer 本身就满足），`t_materialized` 才表示"该 buffer 被某个 kernel 写出、又被另一个
kernel 读回"。进入配对分析的依据是后者。

`edge_evidence.json` 的判定规则：

* **split**：存在 buffer `B`，其 `empty_strided` 的 shape/dtype 等于 `t` 的
  `(t_shape, t_dtype)`；`B` 由 kernel *i* 通过 `tl.store(out_ptr… )` 写出，并由
  kernel *j>i* 通过 `tl.load(in_ptr…/in_out_ptr… )` 读回；`i` 的 `source_nodes/aten`
  命中模板的生产者算子集合，`j` 命中消费者集合。证据里给出两侧源码行号与 wrapper 行号。
  Inductor 常把 t 的存储直接交给消费者（wrapper 中 `buf1 = buf0; del buf0  # reuse`，
  消费者用 `in_out_ptr0` 读改写同一块），因此解析器先解析 wrapper 的别名赋值，再按
  分配名对齐；该情形记为 `matched_t_flows[*].in_place_read=true`，并给出 `aliases`。
* **fused**：某个 kernel 的 `source_nodes/aten` 同时命中生产者与消费者算子集合，且
  不存在符合 `t` shape/dtype 的中间 buffer 写入→读回链。
  注意"同一 kernel 读写同一 buffer"不算物化：persistent reduction 会把输出 buffer 作为
  `in_out_ptr0` 读改写（R1 的 fused 产物就是这种形态），若把它当成写→读链，每个 fused
  归约都会被误判成 split。判定函数只接受 **写者 kernel ≠ 读者 kernel** 的链。
* 两者都不成立 → `unclear`，状态记 `partition_mismatch`，不进入配对分析。

`default` 不预设结论：按上面两者自动判定，再把 `default` 的分区签名与 `split`/
`fused` 比较，写 `default_maps_to ∈ {fused, split, distinct}`。若默认形成第三种分区，
`partition_equal_default_*` 均为 `false`，`default_maps_to="distinct"`，该行仍保留但
从"默认二选一"评价中移出（05-analysis 的口径）。

## 无关分区变化的判定

`partition_map.json` 中：

* `same_op_set_split_fused`：两策略全部 kernel 的 `source_nodes ∪ original_aten` 集合
  是否相同。不相等说明控制手段引入了新的算子/图改写 → `unrelated_partition_change=true`。
* `shared_kernel_count`：源码哈希完全相同的 kernel 数。多阶段归约（如 R1 在
  `M=4096,K=8192` 上走 split reduction）应有共同的前置 kernel；若为 0 且
  `kernel_count>2`，说明非目标分区被改动。
* `graph_count` / `graph_break_count`（`fx_graph.json`）：split 若通过 dynamo
  graph break 实现，会显示 >1 个图与 >0 次断裂——这正是"额外 Python 调用"的证据，
  因此主 split 走调度器否决，不用 graph break。

## 备用取证通道

若某个 torch 版本的 trace 目录布局与 `forensics` 的期望不同（`edge_evidence.json` 的
`parse_notes` 会写出来），worker 日志里有同样的内容：driver 给每个 worker 设
`TORCH_LOGS=ir_pre_fusion,ir_post_fusion,output_code`，IR 与生成代码会写进
`logs/<case_id>-<strategy>.log`。两者都在 run 目录内，报告引用时写明来源。

## 可复算

```bash
# 从原始行重建状态表
python -m fusionboundary status --run runs/pilot-20261125

# 单条边证据（文本）
cat runs/pilot-20261125/artifacts/P1-n65536-f32-contig-t/split/evidence.md
```

`output_code.py` 与其 sha256 记在 `edge_evidence.json`；报告引用时写
`artifacts/<case_id>/<strategy>/trace/output_code.py` + 行号。
