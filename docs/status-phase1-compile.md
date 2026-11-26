# 编译/取证交付状态（phase 1）

版本：`torch 2.6.0+cu124`、`triton 3.2.0`（`requirements.lock`）。
本地验证机：RTX 4060 Laptop（sm_89）。**本文件中的本机结果只证明流程与取证，
不作为实验数据**；实验数据必须在 3090 Ti 上以同一命令重跑（见
[3090ti-runbook.md](3090ti-runbook.md)）。

## 1. 试点结果

`runs/pilot-4060-local/`（`run.json.purpose = local-smoke-4060`）：

| case_id | 策略 | 状态 | kernel 数 | t 物化 | 校验 |
|---|---|---|---|---|---|
| `P1-n65536-f32-contig-t` | default | valid | 1 | 否 | 4/4 |
| `P1-n65536-f32-contig-t` | split | valid | 2 | 是（`buf0`, fp32, 65536） | 4/4 |
| `P1-n65536-f32-contig-t` | fused | valid | 1 | 否 | 4/4 |
| `R1-m256-k512-f32-contig-t` | default | valid | 1 | 否 | 4/4 |
| `R1-m256-k512-f32-contig-t` | split | valid | 2 | 是（`buf0`, fp32, 256） | 4/4 |
| `R1-m256-k512-f32-contig-t` | fused | valid | 1 | 否 | 4/4 |

`verify` 重新加载（`CandidateStore`）后每个变体再校验 4 组输入；6/6 `cache_reused=true`。
`capture_replay()` 在两个试点的 split/fused 上各捕获一次并重放校验通过。

## 2. 三项证据（每个试点）

### P1 · split：t 在哪里写出、哪里读回

`artifacts/P1-n65536-f32-contig-t/split/output_code.py`（sha256 `915aca83f85c043f…`，
完整值见 `edge_evidence.json`）：

```
# kernel 0  (Source Nodes: [mul, t], ATen: [aten.mul, aten.add])
L30: tl.store(out_ptr0 + (x0), tmp4, None)              # t 写入 buf0
# kernel 1  (Source Nodes: [relu], ATen: [aten.relu])
L25: tmp0 = tl.load(in_out_ptr0 + (x0), None)           # t 从同一 buffer 读回
wrapper L132: triton_poi_fused_add_mul_0.run(arg0_1, buf0, 65536, grid=grid(65536), ...)
wrapper L137: triton_poi_fused_relu_1.run(buf1, 65536, grid=grid(65536), ...)
```

`buf1 = buf0; del buf0  # reuse`：分配器把 t 的存储交给消费者（`in_out_ptr0` 读改写）。
逻辑上仍是一次写 t + 一次读 t（`logical_save_bytes = 524288`），但不额外占显存。

### P1 · fused：哪段生成代码跨过了该边

单个 kernel `triton_poi_fused_add_mul_relu_0`，`Original ATen: [aten.mul, aten.add,
aten.relu]`，`grid(65536)`，读 `arg0_1`、写 `buf0`（输出）；不存在 t 形状的写→读链。
寄存器 14、warps 4、shared 0、`XBLOCK=512`（来自 `kernel_resources`，来源属性已记录）。

### R1 · split / fused

split：`triton_per_fused_sum_0`（persistent reduction，14 regs / 16 B shared，grid(256)）
`L35 tl.store(out_ptr0 + (x0), tmp3, None)` 写 `buf0`（256, fp32）；`triton_poi_fused_relu_1`
`L25 tl.load(in_out_ptr0 + (x0), xmask)` 读回。
fused：单个 `triton_per_fused_relu_sum_0`，`ATen: [aten.sum, aten.relu]`，即归约的 epilogue
跨边生成在同一 kernel 内。

### 默认策略的实际选择（包括选择发生在哪个 pass）

| case | default 分区 | `fusion_decision.stage` | 调度器考虑的配对数 | 说明 |
|---|---|---|---|---|
| P1 | 1 kernel | `lowering` | 0 | pointwise 链在 lowering 阶段内联，调度器从未被询问 |
| R1 | 1 kernel | `scheduler` | 2 | 归约与 relu 由调度器合并 |

判定来源：`torch/_inductor/scheduler.py` 的 `Scheduler.can_fuse_vertical` 被动计数 +
`output_code.py` 的最终 kernel 数，写入 `edge_evidence.json.fusion_decision` 与
`variants.jsonl.fusion_decision_stage`。Inductor 源文件路径与 sha256 记录在
`run.json.env.inductor_source`。

两个试点 `default_maps_to = "fused"`，且 `partition_equal_default_fused = true`
（按 kernel 名、`Source Nodes`、ATen 集合、读写 buffer、grid、源码哈希逐项比较）。
`split` 与 `fused` 的 `same_op_set_split_fused = true`、`unrelated_partition_change = false`、
`graph_count = 1`、`graph_break_count = 0`：切边没有引入额外图断裂或算子变化。

## 2b. 全量语料（本机冒烟，`runs/corpus-4060-local/`）

`check-run`：`cases=72`、`variant_rows=216`、`check_rows=864`、`statuses={valid:216}`、
`problems=[]`。分布：

| 统计 | 值 |
|---|---|
| `default_maps_to` | `fused` × 72 |
| split 的 kernel 数 | 2 × 72；`t_materialized=true` × 72 |
| fused/default 的 kernel 数 | 1 × 72 |
| `unrelated_partition_change` | false × 216 |
| `graph_break_count` | 0 × 216 |
| 数值校验 | 864/864 通过 |
| `fusion_decision_stage`（default/fused） | `lowering` × 24 case（P1/P2 的 pointwise 链，调度器 0 次询问）、`scheduler` × 48 case（R1/R2 归约边） |
| `fusion_decision_stage`（split） | `not_fused` × 72（各边都被切开） |

语料内没有出现多阶段（split）归约：`K=8192, M=16` 仍是单个 `triton_red_*` kernel
（grid = M，每行一个 program 循环 rnumel），所以 `shared_kernel_count` 没有可比较的公共
前置 kernel；这一点如实记录，不是校验失败。

## 3. 控制手段的副作用（`runs/diag-4060-local/`）

| 策略 | P1 | R1 | 结论 |
|---|---|---|---|
| `diag_no_fusion`（`max_fusion_size=0`） | 1 kernel，未切开 | 2 kernels，被切开 | 全局禁融合只作用于调度器阶段，对 lowering 内联无效；不是可选的边级控制 |
| `diag_clone`（消费者前 `.clone()`） | 1 kernel | 1 kernel | 克隆是 pointwise，仍被内联；不能用来切边 |
| `diag_graph_break`（`dynamo.graph_break()`） | 2 个编译图、1 次 graph break | 同 | 换来切边但多一次 Python 入口调用，违反"同一单图入口" |

同一张表也记录了"仅调度器否决无效"这一事实：P1 在 2.6 上 pointwise 链在 lowering 阶段
就已内联，`Scheduler.can_fuse_vertical` 的调用次数为 0。因此主 split 采用
"lowering `realize()` + 调度器按物化 buffer 否决再融合"，细节见
[design-decisions.md](design-decisions.md)。

## 4. 运行命令

```bash
# 0) 环境（3090 Ti 主机或开发机）
VENV=.venv bash scripts/setup_env.sh
export PYTHONPATH=src
# 非 Debian 系本机额外需要：LD_LIBRARY_PATH / TRITON_LIBCUDA_PATH / CC / CXX，见 runbook §6

# 1) 72 case 清单
.venv/bin/python -m fusionboundary manifest --run runs/pilot-3090ti

# 2) 两个试点（编译 + 数值校验 + 证据 + 重载校验）
bash scripts/run_pilot.sh                      # -> runs/pilot-<UTC>/

# 3) 全量 72 case（tmux 内）
bash scripts/run_corpus.sh                     # -> runs/corpus-<UTC>/

# 4) 诊断控制（单独归档）
bash scripts/run_diag.sh                       # -> runs/diag-<UTC>/

# 5) 状态表 / 重载校验 / 跨表自检
.venv/bin/python -m fusionboundary status --run runs/pilot-<UTC>
.venv/bin/python -m fusionboundary verify --run runs/pilot-<UTC> --pilot
.venv/bin/python -m fusionboundary check-run --run runs/pilot-<UTC>

# 6) 单条边证据（人读）
cat runs/pilot-<UTC>/evidence/pilot-evidence.md
```

## 5. 产物路径

```
runs/<run_id>/cases.jsonl                       72 行清单
runs/<run_id>/variants.jsonl                    每个 (case, strategy) 一行
runs/<run_id>/checks.jsonl                      每个 (case, strategy, input) 一行
runs/<run_id>/run.json                          环境/GPU/git/命令
runs/<run_id>/evidence/pilot-evidence.md        试点汇总证据
runs/<run_id>/artifacts/<case>/<strategy>/
    output_code.py                              生成的 wrapper + Triton kernel（冻结副本）
    trace/                                      Inductor 原始 trace 目录
    ir_pre_fusion.txt / ir_post_fusion.txt      调度前后 IR
    kernels.json                                算子→调度节点→kernel 关联表 + 静态资源
    edge_evidence.json                          切边判定（写/读行号、grid、资源、来源属性）
    evidence.md                                 上述内容的人读版
    compile_key.json                            配置实际取值/版本/缓存目录/产物哈希
    compile_result.json                         状态、首次调用耗时、notes、errors
    fx_graph.json                               图数量与 graph break 数
    checks.jsonl                                本变体数值校验
runs/<run_id>/artifacts/<case>/partition_map.json   三策略分区等价性与 default 映射
runs/<run_id>/cache/<case>/<strategy>/          隔离的 TORCHINDUCTOR_CACHE_DIR
runs/<run_id>/logs/<case>-<strategy>.log        该 worker 的完整日志
```

## 6. 待 3090 Ti 实测（本机不做结论）

- [ ] 两个试点的 `runs/` 产物（GPU 名、UUID、驱动、SM 数、L2 写入 `run.json`）
- [ ] 72 case 三策略状态表与 `partition_map.json` 分布（`default_maps_to` 计数）
- [ ] 多阶段归约（`M=4096, K=8192`）的 `shared_kernel_count` 与分区一致性
- [ ] fp16 归约样本（R1 累加 fp32、R2 保持 fp16）的 `fp64` 误差量级
- [ ] 诊断控制在该机上的同一张副作用表
- [ ] 与 B 的配对计时对齐（同一 `run_id` 的 `variants.jsonl` 被 `CandidateStore` 读取）
- [ ] 3090 Ti 上的 `capture_replay()` 校验与 B 的 replay 模式

## 7. 已知边界

- split 的"同一单图入口"成立：一个 `torch.compile` 产物、一次 Python 调用、wrapper 内
  两次 kernel 发射；`diag_graph_break` 的对照数据说明了违反该条时会多出编译图。
- 3.2.0 之前的 Triton 或 2.6 之外的 torch 未验证：`compile_key.json` 会因版本不符拒绝加载。
- 本机结果因 GPU 不同，`kernel_resources`（寄存器、shared、block）与 3090 Ti 可能不同；
  报告中的资源量必须以该机产物为准。
